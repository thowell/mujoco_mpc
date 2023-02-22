#include "mjpc/learning/ppo.h"

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include <functional>

#include <algorithm>
#include <random>
#include <vector>

#include "mjpc/threadpool.h"

namespace mjpc {

// ----- test problem ----- //
// reward function
double Reward(const double* observation, const double* action) {
  double Q = 1.0;
  double R = 1.0e-3;
  return mju_exp(-1.0 * (0.5 * Q * observation[0] * observation[0] +
                         0.5 * Q * observation[1] * observation[1] +
                         0.5 * R * action[0] * action[0]));
}

// dynamics
void Dynamics(double* next_obs, const double* observation,
              const double* action) {
  next_obs[0] = observation[0] + 0.1 * observation[1];
  next_obs[1] = observation[1] + mju_max(mju_min(action[0], 1.0), -1.0);
}
// ----- test problem ----- //

// initialize
void RolloutData::Initialize(int dim_obs, int dim_action, int num_steps,
                             int num_env) {
  // total experience
  int experience = num_steps * num_env;

  // allocate memory
  observation.resize(dim_obs * experience);
  action.resize(dim_action * experience);
  reward.resize(experience);
  logprob.resize(experience);
  value.resize(experience);
  rewardtogo.resize(experience);
  advantage.resize(experience);
  done.resize(experience);

  // dimensions
  this->dim_obs = dim_obs;
  this->dim_action = dim_action;
  this->num_steps = num_steps;
  this->num_env = num_env;
}

// ----- PPO ----- //
void PPO::Initialize(int dim_obs, int dim_action, int num_steps, int num_env,
                     int dim_minibatch,
                     std::function<void(MLP& mlp)> actor_initialization,
                     std::function<void(MLP& mlp)> critic_initialization) {
  // data
  data.Initialize(dim_obs, dim_action, num_steps, num_env);

  // experience steps
  int experience = num_steps * num_env;

  // actor network
  actor_initialization(actor_mlp);
  // actor_mlp.Initialize(dim_obs, 2 * dim_action, dim_hidden, actor_activations);
  for (int i = 0; i < num_env; i++) {
    actors.emplace_back(new MLP);
    actor_initialization(*actors[i]);
    // actors[i]->Initialize(dim_obs, 2 * dim_action, dim_hidden, actor_activations);
  }

  // actor optimizer
  actor_opt.dim_batch = experience;
  actor_opt.dim_minibatch = dim_minibatch;
  actor_opt.epochs = 10;
  actor_opt.alpha = 3.0e-4;
  actor_opt.eps = 1.0e-5;
  actor_opt.Initialize(actor_mlp.num_parameters);

  // critic network
  // critic_mlp.Initialize(dim_obs, 1, dim_hidden, critic_activations);
  critic_initialization(critic_mlp);
  for (int i = 0; i < num_env; i++) {
    critics.emplace_back(new MLP);
    // critics[i]->Initialize(dim_obs, 1, dim_hidden, critic_activations);
    critic_initialization(*critics[i]);
  }

  // critic optimizer
  critic_opt.dim_batch = experience;
  critic_opt.dim_minibatch = dim_minibatch;
  critic_opt.epochs = 10;
  critic_opt.alpha = 1.0e-3;
  critic_opt.eps = 1.0e-5;
  critic_opt.Initialize(critic_mlp.num_parameters);

  // loss gradient
  loss_gradient.resize(actor_mlp.dim_layer[actor_mlp.dim_layer.size() - 1]);
  mju_zero(loss_gradient.data(), loss_gradient.size());

  // batch indices
  batch_indices.resize(experience);
  for (int i = 0; i < experience; i++) {
    batch_indices[i] = i;
  }

  // number of minibatches
  num_minibatch = std::floor(actor_opt.dim_batch / actor_opt.dim_minibatch);
}

// parallel rollouts
void PPO::Rollouts(const double* obs_init, ThreadPool& pool) {
  int count_before = pool.GetCount();
  for (int i = 0; i < data.num_env; i++) {
    pool.Schedule([&data = this->data, &obs_init, &ppo = *this, i]() {
      // copy parameters
      ppo.actors[i]->parameters = ppo.actor_mlp.parameters;
      ppo.critics[i]->parameters = ppo.critic_mlp.parameters;

      // collect experience
      for (int t = 0; t < data.num_steps; t++) {
        // shift by environment and time
        int obs_shift = i * data.dim_obs * data.num_steps + t * data.dim_obs;
        int action_shift =
            i * data.dim_action * data.num_steps + t * data.dim_action;
        int time_shift = i * data.num_steps + t;

        // observation reset
        if (t == 0 || (t > 0 && data.done[time_shift - 1] == 1)) {
          mju_copy(data.observation.data() + obs_shift, obs_init, data.dim_obs);
        }

        // value
        data.value[time_shift] =
            ppo.Value(data.observation.data() + obs_shift, *ppo.critics[i]);

        // evaluate policy
        ppo.Policy(data.action.data() + action_shift,
                   data.observation.data() + obs_shift, *ppo.actors[i]);

        // logprob
        data.logprob[time_shift] =
            ppo.LogProb(data.action.data() + action_shift,
                        data.observation.data() + obs_shift, *ppo.actors[i]);

        // reward
        data.reward[time_shift] = Reward(data.observation.data() + obs_shift,
                                         data.action.data() + action_shift);

        // step
        if (t == (data.num_steps - 1)) {
          data.done[time_shift] = 1;
        } else {
          data.done[time_shift] = 0;
          Dynamics(data.observation.data() + obs_shift + data.dim_obs,
                   data.observation.data() + obs_shift,
                   data.action.data() + action_shift);
        }
      }
    });
  }
  pool.WaitCount(count_before + data.num_env);
  pool.ResetCount();
}

// reward-to-go
void PPO::RewardToGo() {
  // experience
  int experience = data.num_steps * data.num_env;

  // initialize cumulative discounted reward
  double discounted = 0.0;

  // iterate backward through experience
  for (int i = experience - 1; i >= 0; i--) {
    if (data.done[i] == 1) {
      discounted = 0.0;  // reset
    }
    discounted = data.reward[i] + discount_factor * discounted;
    data.rewardtogo[i] = discounted;
  }
}

// advantage
void PPO::Advantage() {
  int experience = data.num_env * data.num_steps;
  // double next_value = 0.0;
  // double lastgaelam = 0.0;
  // double nextnonterminal = 0.0;
  // double nextvalues = 0.0;
  // double delta = 0.0;

  // mju_zero(data.advantage.data(), experience);
  // mju_zero(data.rewardtogo.data(), experience);

  // // advantage
  // for (int i = experience - 1; i >= 0; i--) {
  //   if (i == experience - 1) { 
  //     nextnonterminal = 1.0 - data.done[experience - 1];
  //     nextvalues = next_value;
  //   } else {
  //     nextnonterminal = 1.0 - data.done[i + 1];
  //     nextvalues = data.value[i + 1];
  //   }
  //   delta = data.reward[i] + discount_factor * nextvalues * nextnonterminal - data.value[i];
  //   lastgaelam = delta + discount_factor * GAE_factor * nextnonterminal * lastgaelam;
  //   data.advantage[i] = lastgaelam;
  // }
  
  // reward-to-go
  // mju_add(data.rewardtogo.data(), data.advantage.data(), data.value.data(), experience);

  mju_sub(data.advantage.data(), data.rewardtogo.data(), data.value.data(),
          experience);
}

// get minibatch
const int* PPO::Minibatch(int id) {
  return batch_indices.data() + actor_opt.dim_minibatch * id;
}

// normalize advantages by minibatch
void PPO::NormalizeAdvantages() {
  // loop over minibatches
  for (int k = 0; k < num_minibatch; k++) {
    // get minibatch
    const int* minibatch = this->Minibatch(k);

    // compute mean
    double mean = 0.0;
    for (int i = 0; i < actor_opt.dim_minibatch; i++) {
      mean += data.advantage[minibatch[i]];
    }
    mean = mean / actor_opt.dim_minibatch;

    // compute difference
    double std = 0.0;
    for (int i = 0; i < actor_opt.dim_minibatch; i++) {
      double diff = data.advantage[minibatch[i]] - mean;
      std += diff * diff;
    }
    std = mju_sqrt(std / actor_opt.dim_minibatch);

    // shift + scale
    for (int i = 0; i < actor_opt.dim_minibatch; i++) {
      data.advantage[minibatch[i]] =
          (data.advantage[minibatch[i]] - mean) / (std + 1.0e-8);
    }
  }
}

// sample action from policy
void PPO::Policy(double* action, const double* observation, MLP& mlp) {
  // evaluate forward network
  mlp.Forward(observation);

  // output 
  double* output = mlp.Output();

  // network output
  mju_copy(action, output, data.dim_action);

  // add Gaussian noise
  absl::BitGen gen_;

  // sample noise
  for (int k = 0; k < data.dim_action; k++) {
    double sigma = mju_exp(output[data.dim_action + k]);
    action[k] += sigma * absl::Gaussian<double>(gen_, 0.0, 1.0);
  }
}

// logarithmic probability of action
double PPO::LogProb(const double* action, const double* observation, MLP& mlp) {
  // evaluate forward network
  mlp.Forward(observation);

  // network output
  double* output = mlp.Output();

  // compute quadratic
  double sum = 0.0;
  for (int i = 0; i < data.dim_action; i++) {
    double diff = action[i] - output[i];
    double sigma = mju_exp(output[data.dim_action + i]);
    sum += diff * diff / (sigma * sigma) + 2.0 * mju_log(sigma);
  }

  // log likelihood
  return -0.5 * (sum + data.dim_action * mju_log(2.0 * mjPI));
}

// logarithmic probability of action
void PPO::LogProbGradient(double* gradient, const double* action,
                          const double* observation) {
  // evaluate forward network
  actor_mlp.Forward(observation);

  // network output
  double* output = actor_mlp.Output();

  // compute quadratic
  for (int i = 0; i < data.dim_action; i++) {
    double diff = action[i] - output[i];
    double sigma = mju_exp(output[data.dim_action + i]);
    gradient[i] = diff / (sigma * sigma);
    gradient[data.dim_action + i] = diff * diff / sigma / sigma - 1.0;
  }
}

// policy loss
double PPO::PolicyLoss(const double* observation, const double* action,
                       double logprob, double advantage) {
  // compute log probabilty for (action, observation) given new parameters
  double logprob_new = LogProb(action, observation, actor_mlp);

  // policy ratio
  double ratio = mju_exp(logprob_new - logprob);

  // losses
  double l1 = ratio * advantage;
  double l2 = mju_min(mju_max(ratio, 1.0 - clip), 1.0 + clip) * advantage;

  // final loss
  return -mju_min(l1, l2);
}

// total policy loss over all experience
double PPO::TotalPolicyLoss() {
  double loss = 0.0;

  for (int i = 0; i < data.num_steps * data.num_env; i++) {
    loss += this->PolicyLoss(data.observation.data() + i * data.dim_obs,
                             data.action.data() + i * data.dim_action,
                             data.logprob[i], data.advantage[i]);
  }

  return loss / data.num_steps / data.num_env;
}

// policy loss gradient
void PPO::PolicyLossGradient(double* gradient, const double* observation,
                             const double* action, double logprob,
                             double advantage) {
  // ----- finite difference ----- //

  // save current parameters
  std::vector<double> cache(actor_mlp.num_parameters);
  cache = actor_mlp.parameters;

  auto policy_loss = [&ppo = *this, &observation, &action, &logprob,
                      &advantage](const double* x, int n) {
    // copy parameters
    mju_copy(ppo.actor_mlp.parameters.data(), x, n);

    // evaluate forward network
    ppo.actor_mlp.Forward(observation);

    return ppo.PolicyLoss(observation, action, logprob, advantage);
  };

  mjpc::FiniteDifferenceGradient fd_pl;
  fd_pl.Allocate(policy_loss, actor_mlp.num_parameters, 1.0e-6);
  fd_pl.Gradient(actor_mlp.parameters.data());
  mju_addTo(gradient, fd_pl.gradient.data(), fd_pl.gradient.size());

  // restore current parameters
  actor_mlp.parameters = cache;

  return;

  // compute log probabilty for (action, observation) given new parameters
  // double logprob_new = this->LogProb(action, observation, actor_mlp);

  // // policy ratio
  // double ratio = mju_exp(logprob_new - logprob);

  // // // gradient
  // if (ratio < 1.0 + clip && ratio > 1.0 - clip) {
  //   mju_zero(loss_gradient.data(), loss_gradient.size());

  //   this->LogProbGradient(loss_gradient.data(), action, observation);
  //   mju_scl(loss_gradient.data(), loss_gradient.data(), -ratio * advantage,
  //           loss_gradient.size());

  //   actor_mlp.Backward(loss_gradient.data());

  //   mju_addTo(gradient, actor_mlp.gradient.data(), actor_mlp.gradient.size());
  // }

  // compute log probabilty for (action, observation) given new parameters
  // double logprob_new = this->LogProb(action, observation, actor_mlp);

  // // policy ratio
  // double ratio = mju_exp(logprob_new - logprob);

  // gradient
  // if (ratio < 1.0 + clip && ratio > 1.0 - clip) {
  //   auto lp = [&ppo = *this, &action, &logprob, &advantage](const double* x, int n) {

  //     // compute quadratic
  //     double sum = 0.0;
  //     for (int i = 0; i < ppo.data.dim_action; i++) {
  //       double diff = action[i] - x[i];
  //       double sigma = mju_exp(x[ppo.data.dim_action + i]);
  //       sum += diff * diff / (sigma * sigma) + 2.0 * mju_log(sigma);
  //     }

  //     // log likelihood
  //     return mju_exp(-0.5 * (sum + ppo.data.dim_action * mju_log(2.0 * mjPI)) - logprob) * advantage;
  //   };

  //   mjpc::FiniteDifferenceGradient fd_pl;
  //   fd_pl.Allocate(lp, actor_mlp.dim_layer[actor_mlp.dim_layer.size() - 1], 1.0e-6);
  //   fd_pl.Gradient(actor_mlp.Output());
    
  //   // this->LogProbGradient(fd_pl.gradient.data(), action, observation);
  //   mju_scl(fd_pl.gradient.data(), fd_pl.gradient.data(), -1.0,
  //           fd_pl.gradient.size());

  //   actor_mlp.Backward(fd_pl.gradient.data());

  //   mju_addTo(gradient, actor_mlp.gradient.data(), actor_mlp.gradient.size());
  // // }
  // zero gradient
}

// entropy loss 
double PPO::EntropyLoss(const double* observation, const double* action) {
  // compute log probabilty for (action, observation) given new parameters
  double logprob = LogProb(action, observation, actor_mlp);

  // compute prob
  double prob = mju_exp(logprob);

  return -prob * logprob;
}

// total entropy loss 
double PPO::TotalEntropyLoss() {
  double loss = 0.0;

  for (int i = 0; i < data.num_steps * data.num_env; i++) {
    loss += this->EntropyLoss(data.observation.data() + i * data.dim_obs,
                              data.action.data() + i * data.dim_action);
  }

  return loss / data.num_steps / data.num_env;
}

// entropy loss gradient
void PPO::EntropyLossGradient(double* gradient, const double* observation, const double* action) {
  // ----- finite difference ----- //

  // save current parameters
  std::vector<double> cache(actor_mlp.num_parameters);
  cache = actor_mlp.parameters;

  auto entropy_loss = [&ppo = *this, &observation, &action](const double* x, int n) {
    // copy parameters
    mju_copy(ppo.actor_mlp.parameters.data(), x, n);
    return ppo.EntropyLoss(observation, action);
  };

  mjpc::FiniteDifferenceGradient fd_pl;
  fd_pl.Allocate(entropy_loss, actor_mlp.num_parameters, 1.0e-6);
  fd_pl.Gradient(actor_mlp.parameters.data());
  mju_addTo(gradient, fd_pl.gradient.data(), fd_pl.gradient.size());

  // restore current parameters
  actor_mlp.parameters = cache;
}

// value
double PPO::Value(const double* observation, MLP& mlp) {
  // evaluate forward network
  mlp.Forward(observation);

  return mlp.Output()[0];
}

// critic loss
double PPO::CriticLoss(const double* observation, double rewardtogo) {
  // value
  double value = this->Value(observation, critic_mlp);

  // difference
  double diff = value - rewardtogo;

  // MSE
  return 0.5 * diff * diff;
}

// total critic loss over all experience
double PPO::TotalCriticLoss() {
  double loss = 0.0;

  for (int i = 0; i < data.num_steps * data.num_env; i++) {
    loss += this->CriticLoss(data.observation.data() + i * data.dim_obs,
                             data.rewardtogo[i]);
  }

  return loss / data.num_steps / data.num_env;
}

void PPO::CriticLossGradient(double* gradient, const double* observation,
                             double rewardtogo) {
  // backward
  critic_mlp.Forward(observation);
  double loss_gradient[1] = {critic_mlp.Output()[0] - rewardtogo};
  critic_mlp.Backward(loss_gradient);
  mju_addTo(gradient, critic_mlp.gradient.data(), critic_mlp.gradient.size());
}

// learn
void PPO::Learn(int iterations, ThreadPool& pool) {
  for (int k = 0; k < iterations; k++) {
    // rollouts
    std::vector<double> obs_init = {1.0, 0.0};
    this->Rollouts(obs_init.data(), pool);

    // reward-to-go
    this->RewardToGo();

    // average reward-to-go
    double avg_rtg = 0.0;
    for (int q = 0; q < data.num_env; q++) {
      avg_rtg += data.rewardtogo[q * data.num_steps];
    }
    avg_rtg /= data.num_env;

    printf("Iteration (%i) Average Reward-to-go: %f\n", k, avg_rtg);

    // advantage
    this->Advantage();

    // shuffle batch indices
    std::shuffle(batch_indices.begin(), batch_indices.end(),
                 std::default_random_engine(actor_opt.seed));

    // normalize advantages by minibatches
    this->NormalizeAdvantages();

    // learning epochs
    for (int i = 0; i < actor_opt.epochs; i++) {
      // ----- actor update ----- //
      for (int j = 0; j < num_minibatch; j++) {
        // minibatch
        const int* minibatch = this->Minibatch(j);

        // gradient
        mju_zero(actor_opt.g.data(), actor_opt.g.size());

        // --- entropy loss gradient --- //
        if (entropy_coeff > 0.0) {
          for (int q = 0; q < actor_opt.dim_minibatch; q++) {
            // entropy loss
            this->EntropyLossGradient(
                actor_opt.g.data(), actor_opt.g.data(),
                data.observation.data() + minibatch[q] * data.dim_obs);
          }

          // scaling by entropy coefficient
          mju_scl(actor_opt.g.data(), actor_opt.g.data(),
                -entropy_coeff, actor_opt.g.size());
        }
        
        // --- policy loss gradient --- //
        for (int q = 0; q < actor_opt.dim_minibatch; q++) {
          // policy loss
          this->PolicyLossGradient(
              actor_opt.g.data(),
              data.observation.data() + minibatch[q] * data.dim_obs,
              data.action.data() + minibatch[q] * data.dim_action,
              data.logprob[minibatch[q]], data.advantage[minibatch[q]]);
        }

        // scale by minibatch size
        mju_scl(actor_opt.g.data(), actor_opt.g.data(),
                1.0 / actor_opt.dim_minibatch, actor_opt.g.size());

        // gradient clipping
        double gradient_norm = mju_norm(actor_opt.g.data(), actor_opt.g.size());
        if (gradient_norm > gradient_norm_max) {
          mju_scl(actor_opt.g.data(), actor_opt.g.data(), gradient_norm_max / gradient_norm, actor_opt.g.size());
        }

        // update
        actor_opt.Step(actor_mlp.parameters);
      }

      // ----- critic update ----- //
      for (int j = 0; j < num_minibatch; j++) {
        // minibatch
        const int* minibatch = this->Minibatch(j);

        // gradient
        mju_zero(critic_opt.g.data(), critic_opt.g.size());

        // --- critic loss gradient --- //
        for (int q = 0; q < critic_opt.dim_minibatch; q++) {
          this->CriticLossGradient(
              critic_opt.g.data(),
              data.observation.data() + minibatch[q] * data.dim_obs,
              data.rewardtogo[minibatch[q]]);
        }

        // scale by minibatch size
        mju_scl(critic_opt.g.data(), critic_opt.g.data(),
                value_coeff / critic_opt.dim_minibatch, critic_opt.g.size());

        // gradient clipping
        double gradient_norm = mju_norm(critic_opt.g.data(), critic_opt.g.size());
        if (gradient_norm > gradient_norm_max) {
          mju_scl(critic_opt.g.data(), critic_opt.g.data(), gradient_norm_max / gradient_norm, critic_opt.g.size());
        }

        // update
        critic_opt.Step(critic_mlp.parameters);
      }
    }
  }
}

// allocate memory
void FiniteDifferenceGradient::Allocate(
    std::function<double(const double*, int)> f, int n, double eps) {
  // dimension
  dimension = n;

  // epsilon
  epsilon = eps;

  // evaluation function
  eval = f;

  // gradient
  gradient.resize(n);

  // workspace
  workspace.resize(n);
}

// compute gradient
void FiniteDifferenceGradient::Gradient(const double* x) {
  // set workspace
  mju_copy(workspace.data(), x, dimension);

  // centered finite difference
  for (int i = 0; i < dimension; i++) {
    // positive
    workspace[i] += 0.5 * epsilon;
    double fp = eval(workspace.data(), dimension);

    // negative
    workspace[i] -= 1.0 * epsilon;
    double fn = eval(workspace.data(), dimension);
    gradient[i] = (fp - fn) / epsilon;

    // reset
    workspace[i] = x[i];
  }
}

}  // namespace mjpc