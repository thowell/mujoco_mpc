#ifndef MJPC_LEARNING_PPO_H_
#define MJPC_LEARNING_PPO_H_

#include <vector>

#include "mjpc/learning/adam.h"
#include "mjpc/learning/mlp.h"
#include "mjpc/threadpool.h"

namespace mjpc {

double Reward(const double* observation, const double* action);
void Dynamics(double* next_obs, const double* observation,
              const double* action);

// data for rollouts
class RolloutData {
 public:
  // constructor
  RolloutData() = default;

  // destructor
  ~RolloutData() = default;

  // initialize
  void Initialize(int dim_obs, int dim_action, int num_steps, int num_env);

  // memory
  std::vector<double> observation;  // observation
  std::vector<double> action;       // action
  std::vector<double> reward;       // reward
  std::vector<double> logprob;      // logarithmic probability of action
  std::vector<double> value;        // value
  std::vector<double> rewardtogo;   // reward-to-go
  std::vector<double> advantage;    // advantage;
  std::vector<int> done;            // done

  // dimensions
  int dim_obs;
  int dim_action;
  int num_steps;
  int num_env;
};

// ----- PPO ----- //
class PPO {
 public:
  // constructor
  PPO() {
    discount_factor = 0.99;
    GAE_factor = 0.95;
    clip = 0.2;
    value_coeff = 0.5;
    entropy_coeff = 0.01;
    gradient_norm_max = 0.5;
  };

  // destructor
  ~PPO() = default;

  // initialize
  void Initialize(int dim_obs, int dim_action, int num_steps, int num_env,
                  int dim_minibatch,
                  std::function<void(MLP& mlp)> actor_initialization,
                  std::function<void(MLP& mlp)> critic_initialization);

  // rollouts
  void Rollouts(const double* obs_init, ThreadPool& pool);

  // reward-to-go
  void RewardToGo();

  // advantage
  void Advantage();

  // normalize advantages by minibatch
  void NormalizeAdvantages();

  // get minibatch
  const int* Minibatch(int id);

  // sample action from policy
  void Policy(double* action, const double* observation, MLP& mlp);

  // logarithmic probability of action
  double LogProb(const double* action, const double* observation, MLP& mlp);

  // logarithmic probability of action gradient
  void LogProbGradient(double* gradient, const double* action,
                       const double* observation);

  // policy loss
  double PolicyLoss(const double* observation, const double* action,
                    double logprob, double advantage);

  // total policy loss over all experience
  double TotalPolicyLoss();

  // policy loss gradient
  void PolicyLossGradient(double* gradient, const double* observation,
                          const double* action, double logprob,
                          double advantage);

  // entropy loss 
  double EntropyLoss(const double* observation, const double* action);

  // total entropy loss 
  double TotalEntropyLoss();

  // entropy loss gradient
  void EntropyLossGradient(double* gradient, const double* observation, const double* action);

  // value
  double Value(const double* observation, MLP& mlp);

  // critic loss
  double CriticLoss(const double* observation, double rewardtogo);

  // total critic loss over all experience
  double TotalCriticLoss();

  // critic loss gradient
  void CriticLossGradient(double* gradient, const double* observation,
                          double rewardtogo);

  // learn
  void Learn(int iterations, ThreadPool& pool);

  // actor network
  MLP actor_mlp;
  std::vector<std::unique_ptr<MLP>> actors;
  AdamOptimizer actor_opt;

  // critic network
  MLP critic_mlp;
  std::vector<std::unique_ptr<MLP>> critics;
  AdamOptimizer critic_opt;

  // loss gradient
  std::vector<double> loss_gradient;

  // rollout data
  RolloutData data;

  // batch indices
  std::vector<int> batch_indices;
  int num_minibatch;

  // settings
  double discount_factor;
  double GAE_factor;
  double clip;
  double value_coeff;
  double entropy_coeff;
  double gradient_norm_max;
};

// finite difference gradient for scalar output functions
class FiniteDifferenceGradient {
 public:
  // contstructor
  FiniteDifferenceGradient() = default;

  // destructor
  ~FiniteDifferenceGradient() = default;

  // ----- methods ----- //

  // allocate memory, set function and settings
  void Allocate(std::function<double(const double*, int)> f, int n, double eps);

  // compute gradient
  void Gradient(const double* x);

  // ----- members ----- //
  std::function<double(const double*, int)> eval;
  std::vector<double> gradient;
  std::vector<double> workspace;
  int dimension;
  double epsilon;
};

}  // namespace mjpc

#endif  // MJPC_LEARNING_PPO_H_
