// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ostream>
#include <vector>
#include <absl/flags/parse.h>
#include <absl/random/random.h>

#include "mjpc/app.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/tasks.h"

#include "mjpc/learning/adam.h"
#include "mjpc/learning/mlp.h"
#include "mjpc/learning/ppo.h"

// machinery for replacing command line error by a macOS dialog box
// when running under Rosetta
// #if defined(__APPLE__) && defined(__AVX__)
// extern void DisplayErrorDialogBox(const char* title, const char* msg);
// static const char* rosetta_error_msg = nullptr;
// __attribute__((used, visibility("default")))
// extern "C" void _mj_rosettaError(const char* msg) {
//   rosetta_error_msg = msg;
// }
// #endif

// // run event loop
// int main(int argc, char** argv) {
//   // display an error if running on macOS under Rosetta 2
// #if defined(__APPLE__) && defined(__AVX__)
//   if (rosetta_error_msg) {
//     DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
//     std::exit(1);
//   }
// #endif
//   absl::ParseCommandLine(argc, argv);

//   mjpc::StartApp(mjpc::GetTasks(), 3);  // start with humanoid stand
//   return 0;
// }

// double f(const double* x) {
//   return 0.5 * mju_dot(x, x, 2);
// }

// void fx(double* gradient, const double* parameters, const int* minibatch) {
//   mju_copy(gradient, parameters, 2);
// }


// Test PPO and utilities
int main(int argc, char** argv) {
  // printf("Adam tests\n");
  // // adam 
  // mjpc::AdamOptimizer opt;
  // opt.epochs = 1500;
  // opt.dim_batch = 4;
  // opt.dim_minibatch = 2;

  // std::vector<double> x = {1.0, -1.0};
  // opt.Initialize(2);
  
  // opt.Optimize(x, fx);
  // printf("gradient:\n");
  // mju_printMat(x.data(), 1, 2);

  // printf("Adam tests complete\n");

  // printf("MLP tests\n");
  // mjpc::MLP mlp;
  // int dim_input = 4;
  // int dim_output = 1;
  // std::vector<int> dim_hidden = {4, 4};
  // std::vector<mjpc::Activations> activations = {mjpc::kTanh, mjpc::kTanh, mjpc::kPassThrough};
  // mlp.Initialize(dim_input, dim_output, dim_hidden, activations);
  // std::fill(mlp.parameters.begin(), mlp.parameters.end(), 1.0);
  // std::vector<double> input = {-1.0, -2.0, -3.0, 4.0};
  // std::vector<double> output = {0.0};
  // mlp.Forward(output.data(), input.data());
  // mlp.Backward(output.data());

  // printf("num_layer: %i\n", (int)mlp.dim_layer.size());

  // printf("input: \n");
  // mju_printMat(input.data(), 1, 4);

  // printf("output: \n");
  // mju_printMat(output.data(), 1, 1);

  // printf("gradient: \n");
  // mju_printMat(mlp.gradient.data(), mlp.num_parameters, 1);

  // printf("delta: \n");
  // mju_printMat(mlp.delta.data(), (int)mlp.delta.size(), 1);

  // printf("parameters: \n");
  // mju_printMat(mlp.parameters.data(), 1, mlp.num_parameters);

  // printf("MLP tests complete\n");


  // printf("PPO tests\n");
  // problem setup
  int dim_obs = 2;
  int dim_action = 1;
  int num_steps = 25;
  int num_env = 64;
  int dim_minibatch = 8; // num_env * num_steps; // num_env * num_steps;

  // initialization
  auto actor_initialization = [&dim_obs, &dim_action](mjpc::MLP& mlp) {
    std::vector<int> dim_hidden = {2, 2};
    std::vector<mjpc::Activations> activations = {mjpc::kTanh, mjpc::kTanh,
                                                  mjpc::kPassThrough};

    // setup
    mlp.Initialize(dim_obs, 2 * dim_action, dim_hidden, activations);

    // parameter initialization
    absl::BitGen gen_;
    for (int i = 0; i < 3; i++) {
      // weight
      double* Wi = mlp.Weight(i);
      for (int j = 0; j < mlp.dim_layer[i + 1] * mlp.dim_layer[i]; j++) {
        Wi[j] = absl::Gaussian<double>(gen_, 0.0, 1.0) / mju_sqrt(mlp.dim_layer[i]);
        if (i == 3) {
          Wi[j] *= 0.01;
        }
      }

      // bias
      double* bi = mlp.Bias(i);
      mju_zero(bi, mlp.dim_layer[i + 1]);
    }
  };

  // initialization
  auto critic_initialization = [&dim_obs](mjpc::MLP& mlp) {
    std::vector<int> dim_hidden = {2, 2};
    std::vector<mjpc::Activations> activations = {mjpc::kTanh, mjpc::kTanh,
                                                  mjpc::kPassThrough};

    // setup
    mlp.Initialize(dim_obs, 1, dim_hidden, activations);

    // parameter initialization
    absl::BitGen gen_;
    for (int i = 0; i < 3; i++) {
      // weight
      double* Wi = mlp.Weight(i);
      for (int j = 0; j < mlp.dim_layer[i + 1] * mlp.dim_layer[i]; j++) {
        Wi[j] = absl::Gaussian<double>(gen_, 0.0, 1.0) / mju_sqrt(mlp.dim_layer[i]);
      }

      // bias
      double* bi = mlp.Bias(i);
      mju_zero(bi, mlp.dim_layer[i + 1]);
    }
  };

  mjpc::ThreadPool pool(1);
  mjpc::PPO ppo;
  ppo.Initialize(dim_obs, dim_action, num_steps, num_env, dim_minibatch,
                 actor_initialization, critic_initialization);

  // double obs_init[2] = {1.0, 0.0};
  // ppo.Rollouts(obs_init, pool);
  // ppo.RewardToGo();
  // ppo.Advantage();
  ppo.Learn(100, pool);

  // printf("actor t = %i\n", ppo.actor_opt.t);
  // printf("critic t = %i\n", ppo.critic_opt.t);

  // printf("actor grad\n");
  // mju_printMat(ppo.actor_opt.g.data(), 1, ppo.actor_opt.g.size());

  // printf("critic grad\n");
  // mju_printMat(ppo.critic_opt.g.data(), 1, ppo.critic_opt.g.size());

  // printf("actor m\n");
  // mju_printMat(ppo.actor_opt.m.data(), 1, ppo.actor_opt.m.size());

  // printf("critic m\n");
  // mju_printMat(ppo.critic_opt.m.data(), 1, ppo.critic_opt.m.size());

  // printf("actor v\n");
  // mju_printMat(ppo.actor_opt.v.data(), 1, ppo.actor_opt.v.size());

  // printf("critic v\n");
  // mju_printMat(ppo.critic_opt.v.data(), 1, ppo.critic_opt.v.size());

  // printf("actor m_hat\n");
  // mju_printMat(ppo.actor_opt.m_hat.data(), 1, ppo.actor_opt.m_hat.size());

  // printf("critic m_hat\n");
  // mju_printMat(ppo.critic_opt.m_hat.data(), 1, ppo.critic_opt.m_hat.size());

  // printf("actor v_hat\n");
  // mju_printMat(ppo.actor_opt.v_hat.data(), 1, ppo.actor_opt.v_hat.size());

  // printf("critic v_hat\n");
  // mju_printMat(ppo.critic_opt.v_hat.data(), 1, ppo.critic_opt.v_hat.size());

  // policy loss gradient
  // double observation[2] = {1.2, 0.75};
  // double action[1] = {0.24};
  // double logprob = 0.24;
  // double advantage = 0.73;
  // double rewardtogo = 1.23;

  // // int i = 2;
  // mju_zero(ppo.actor_opt.g.data(), ppo.actor_opt.g.size());
  // // ppo.PolicyLossGradient(ppo.actor_opt.g.data(),
  // //                        ppo.data.observation.data() + i * ppo.data.dim_obs,
  // //                        ppo.data.action.data() + i * ppo.data.dim_action,
  // //                        ppo.data.logprob[i], ppo.data.advantage[i]);
  // ppo.PolicyLossGradient(ppo.actor_opt.g.data(),
  //                        observation,
  //                        action,
  //                        logprob, advantage);

  // printf("pl analytical gradient\n");
  // mju_printMat(ppo.actor_opt.g.data(), 1, ppo.actor_opt.g.size());

  // std::vector<double> policy_parameters;
  // policy_parameters = ppo.actor_mlp.parameters;

  // // auto policy_loss = [&ppo, i](const double* x, int n) {
  // //   mju_copy(ppo.actor_mlp.parameters.data(), x, n);
  // //   return ppo.PolicyLoss(ppo.data.observation.data() + i * ppo.data.dim_obs,
  // //                         ppo.data.action.data() + i * ppo.data.dim_action,
  // //                         ppo.data.logprob[i], ppo.data.advantage[i]);
  // // };
  // auto policy_loss = [&ppo, &observation, &action, &logprob, &advantage](
  //                        const double* x, int n) {
  //   mju_copy(ppo.actor_mlp.parameters.data(), x, n);
  //   return ppo.PolicyLoss(observation, action, logprob, advantage);
  // };

  // mjpc::FiniteDifferenceGradient fd_pl;
  // fd_pl.Allocate(policy_loss, ppo.actor_mlp.num_parameters, 1.0e-6);
  // fd_pl.Gradient(policy_parameters.data());

  // printf("pl finite difference gradient\n");
  // mju_printMat(fd_pl.gradient.data(), 1, fd_pl.gradient.size());

  // // critic loss gradient 
  // mju_zero(ppo.critic_opt.g.data(), ppo.critic_opt.g.size());
  // // ppo.CriticLossGradient(ppo.critic_opt.g.data(),
  // //                        ppo.data.observation.data() + i * ppo.data.dim_obs,
  // //                        ppo.data.rewardtogo[i]);
  // ppo.CriticLossGradient(ppo.critic_opt.g.data(), observation, rewardtogo);

  // printf("cl analytical gradient\n");
  // mju_printMat(ppo.critic_opt.g.data(), 1, ppo.critic_opt.g.size());

  // std::vector<double> critic_parameters;
  // critic_parameters = ppo.critic_mlp.parameters;

  // auto critic_loss = [&ppo, i](const double* x, int n) {
  //   mju_copy(ppo.critic_mlp.parameters.data(), x, n);
  //   return ppo.CriticLoss(ppo.data.observation.data() + i * ppo.data.dim_obs,
  //                         ppo.data.rewardtogo[i]);
  // };
  // auto critic_loss = [&ppo, &observation, &rewardtogo](const double* x, int n) {
  //   mju_copy(ppo.critic_mlp.parameters.data(), x, n);
  //   return ppo.CriticLoss(observation, rewardtogo);
  // };

  // mjpc::FiniteDifferenceGradient fd_cl;
  // fd_cl.Allocate(critic_loss, ppo.critic_mlp.num_parameters, 1.0e-6);
  // fd_cl.Gradient(critic_parameters.data());

  // printf("cl finite difference gradient\n");
  // mju_printMat(fd_cl.gradient.data(), 1, fd_cl.gradient.size());

  // printf("total policy loss = %f\n", ppo.TotalPolicyLoss());
  // printf("total critic loss = %f\n", ppo.TotalCriticLoss());

  // ppo.Learn(20, pool);
  // printf("dim_batch: %i\n", ppo.actor_opt.dim_batch);
  // printf("dim_minibatch: %i\n", ppo.actor_opt.dim_minibatch);

  // std::vector<double> obs_init = {2.0, 0.5};
  // std::vector<double> action_init = {1.0};

  // // evaluate forward network
  // ppo.actor_mlp.Forward(obs_init.data());

  // // network output
  // double* mean = ppo.actor_mlp.layer_output[ppo.actor_mlp.OutputIndex(ppo.actor_mlp.dim_layer.size() - 2)];
  // printf("mean: %f \n", *mean);
  // mju_printMat(mean, 1, data.dim_action);

  // printf("MLP output:\n");
  // mju_printMat(ppo.actor_mlp.layer_output.data(), 1, ppo.actor_mlp.layer_output.size());
  // mju_printMat(ppo.actor_mlp.layer_output.data() + ppo.actor_mlp.OutputIndex(ppo.actor_mlp.dim_layer.size() - 2), 1, ppo.actor_mlp.dim_layer[ppo.actor_mlp.dim_layer.size() - 1]);

  // printf("output index: %i\n", ppo.actor_mlp.OutputIndex(ppo.actor_mlp.dim_layer.size() - 2));

  // double logprob = ppo.PolicyLogProb(obs_init.data(), action_init.data());
  // printf("logprob: %f\n", logprob);


  // printf("pdf: %f\n", ppo.ProbabilityDensity(action_init.data(), action_init.data(),
  //                        ppo.exploration_noise, ppo.data.dim_action));


  // printf("layer output");
  // printf("output size: %i\n", (int)ppo.actor_mlp.layer_output.size());
  // printf("output shift: %i\n", ppo.actor_mlp.OutputIndex(ppo.actor_mlp.dim_layer.size() - 2));
  // printf("mean: %f\n", ppo.actor_mlp.layer_output[ppo.actor_mlp.OutputIndex(ppo.actor_mlp.dim_layer.size() - 2)]);
  // double mean[1];
  // mju_copy(mean, ppo.actor_mlp.layer_output.data() + ppo.actor_mlp.OutputIndex(ppo.actor_mlp.dim_layer.size() - 2), 1);
  // printf("mean2: %f\n", mean[0]);

  // mju_printMat(ppo.actor_mlp.layer_output.data(), 1, ppo.actor_mlp.layer_output.size());
  // mju_printMat(ppo.actor_mlp.layer_output.data(), 1, ppo.actor_mlp.layer_output.size());

  // double lp = ppo.PolicyLogProb(action_init.data(), obs_init.data());
  // printf("lp: %f\n", lp);

  // std::vector<double> grad_;
  // grad_.resize(ppo.data.dim_action);

  // ppo.ProbabilityDensityGradient(grad_.data(), action_init.data(),
                                    //  ppo.actor_mlp.layer_output.data() + ppo.actor_mlp.OutputIndex(ppo.actor_mlp.dim_layer.size() - 2), 0.1,
                                    //  ppo.data.dim_action);

  // printf("lp grad\n");
  // mju_printMat(grad_.data(), 1, ppo.data.dim_action);
  // printf("mean: \n");
  // mju_printMat(mean, 1, 1);
  // // Gaussian probability
  // double pdf = ppo.ProbabilityDensity(action_init.data(), mean, 0.1,
  //                                     ppo.data.dim_action);
  // printf("pdf: %f\n", pdf);

  // std::vector<double> grad_ = {0.0};
  // ppo.ProbabilityDensityGradient(grad_.data(), action_init.data(), mean, 0.1,
  //                                     ppo.data.dim_action);
  
  // printf("pdf_grad: \n");
  // mju_printMat(grad_.data(), 1, 1);

  // double pl = ppo.PolicyLoss(obs_init.data(), action_init.data(), 1.0, 1.0);
  // printf("pl: %f\n", pl);

  // std::vector<double> grad_policy;
  // grad_policy.resize(ppo.actor_mlp.num_parameters);
  // mju_zero(grad_policy.data(), grad_policy.size());

  // ppo.PolicyLossGradient(grad_policy.data(), obs_init.data(), action_init.data(), 1.0, 1.0);

  // printf("pl_grad:\n");
  // mju_printMat(grad_policy.data(), 1, ppo.actor_mlp.num_parameters);

  // grad_policy.resize(ppo.critic_mlp.num_parameters);
  // mju_zero(grad_policy.data(), grad_policy.size());
  // ppo.CriticLossGradient(grad_policy.data(), obs_init.data(), 1.0);
  // printf("critic loss grad\n");
  // mju_printMat(grad_policy.data(), 1, ppo.critic_mlp.num_parameters);

  // printf("critic parameters: \n");
  // mju_printMat(ppo.critic_mlp.parameters.data(), 1, ppo.critic_mlp.parameters.size());


  // double cl = ppo.CriticLoss(obs_init.data(), 1.0);
  // printf("cl: %f\n", cl);

  // ppo.Rollouts(obs_init.data(), pool);
  // ppo.RewardToGo();
  // ppo.Advantage();

  // printf("observation: \n");
  // for (int i = 0; i < ppo.data.num_env; i++) {
  //   mju_printMat(ppo.data.observation.data() + i * ppo.data.num_steps * ppo.data.dim_obs, ppo.data.num_steps, ppo.data.dim_obs);
  // }
  // printf("action: \n");
  // for (int i = 0; i < ppo.data.num_env; i++) {
  //   mju_printMat(ppo.data.action.data() + i * ppo.data.num_steps * ppo.data.dim_action, ppo.data.num_steps, ppo.data.dim_action);
  // }
  // printf("logprob: \n");
  // for (int i = 0; i < ppo.data.num_env; i++) {
  //   mju_printMat(ppo.data.logprob.data() + i * ppo.data.num_steps, 1, ppo.data.num_steps);
  // }
  // printf("reward-to-go: \n");
  // for (int i = 0; i < ppo.data.num_env; i++) {
  //   mju_printMat(ppo.data.rewardtogo.data() + i * ppo.data.num_steps, 1, ppo.data.num_steps);
  // }
  // printf("value: \n");
  // for (int i = 0; i < ppo.data.num_env; i++) {
  //   mju_printMat(ppo.data.value.data() + i * ppo.data.num_steps, 1, ppo.data.num_steps);
  // }
  // printf("advantage: \n");
  // for (int i = 0; i < ppo.data.num_env; i++) {
  //   mju_printMat(ppo.data.advantage.data() + i * ppo.data.num_steps, 1, ppo.data.num_steps);
  // }
  // const int minibatch[3] = {0, 1, 2};
  // ppo.NormalizeMinibatch(ppo.data.advantage.data(), minibatch, num_steps);
  // printf("normalized advantage: \n");
  // mju_printMat(ppo.data.advantage.data(), ppo.data.num_steps, ppo.data.num_env);

  // int t = 2;
  // printf("policy loss: %f\n",
  //        ppo.PolicyLoss(ppo.data.observation.data() + t * ppo.data.dim_obs, ppo.data.action.data() + t * ppo.data.dim_action,
  //                       ppo.data.logprob[t], ppo.data.advantage[t]));

  // t = 2;
  // printf("critic loss: %f\n", ppo.CriticLoss(ppo.data.observation.data() + t * ppo.data.dim_obs, ppo.data.rewardtogo[t]));


  // double x[2] = {2.0, 0.5};
  // double u[1] = {0.25};
  // std::vector<double> grad_;
  // grad_.resize(ppo.actor_mlp.num_parameters);
  // mju_zero(grad_.data(), ppo.actor_mlp.num_parameters);

  // printf("check gradients\n");

  // printf("policy loss: %f\n",
  //        ppo.PolicyLoss(x, u, 0.1, 0.5));

  // ppo.PolicyLossGradient(grad_.data(), x, u, 0.1, 0.5);
  // printf("policy loss gradient: \n");
  // mju_printMat(grad_.data(), 1, ppo.actor_mlp.num_parameters);

  // std::vector<double> grad__;
  // grad__.resize(ppo.critic_mlp.num_parameters);
  // mju_zero(grad__.data(), ppo.critic_mlp.num_parameters);

  // printf("critic loss: %f\n",
  //        ppo.CriticLoss(x, 0.75));

  // ppo.CriticLossGradient(grad__.data(), x, 0.75);
  // printf("critic loss gradient: \n");
  // mju_printMat(grad__.data(), 1, ppo.critic_mlp.num_parameters);

  // printf("actor gradient:\n");
  // mju_printMat(ppo.actor_opt.g.data(), 1, ppo.actor_opt.g.size());

  // printf("critic gradient:\n");
  // mju_printMat(ppo.critic_opt.g.data(), 1, ppo.critic_opt.g.size());

  // printf("actor minibatch: %i\n", ppo.actor_opt.dim_minibatch);
  // printf("critic minibatch: %i\n", ppo.critic_opt.dim_minibatch);

  // std::vector<double> obs_next = {0.0, 0.0};
  // mjpc::Dynamics(obs_next.data(), obs_init.data(), action_init.data());
  // printf("obs_next: \n");
  // mju_printMat(obs_next.data(), 1, 2);

  // printf("reward: %f\n", mjpc::Reward(obs_init.data(),
  // action_init.data()));

  // printf("PPO tests complete\n");

  // printf("log(2): %f\n", mju_log(2.0));
  
  return 0;
}
