// Copyright 2022 DeepMind Technologies Limited
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

#ifndef MJPC_PLANNERS_CMA_OPTIMIZER_H_
#define MJPC_PLANNERS_CMA_OPTIMIZER_H_

#include <mujoco/mujoco.h>

#include <atomic>
#include <shared_mutex>
#include <vector>

#include "planners/cma/policy.h"
#include "planners/planner.h"
#include "states/state.h"
#include "trajectory.h"

namespace mjpc {

// cma planner limits
inline constexpr int MinCMASplinePower = 1;
inline constexpr int MaxCMASplinePower = 5;
inline constexpr double MinCMANoiseStdDev = 0.0;
inline constexpr double MaxCMANoiseStdDev = 1.0;

class CMAPlanner : public Planner {
 public:
  // constructor
  CMAPlanner() = default;

  // destructor
  ~CMAPlanner() override = default;

  // ----- methods ----- //

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  // allocate memory
  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon) override;

  // set state
  void SetState(State& state) override;

  // optimize nominal policy using random cma
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override;

  // set action from policy
  void ActionFromPolicy(double* action, const double* state,
                        double time) override;

  // resample nominal policy
  void ResamplePolicy(int horizon);

  // add noise to nominal policy
  void AddNoiseToPolicy(int i);

  // compute candidate trajectories
  void Rollouts(int num_trajectories, int horizon, ThreadPool& pool);

  // return trajectory with best total return
  const Trajectory* BestTrajectory() override;

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override;

  // planner-specific GUI elements
  void GUI(mjUI& ui) override;

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning) override;

  // ----- members ----- //
  mjModel* model;
  const Task* task;

  // state
  std::vector<double> state;
  double time;
  std::vector<double> mocap;
  std::vector<double> userdata;

  // policy
  CMAPolicy policy;  // (Guarded by mtx_)
  CMAPolicy candidate_policy[kMaxTrajectory];

  // scratch
  std::vector<double> parameters_scratch;
  std::vector<double> times_scratch;

  // trajectories
  Trajectory trajectories[kMaxTrajectory];

  // rollout parameters
  double timestep_power;

  // ----- noise ----- //
  std::vector<double> noise;

  // best trajectory
  int winner;

  // improvement
  double improvement;

  // timing
  double nominal_compute_time;
  std::atomic<double> noise_compute_time;
  double rollouts_compute_time;
  double policy_update_compute_time;

  // CMA-ES
  int num_elite;
  double step_size;
  double mu_eff;
  double c_sigma;
  double d_sigma;
  double c_Sigma;
  double c1;
  double c_mu;
  double E;
  double eps;

  std::vector<double> p_sigma;
  std::vector<double> p_sigma_tmp;
  std::vector<double> p_Sigma;
  std::vector<double> Sigma;
  std::vector<double> Sigma_tmp;
  std::vector<double> covariance;
  std::vector<double> covariance_lower;
  std::vector<double> B;
  std::vector<double> C;
  std::vector<double> C_tmp;
  std::vector<double> D;
  std::vector<double> fitness;
  std::vector<int> fitness_sort;
  std::vector<double> sample;
  std::vector<double> delta_s;
  std::vector<double> delta_w;
  std::vector<double> weight;
  std::vector<double> weight_update;
  std::vector<double> C_delta_s;

 private:
  int num_trajectories_;
  mutable std::shared_mutex mtx_;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_CMA_OPTIMIZER_H_
