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

#include <atomic>
#include <shared_mutex>
#include <vector>

#include <mujoco/mujoco.h>
#include "planners/planner.h"
#include "planners/cma/policy.h"
#include "states/state.h"
#include "trajectory.h"

namespace mjpc {

// cma planner limits
inline constexpr int MinCMASplinePoints = 1;
inline constexpr int MaxCMASplinePoints = 36;
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
  void NominalTrajectory(int horizon) override;

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
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
             int planning) override;

  // ----- members ----- //
  mjModel* model;
  const Task* task;

  // state
  std::vector<double> state;
  double time;
  std::vector<double> mocap;

  // policy
  CMAPolicy policy; // (Guarded by mtx_)
  CMAPolicy candidate_policy[kMaxTrajectory];

  // scratch
  std::vector<double> parameters_scratch;
  std::vector<double> times_scratch;

  // trajectories
  Trajectory trajectories[kMaxTrajectory];

  // rollout parameters
  double timestep_power;

  // ----- noise ----- //
  double noise_exploration;  // standard deviation for cma normal: N(0,
                             // exploration)
  std::vector<double> noise;

  // best trajectory
  int winner;

  // improvement
  double improvement;

  // flags
  int processed_noise_status;

  // timing
  double nominal_compute_time;
  std::atomic<double> noise_compute_time;
  double rollouts_compute_time;
  double policy_update_compute_time;

 private:
  int num_trajectories_;
  mutable std::shared_mutex mtx_;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_CMA_OPTIMIZER_H_
