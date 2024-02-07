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

#ifndef MJPC_PLANNERS_DIRECT_PLANNER_H_
#define MJPC_PLANNERS_DIRECT_PLANNER_H_

#include <mujoco/mujoco.h>

#include "mjpc/direct/trajectory.h"
#include "mjpc/direct_planner/direct.h"
#include "mjpc/planners/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

// inverse dynamics direct planner
class DirectPlanner: public Planner {
 public:
  // constructor
  DirectPlanner() = default;

  // destructor
  // ~DirectPlanner() {
  //   if (model) mj_deleteModel(model);
  // }

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  // allocate memory
  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override;

  // set state
  void SetState(const State& state) override;

  // optimize nominal policy
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override;

  // set action from policy
  void ActionFromPolicy(double* action, const double* state, double time,
                        bool use_previous = false) override;

  // return trajectory with best total return, or nullptr if no planning
  // iteration has completed
  const Trajectory* BestTrajectory() override;

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override;

  // planner-specific GUI elements
  void GUI(mjUI& ui) override;

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // return number of parameters optimized by planner
  int NumParameters() override;

  // model
  mjModel* model;

  // task
  const Task* task;

  // direct optimizer
  Direct2 direct;

  // trajectory
  Trajectory policy;
  Trajectory trajectory;
  Trajectory nominal;

  // state
  std::vector<double> state;
  double time;
  std::vector<double> mocap;
  std::vector<double> userdata;

  // sensor scaling
  double sensor_scaling = 1.0;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_DIRECT_PLANNER_H_
