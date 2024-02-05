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

#include "mjpc/planners/direct/planner.h"

#include <algorithm>
#include <chrono>
#include <shared_mutex>

#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

// initialize data and settings
void DirectPlanner::Initialize(mjModel* model, const Task& task) {
  // task
  this->task = &task;
};

// allocate memory
void DirectPlanner::Allocate(){};

// reset memory to zeros
void DirectPlanner::Reset(int horizon, const double* initial_repeated_action){};

// set state
void DirectPlanner::SetState(const State& state){};

// optimize nominal policy
void DirectPlanner::OptimizePolicy(int horizon, ThreadPool& pool){};

// compute trajectory using nominal policy
void DirectPlanner::NominalTrajectory(int horizon, ThreadPool& pool){};

// set action from policy
void DirectPlanner::ActionFromPolicy(double* action, const double* state,
                                     double time, bool use_previous){};

// return trajectory with best total return, or nullptr if no planning
// iteration has completed
const Trajectory* DirectPlanner::BestTrajectory() { return NULL; };

// visualize planner-specific traces
void DirectPlanner::Traces(mjvScene* scn){};

// planner-specific GUI elements
void DirectPlanner::GUI(mjUI& ui){};

// planner-specific plots
void DirectPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                          int planner_shift, int timer_shift, int planning,
                          int* shift){};

// return number of parameters optimized by planner
int DirectPlanner::NumParameters() { return 0; };

}  // namespace mjpc
