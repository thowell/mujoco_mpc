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

#include <absl/random/random.h>
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
namespace mju = ::mujoco::util_mjpc;

// initialize data and settings
void DirectPlanner::Initialize(mjModel* model, const Task& task) {
  // model
  this->model = model;

  // task
  this->task = &task;

  // direct optimizer
  direct.Initialize(model);

  // MPC settings
  direct.settings.max_search_iterations = 3;
  direct.settings.max_smoother_iterations = 1;

  // trajectory
  trajectory.Initialize(model->nq + model->nv + model->na, model->nu,
                        task.num_residual, task.num_trace,
                        kMaxTrajectoryHorizon);
  buffer.Initialize(model->nq + model->nv + model->na, model->nu,
                    task.num_residual, task.num_trace, kMaxTrajectoryHorizon);
}

// allocate memory
void DirectPlanner::Allocate(){
  // direct optimizer
  direct.Allocate();

  // trajectory
  trajectory.Allocate(kMaxTrajectoryHorizon);
  buffer.Allocate(kMaxTrajectoryHorizon);
}

// reset memory to zeros
void DirectPlanner::Reset(int horizon, const double* initial_repeated_action){
  // direct optimizer
  direct.Reset();

  // random initialization
  std::vector<double> perturb(direct.model->nv);
  for (int t = 0; t < horizon; t++) {
    // initialize with default qpos0
    double* qpos = direct.qpos.Get(t);
    mju_copy(qpos, direct.model->qpos0, direct.model->nq);

    // random
    absl::BitGen gen_;
    for (int i = 0; i < direct.model->nv; i++) {
      perturb[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    mj_integratePos(direct.model, qpos, perturb.data(), 1.0e-1);
  }
  
  // trajectory
  trajectory.Reset(horizon);
  buffer.Reset(horizon);
}

// set state
void DirectPlanner::SetState(const State& state){
  // pin qpos0, qpos1
  direct.pinned[0] = true;
  direct.pinned[1] = true;

  // state
  int nq = direct.model->nq;
  const double* qpos = state.state().data();
  const double* qvel = state.state().data() + direct.model->nq;

  // set qpos1
  mju_copy(direct.qpos.Get(1), qpos, nq);
  
  // set qpos0
  double* qpos0 = direct.qpos.Get(0);
  mju_copy(qpos0, qpos, nq);
  mj_integratePos(direct.model, qpos0, qvel, -1.0 * direct.model->opt.timestep);

  // set time
  time = state.time();

  // set mocap
  for (int i = 0; i < direct.data_.size(); i++) {
    for (int j = 0; j < model->nmocap; j++) {
      mju_copy(direct.data_[i].get()->mocap_pos + 3 * j,
               DataAt(state.mocap(), 7 * j), 3);
      mju_copy(direct.data_[i].get()->mocap_quat + 4 * j,
               DataAt(state.mocap(), 7 * j + 3), 4);
    }
  }
}

// optimize nominal policy
void DirectPlanner::OptimizePolicy(int horizon, ThreadPool& pool){
  // resample qpos trajectory
  for (int t = 0; t < horizon + 2; t++) {
    // set time
    direct.time.Get(t)[0] = time + (t - 1) * direct.model->opt.timestep;

    // skip pinned qpos
    if (t <= 1) continue; 

    // interpolate qpos
    int bounds[2];
    FindInterval(bounds, trajectory.times, direct.time.Get(t)[0], trajectory.horizon);
    if (bounds[0] == bounds[1]) {
      mju_copy(direct.qpos.Get(t),
                DataAt(trajectory.states, trajectory.dim_state * bounds[0]),
                direct.model->nq);
    } else {
      double time_norm =
          (direct.time.Get(t)[0] - trajectory.times[bounds[0]]) /
          (trajectory.times[bounds[1]] - trajectory.times[bounds[0]]);
      InterpolateConfiguration(
          direct.qpos.Get(t), direct.model, time_norm,
          DataAt(trajectory.states, trajectory.dim_state * bounds[0]),
          DataAt(trajectory.states, trajectory.dim_state * bounds[1]));
    }
  }

  // set sensor weights
  // TODO(taylor): thread safe
  mju_copy(direct.weight_sensor.data(), task->weight.data(),
           direct.weight_sensor.size());

  // set force weights
  
  // optimize qpos trajectory
  direct.Optimize(horizon + 2);

  // update trajectory
  buffer.horizon = horizon;

  // memory
  std::vector<double> Mf(direct.model->nu);
  std::vector<double> MMT(direct.model->nu * direct.model->nu);
  std::vector<double> ctrl(direct.model->nu);

  // dimensions
  int nv = direct.model->nv;
  int nu = direct.model->nu;

  for (int t = 1; t < direct.qpos_horizon_ - 1; t++) {
    buffer.times[t - 1] = direct.time.Get(t)[0];
    mju_copy(buffer.states.data() + buffer.dim_state * (t - 1),
              direct.qpos.Get(t), direct.model->nq);

    // -- recover ctrl -- //
    // actuator_moment
    double* actuator_moment = direct.data_[t].get()->actuator_moment;

    // actuator_moment * qfrc_inverse
    mju_mulMatVec(Mf.data(), actuator_moment, direct.force.Get(t), nu, nv);

    // actuator_moment * actuator_moment'
    mju_mulMatMatT(MMT.data(), actuator_moment, actuator_moment, nu, nv, nu);

    // factorize
    int rank = mju_cholFactor(MMT.data(), nu, 0.0);
    if (rank < nu) {
      printf("Cholesky failure\n");
    }

    // gain * ctrl = (M M') * M * f
    mju_cholSolve(ctrl.data(), MMT.data(), Mf.data(), nu);

    // divide by gains to recover ctrl
    for (int i = 0; i < nu; i++) {
      double gain = direct.model->actuator_gainprm[mjNGAIN * i];
      ctrl[i] /= gain;
    }

    // set ctrl
    mju_copy(buffer.actions.data() + buffer.dim_action * (t - 1), ctrl.data(),
              nu);

    // set residual terms
    mju_copy(buffer.residual.data() + buffer.dim_residual * (t - 1),
              direct.sensor.Get(t), buffer.dim_residual);
  }

  // TODO(taylor): thread safe copy
  trajectory = buffer;
}

// compute trajectory using nominal policy
void DirectPlanner::NominalTrajectory(int horizon, ThreadPool& pool){}

// set action from policy
void DirectPlanner::ActionFromPolicy(double* action, const double* state,
                                     double time, bool use_previous){
                                      // find times bounds
  int bounds[2];
  FindInterval(bounds, trajectory.times, time, trajectory.horizon);

  // ----- get action ----- //

  // if (bounds[0] == bounds[1] ||
  //     representation == PolicyRepresentation::kZeroSpline) {
  ZeroInterpolation(action, time, trajectory.times, trajectory.actions.data(),
                    direct.model->nu, trajectory.horizon);
  // } else if (representation == PolicyRepresentation::kLinearSpline) {
  //   LinearInterpolation(action, time, times, parameters.data(), model->nu,
  //                       num_spline_points);
  // } else if (representation == PolicyRepresentation::kCubicSpline) {
  //   CubicInterpolation(action, time, times, parameters.data(), model->nu,
  //                      num_spline_points);
  // }

  // Clamp controls
  Clamp(action, direct.model->actuator_ctrlrange, direct.model->nu);
}

// return trajectory with best total return, or nullptr if no planning
// iteration has completed
const Trajectory* DirectPlanner::BestTrajectory() { return &trajectory; }

// visualize planner-specific traces
void DirectPlanner::Traces(mjvScene* scn){}

// planner-specific GUI elements
void DirectPlanner::GUI(mjUI& ui){
  // -- force weights -- //
  int shift = 0;
  mjuiDef defForceWeight[128];

  // separator
  defForceWeight[0] = {mjITEM_SEPARATOR, "Force Weight", 1};
  shift++;

  // loop over sensors
  int nv = direct.model->nv;
  std::string str;
  for (int i = 0; i < nv; i++) {
    // element
    defForceWeight[shift] = {
        mjITEM_SLIDERNUM, "", 2,
        direct.weight_force.data() + shift - 1, "1.0e-6 1000.0"};

    // add element index
    str = "DOF ";
    str += " (" + std::to_string(i) + ")";

    // set sensor name
    mju::strcpy_arr(defForceWeight[shift].name,
                    str.c_str());

    // shift
    shift++;
  }

  // end
  defForceWeight[shift] = {mjITEM_END};

  // add sensor noise
  mjui_add(&ui, defForceWeight);
}

// planner-specific plots
void DirectPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                          int planner_shift, int timer_shift, int planning,
                          int* shift){}

// return number of parameters optimized by planner
int DirectPlanner::NumParameters() {
  if (direct.model) return direct.qpos_horizon_ * direct.model->nv;
  return 0;
}

}  // namespace mjpc
