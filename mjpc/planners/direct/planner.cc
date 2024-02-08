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
  direct.settings.max_search_iterations = 10;
  direct.settings.max_smoother_iterations = 1;

  // trajectory
  policy.Initialize(model->nq + model->nv + model->na, model->nu,
                    task.num_residual, task.num_trace, kMaxTrajectoryHorizon);
  trajectory.Initialize(model->nq + model->nv + model->na, model->nu,
                        task.num_residual, task.num_trace,
                        kMaxTrajectoryHorizon);
  nominal.Initialize(model->nq + model->nv + model->na, model->nu,
                     task.num_residual, task.num_trace, kMaxTrajectoryHorizon);
}

// allocate memory
void DirectPlanner::Allocate(){
  // direct optimizer
  direct.Allocate();

  // trajectory
  policy.Allocate(kMaxTrajectoryHorizon);
  trajectory.Allocate(kMaxTrajectoryHorizon);
  nominal.Allocate(kMaxTrajectoryHorizon);

  // state
  state.resize(model->nq + model->nv + model->na);
  mocap.resize(7 * model->nmocap);
  userdata.resize(model->nuserdata);
}

// reset memory to zeros
void DirectPlanner::Reset(int horizon, const double* initial_repeated_action){
  // direct optimizer
  direct.Reset();

  // trajectory
  policy.Reset(horizon);
  trajectory.Reset(horizon);
  nominal.Reset(horizon);

  // sensor scaling
  sensor_scaling = GetNumberOrDefault(1.0, model, "direct_sensor_scale");

  // absl::BitGen gen_;
  // for (int t = 0; t < horizon; t++) {
  //   for (int i = 0; i < model->nu; i++) {
  //     policy.actions[t * model->nu + i] =
  //         absl::Gaussian<double>(gen_, 0.0, 1.0e-3);
  //   }
  //   Clamp(policy.actions.data() + t * model->nu, model->actuator_ctrlrange,
  //         model->nu);
  // }

  // random initialization
  // std::vector<double> perturb(direct.model->nv);
  // for (int t = 0; t < horizon; t++) {
  //   // initialize with default qpos0
  //   double* qpos = direct.qpos.Get(t);
  //   mju_copy(qpos, state.data(), direct.model->nq);

  //   // random
  //   absl::BitGen gen_;
  //   for (int i = 0; i < direct.model->nv; i++) {
  //     perturb[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  //   }
  //   mj_integratePos(direct.model, qpos, perturb.data(), 1.0e-3);
  // }
  
  
}

// set state
void DirectPlanner::SetState(const State& state){
  // cache state
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);

  // pin qpos0, qpos1
  direct.pinned[0] = true;
  direct.pinned[1] = true;
  // for (int t = 0; t < direct.qpos_horizon_; t++) {
  //   direct.pinned[t] = true;
  // }

  // state
  int nq = direct.model->nq;
  const double* qpos = state.state().data();
  const double* qvel = state.state().data() + direct.model->nq;

  // set qpos1
  mju_copy(direct.qpos.Get(1), qpos, nq);
  direct.time.Get(1)[0] = time;
  
  // set qpos0
  double* qpos0 = direct.qpos.Get(0);
  mju_copy(qpos0, qpos, nq);
  mj_integratePos(direct.model, qpos0, qvel, -1.0 * direct.model->opt.timestep);
  direct.time.Get(0)[0] = time - direct.model->opt.timestep;

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
  // nominal trajectory
  NominalTrajectory(horizon + 1, direct.pool_);

  // initialize direct optimizer with nominal trajectory
  for (int t = 2; t < horizon + 2; t++) {
    // set time
    direct.time.Get(t)[0] = nominal.times[t - 1];

    // set qpos
    double* qpos = nominal.states.data() + nominal.dim_state * (t - 1);
    direct.qpos.Set(qpos, t);

    // int bounds[2];
    // FindInterval(bounds, trajectory.times, direct.time.Get(t)[0],
    //              trajectory.horizon);
    // printf("t = %i: [%i, %i]\n", t, bounds[0], bounds[1]);
    // if (bounds[0] == bounds[1]) {
    // mju_copy(direct.qpos.Get(t),
    //           DataAt(trajectory.states, trajectory.dim_state * bounds[0]),
    //           direct.model->nq);
    // } else {
    //   double time_norm =
    //       (direct.time.Get(t)[0] - trajectory.times[bounds[0]]) /
    //       (trajectory.times[bounds[1]] - trajectory.times[bounds[0]]);
    //   InterpolateConfiguration(
    //       direct.qpos.Get(t), direct.model, time_norm,
    //       DataAt(trajectory.states, trajectory.dim_state * bounds[0]),
    //       DataAt(trajectory.states, trajectory.dim_state * bounds[1]));
    // }
  }

  // set sensor weights
  // TODO(taylor): thread safe
  mju_scl(direct.weight_sensor.data(), task->weight.data(), sensor_scaling,
          direct.weight_sensor.size());
  // std::fill(direct.weight_sensor.begin(), direct.weight_force.end(), 0.0);
  // printf("weight sensor:\n");
  // mju_printMat(direct.weight_sensor.data(), 1, direct.weight_sensor.size());

  // printf("weight force:\n");
  // mju_printMat(direct.weight_force.data(), 1, direct.weight_force.size());

  // // set force weights

  // // for (int t = 0; t < direct.qpos_horizon_; t++) {
  // //   mju_copy(direct.qpos.Get(t), direct.qpos.Get(1), direct.model->nq);
  // // }
  // // printf("nominal qpos:\n");
  // // for (int t = 0; t < horizon; t++) {
  // //   mju_printMat(nominal.states.data() + nominal.dim_state * t, 1, model->nq);
  // // }

  // printf("qpos (initialization)\n");
  // mju_printMat(direct.qpos.Data(), horizon + 2, direct.model->nq);

  // printf("time: %f\n", time);

  // printf("times: \n");
  // mju_printMat(direct.time.Data(), 1, horizon + 2);

  // optimize qpos trajectory
  // direct.settings.verbose_optimize = true;
  direct.Optimize(horizon + 2);

  // printf("qpos (optimized)\n");
  // mju_printMat(direct.qpos.Data(), horizon + 2, model->nq);

  // printf("qvel (optimized)\n");
  // mju_printMat(direct.qvel.Data(), horizon + 2, model->nv);

  // printf("qacc (optimized)\n");
  // mju_printMat(direct.qacc.Data(), horizon + 2, model->nv);

  // printf("force (optimized)\n");
  // mju_printMat(direct.force.Data(), horizon + 2, model->nv);

  // mju_error("check policy: %i\n", horizon + 2);

  // printf("qpos (optimized)\n");
  // mju_printMat(direct.qpos.Data(), direct.qpos_horizon_, direct.model->nq);

  // printf("sensor (optimized)\n");
  // mju_printMat(direct.sensor.Data(), direct.qpos_horizon_, direct.nsensordata_);

  // printf("force (optimized)\n");
  // mju_printMat(direct.force.Data(), direct.qpos_horizon_, direct.model->nv);

  // for (int t = 0; t < direct.qpos_horizon_; t++) {
  //   // check magnitude
  //   if (mju_norm(direct.qpos.Get(t), direct.model->nq) > 100.0) {
  //     mju_error("qpos %i: large magnitude (optimize)\n", t);
  //   }
  // }

  // memory
  std::vector<double> Mf(direct.model->nu);
  std::vector<double> MMT(direct.model->nu * direct.model->nu);
  std::vector<double> ctrl(direct.model->nu);

  // dimensions
  int nv = direct.model->nv;
  int nu = direct.model->nu;

  // printf("times (optimized): \n");
  // mju_printMat(direct.time.Data(), 1, horizon);

  // printf("times (trajectory PRE): \n");
  // mju_printMat(trajectory.times.data(), 1, horizon);

  trajectory.horizon = horizon;
  for (int t = 0; t < horizon; t++) {
    trajectory.times[t] = direct.time.Get(t + 1)[0];
    mju_copy(trajectory.states.data() + trajectory.dim_state * t,
             direct.qpos.Get(t + 1), direct.model->nq);

    // -- recover ctrl -- //
    // actuator_moment
    double* actuator_moment = direct.data_[t + 1].get()->actuator_moment;

    // actuator_moment * qfrc_inverse
    mju_mulMatVec(Mf.data(), actuator_moment, direct.force.Get(t + 1), nu, nv);

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

    // Clamp(ctrl.data(), direct.model->actuator_ctrlrange, direct.model->nu);

    // set ctrl
    mju_copy(trajectory.actions.data() + trajectory.dim_action * t,
             ctrl.data(), nu);

    // set residual terms
    mju_copy(trajectory.residual.data() + trajectory.dim_residual * t,
             direct.sensor.Get(t + 1), trajectory.dim_residual);

    // total return
    trajectory.total_return = direct.cost_;
  }

  // printf("times (trajectory): \n");
  // mju_printMat(trajectory.times.data(), 1, horizon);

  // nominal = trajectory;

  // printf("trajectory qpos \n");
  // mju_printMat(trajectory., direct.qpos_horizon_, direct.model->nq);

  // printf("trajectory ctrl \n");
  // mju_printMat(trajectory.actions.data(), horizon, direct.model->nu);

  // TODO(taylor): thread safe copy
  policy.horizon = horizon;
  mju_copy(policy.times.data(), trajectory.times.data(), horizon);
  mju_copy(policy.actions.data(), trajectory.actions.data(),
           horizon * direct.model->nu);

  // printf("time: %f\n", time);
  // printf("ctrl (recovered)\n");
  // mju_printMat(policy.times.data(), 1, horizon);
  // mju_printMat(policy.actions.data(), horizon, model->nu);

  // printf("regularization: %f\n", direct.regularization_);
  // printf("reduction ratio: %f\n", direct.reduction_ratio_);
  // printf("gradient norm: %f\n", direct.gradient_norm_);
  // printf("cost sensor: %f\n", direct.cost_sensor_);
  // printf("cost force: %f\n", direct.cost_force_);
  // printf("search iter: %i\n", direct.iterations_search_);
  // printf("smoother iter: %i\n", direct.iterations_smoother_);

  // printf("qpos: \n");
  // mju_printMat(direct.qpos.Data(), direct.qpos_horizon_, direct.model->nq);
  // printf("qvel: \n");
  // mju_printMat(direct.qvel.Data(), direct.qpos_horizon_, direct.model->nv);
  // printf("qacc: \n");
  // mju_printMat(direct.qacc.Data(), direct.qpos_horizon_, direct.model->nv);
  // printf("force: \n");
  // mju_printMat(direct.force.Data(), direct.qpos_horizon_, direct.model->nv);
  // printf("\n");
}

// compute trajectory using nominal policy
void DirectPlanner::NominalTrajectory(int horizon, ThreadPool& pool){
  // policy
  auto policy = [&dp = *this](double* action, const double* state,
                              double time) {
    dp.ActionFromPolicy(action, state, time);
  };

  // policy rollout
  nominal.Rollout(policy, task, model, direct.data_[0].get(), state.data(),
                  time, mocap.data(), userdata.data(), horizon);
}

// set action from policy
void DirectPlanner::ActionFromPolicy(double* action, const double* state,
                                     double time, bool use_previous){
                                      // find times bounds
  int bounds[2];
  FindInterval(bounds, policy.times, time, policy.horizon);

  // ----- get action ----- //

  // if (bounds[0] == bounds[1] ||
  //     representation == PolicyRepresentation::kZeroSpline) {
  ZeroInterpolation(action, time, policy.times, policy.actions.data(),
                    direct.model->nu, policy.horizon);
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
const Trajectory* DirectPlanner::BestTrajectory() { return &nominal; }

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
        direct.weight_force.data() + shift - 1, "1.0e-3 1.0e2"};

    // add element index
    str = "DOF ";
    str += " (" + std::to_string(i) + ")";

    // set sensor name
    mju::strcpy_arr(defForceWeight[shift].name,
                    str.c_str());

    // shift
    shift++;
  }

  defForceWeight[shift++] = {mjITEM_SLIDERNUM, "Sensor Scale", 2,
                             &sensor_scaling, "0.0 1.0e5"};

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
