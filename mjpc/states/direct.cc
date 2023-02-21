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

#include "mjpc/states/direct.h"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <mutex>
#include <shared_mutex>

#include "mjpc/utilities.h"

namespace mjpc {

// initialize settings
void DirectEstimation::Initialize(const mjModel* model) {
  // set sensor history
  horizon_ = GetNumberOrDefault(5, model, "sensory_history");

  // set sensor dimension
  num_sensor_ = GetNumberOrDefault(0, model, "sensor_dimension");

  if (num_sensor_ == 0) {
    mju_warning("Sensor dimension == 0\n");
  }

  // sensor shift
  sensor_shift_ = GetNumberOrDefault(0, model, "sensor_shift");

  if (sensor_shift_ == 0) {
    mju_warning("Sensor shift == 0\n");
  }

  // solver settings
  iterations_ = GetNumberOrDefault(5, model, "solver_iterations");
  gradient_tolerance_ = GetNumberOrDefault(1.0e-3, model, "gradient_tolerance");

  // model
  model_ = model;

  // data
  data_ = mj_makeData(model);
}

// allocate memory
void DirectEstimation::Allocate(const mjModel* model) {
  const std::unique_lock<std::shared_mutex> lock(mtx_);
  state_.resize(model->nq + model->nv + model->na);
  mocap_.resize(7 * model->nmocap);
  userdata_.resize(model->nuserdata);

  // ----- direct estimation memory ----- //
  // TODO(taylor): activations??
  // trajectories
  // configuration_
  // velocity_
  // acceleration_
  // measurements_;
  state1_.resize(model->nq + model->nv);
  state2_.resize(model->nq + model->nv);

  // // Jacobians
  // A_;  // dynamics state Jacobian
  // C_;                                 // sensor state Jacobian
  // E_;                                          // inverse dynamics current
  // state Jacobian

  // // identity matrices
  // I_state_; // identity (state dimension)
  // I_configuration_;         // identity (configuration dimension)
  // I_velocity_;              // identity (velocity dimension)

  // // ----- cost (1): dynamics ----- //
  // cost1_state_gradient_;
  // cost1_state_hessian_;
  // residual1_;
  // residual1_state_jacobian_;
  // P_;                        // weight matrix

  // ----- cost (2): sensors ----- //
  // cost2_state_gradient_;
  // cost2_state_hessian_;
  // residual2_;
  // residual2_state_jacobian_;
  // S_;                        // weight matrix

  // // ----- cost (3): actions ----- //
  // cost3_state_gradient_;
  // cost3_state_hessian_;
  // residual3_;
  // residual3_state_jacobian_;
  // R_;                        // weight matrix

  // // total cost
  // total_cost_configuration_gradient_;
  // total_cost_configuration_hessian_;

  // // configuration to state mapping
  // configuration_to_state_;

  // // search direction
  // search_direction_;
}

// reset memory to zeros
void DirectEstimation::Reset() {
  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    std::fill(state_.begin(), state_.end(), (double)0.0);
    std::fill(mocap_.begin(), mocap_.end(), 0.0);
    std::fill(userdata_.begin(), userdata_.end(), 0.0);
    time_ = 0.0;
  }

  // ----- direct estimation ----- //
}

// set state from data
void DirectEstimation::Set(const mjModel* model, const mjData* data) {
  if (model && data) {
    const std::unique_lock<std::shared_mutex> lock(mtx_);

    state_.resize(model->nq + model->nv + model->na);
    mocap_.resize(7 * model->nmocap);

    // state
    mju_copy(state_.data(), data->qpos, model->nq);
    mju_copy(DataAt(state_, model->nq), data->qvel, model->nv);
    mju_copy(DataAt(state_, model->nq + model->nv), data->act, model->na);

    // mocap
    for (int i = 0; i < model->nmocap; i++) {
      mju_copy(DataAt(mocap_, 7 * i), data->mocap_pos + 3 * i, 3);
      mju_copy(DataAt(mocap_, 7 * i + 3), data->mocap_quat + 4 * i, 4);
    }

    // userdata
    mju_copy(userdata_.data(), data->userdata, model->nuserdata);

    // time
    time_ = data->time;
  }
}

void DirectEstimation::CopyTo(double* dst_state, double* dst_mocap,
                              double* dst_userdata, double* dst_time) const {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  mju_copy(dst_state, this->state_.data(), this->state_.size());
  *dst_time = this->time_;
  mju_copy(dst_mocap, this->mocap_.data(), this->mocap_.size());
  mju_copy(dst_userdata, this->userdata_.data(), this->userdata_.size());
}

// cost (1)
double DirectEstimation::Cost1() {
  // residual dimension
  int dim = (2 * model_->nv) * (horizon_ - 1);
  mju_mulMatVec(cost1_scratch_.data(), P_.data(), residual1_.data(), dim, dim);
  return 0.5 * mju_dot(residual1_.data(), cost1_scratch_.data(), dim);
}

// cost (1) gradient wrt configuration, velocity, acceleration
void DirectEstimation::Cost1Gradient() {
  // residual dimension
  int dim1 = (2 * model_->nv) * (horizon_ - 1);
  int dim2 = (3 * model_->nv) * horizon_;
  mju_mulMatVec(cost1_scratch_.data(), P_.data(), residual1_.data(), dim1,
                dim1);
  mju_mulMatTVec(cost1_state_gradient_.data(), cost1_scratch_.data(),
                 residual1_state_jacobian_.data(), dim1, dim2);
}

// cost (1) Hessian wrt configuration, velocity, acceleration
void DirectEstimation::Cost1Hessian() {
  // residual dimension
  int dim1 = (2 * model_->nv) * (horizon_ - 1);
  int dim2 = (3 * model_->nv) * horizon_;
  mju_mulMatMat(cost1_scratch_.data(), P_.data(),
                residual1_state_jacobian_.data(), dim1, dim1, dim2);
  mju_mulMatTMat(cost1_state_hessian_.data(), cost1_scratch_.data(),
                 residual1_state_jacobian_.data(), dim1, dim2, dim2);
}

// residual (1)
void DirectEstimation::Residual1() {
  // rt = f(xt, ut) - xt+1
  for (int t = 0; t < horizon_ - 1; t++) {
    // get trajectory elements
    double* r = residual1_.data() + (2 * model_->nv) * t;
    double* q = configurations_.data() + 2 * model_->nq * t;
    double* v = velocities_.data() + model_->nv * t;
    double* u = actions_.data() + model_->nu * t;
    double* q_next = configurations_.data() + 2 * model_->nq * (t + 1);
    double* v_next = velocities_.data() + model_->nv * (t + 1);

    // set state
    mju_copy(data_->qpos, q, model_->nq);
    mju_copy(data_->qvel, v, model_->nv);
    mju_copy(data_->ctrl, u, model_->nu);
    mju_copy(state1_.data(), q_next, model_->nq);
    mju_copy(state1_.data() + model_->nq, v_next, model_->nv);

    // step
    mj_step(model_, data_);
    mju_copy(state2_.data(), data_->qpos, model_->nq);
    mju_copy(state2_.data() + model_->nq, data_->qvel, model_->nv);

    // compute dynamics error
    StateDiff(model_, r, state1_.data(), state2_.data(), 1.0);
  }
}

// residual (1) Jacobian wrt configuration, velocity, acceleration
void DirectEstimation::Residual1Jacobian() {
  // dimensions
  int dim_state_derivative = 2 * model_->nv;
  int dim_jacobian_row = dim_state_derivative * (horizon_ - 1);
  int dim_jacobian_col = (dim_state_derivative + model_->nv) * horizon_;

  // zero matrix
  mju_zero(residual1_state_jacobian_.data(),
           dim_jacobian_row * dim_jacobian_col);

  // set dynamics Jacobians
  for (int t = 0; t < horizon_ - 1; t++) {
    SetMatrixInMatrix(
        residual1_state_jacobian_.data(),
        A_.data() + (dim_state_derivative * dim_state_derivative) * t, 1.0,
        dim_jacobian_row, dim_jacobian_col, dim_state_derivative,
        dim_state_derivative, dim_state_derivative * t,
        (dim_state_derivative + model_->nv) * t);
    SetMatrixInMatrix(residual1_state_jacobian_.data(), I_state_.data(), -1.0,
                      dim_jacobian_row, dim_jacobian_col, dim_state_derivative,
                      dim_state_derivative, dim_state_derivative * t,
                      (dim_state_derivative + model_->nv) * (t + 1));
  }
}

// cost (2)
double DirectEstimation::Cost2() {
  // residual dimension
  int dim = num_sensor_ * (horizon_ - 1);
  mju_mulMatVec(cost2_scratch_.data(), S_.data(), residual2_.data(), dim, dim);
  return 0.5 * mju_dot(residual2_.data(), cost2_scratch_.data(), dim);
}

// cost (2) gradient wrt configuration, velocity, acceleration
void DirectEstimation::Cost2Gradient() {
  // residual dimension
  int dim1 = num_sensor_ * (horizon_ - 1);
  int dim2 = (3 * model_->nv) * horizon_;
  mju_mulMatVec(cost2_scratch_.data(), S_.data(), residual2_.data(), dim1,
                dim1);
  mju_mulMatTVec(cost2_state_gradient_.data(), cost2_scratch_.data(),
                 residual2_state_jacobian_.data(), dim1, dim2);
}

// cost (2) Hessian wrt configuration, velocity, acceleration
void DirectEstimation::Cost2Hessian() {
  // residual dimension
  int dim1 = num_sensor_ * (horizon_ - 1);
  int dim2 = (3 * model_->nv) * horizon_;
  mju_mulMatMat(cost2_scratch_.data(), S_.data(),
                residual2_state_jacobian_.data(), dim1, dim1, dim2);
  mju_mulMatTMat(cost2_state_hessian_.data(), cost2_scratch_.data(),
                 residual2_state_jacobian_.data(), dim1, dim2, dim2);
}

// residual (2) wrt configuration, velocity, acceleration
void DirectEstimation::Residual2() {
  // rt = g(xt, ut) - yt
  for (int t = 0; t < horizon_ - 1; t++) {
    // trajectory elements
    double* r = residual2_.data() + num_sensor_ * t;
    double* q = configurations_.data() + 2 * model_->nq * t;
    double* v = velocities_.data() + model_->nv * t;
    double* u = actions_.data() + model_->nu * t;
    double* y = measurements_.data() + num_sensor_ * t;

    // set state
    mju_copy(data_->qpos, q, model_->nq);
    mju_copy(data_->qvel, v, model_->nv);
    mju_copy(data_->ctrl, u, model_->nu);

    // step
    mj_step(model_, data_);

    // compute sensor error
    mju_sub(r, data_->sensordata + sensor_shift_, y, num_sensor_);
  }
}

// residual (2) wrt configuration, velocity, acceleration
void DirectEstimation::Residual2Jacobian() {
  // dimensions
  int jacobian_row = num_sensor_ * (horizon_ - 1);
  int jacobian_col = (3 * model_->nv) * horizon_;

  // zero matrix
  mju_zero(residual2_state_jacobian_.data(), jacobian_row * jacobian_col);

  // set sensor Jacobians
  for (int t = 0; t < horizon_ - 1; t++) {
    mjpc::SetMatrixInMatrix(
        residual2_state_jacobian_.data(),
        C_.data() + model_->nsensordata * (2 * model_->nv) * t, 1.0,
        jacobian_row, jacobian_col, num_sensor_, 2 * model_->nv,
        num_sensor_ * t, (3 * model_->nv) * t);
  }
}

// cost (3)
double DirectEstimation::Cost3() {
  // residual dimension
  int dim = model_->nv * (horizon_ - 1);
  mju_mulMatVec(cost3_scratch_.data(), R_.data(), residual3_.data(), dim, dim);
  return 0.5 * mju_dot(residual3_.data(), cost3_scratch_.data(), dim);
}

// cost (3) gradient wrt configuration, velocity, acceleration
void DirectEstimation::Cost3Gradient() {
  // residual dimension
  int dim1 = model_->nv * (horizon_ - 1);
  int dim2 = (3 * model_->nv) * horizon_;
  mju_mulMatVec(cost3_scratch_.data(), R_.data(), residual3_.data(), dim1,
                dim1);
  mju_mulMatTVec(cost3_state_gradient_.data(), cost3_scratch_.data(),
                 residual3_state_jacobian_.data(), dim1, dim2);
}

// cost (3) Hessian wrt configuration, velocity, acceleration
void DirectEstimation::Cost3Hessian() {
  // residual dimension
  int dim1 = model_->nv * (horizon_ - 1);
  int dim2 = (3 * model_->nv) * horizon_;
  mju_mulMatMat(cost3_scratch_.data(), R_.data(),
                residual3_state_jacobian_.data(), dim1, dim1, dim2);
  mju_mulMatTMat(cost3_state_hessian_.data(), cost3_scratch_.data(),
                 residual3_state_jacobian_.data(), dim1, dim2, dim2);
}

// residual (3)
void DirectEstimation::Residual3() {
  // rt = d(xt, xt+1) - B * ut
  for (int t = 0; t < horizon_ - 1; t++) {
    // trajectory elements
    double* r = residual3_.data() + model_->nv * t;
    double* q = configurations_.data() + 2 * model_->nq * t;
    double* v = velocities_.data() + model_->nv * t;
    double* a = accelerations_.data() + model_->nv * t;
    double* u = actions_.data() + model_->nu * t;

    // set state 
    mju_copy(data_->qpos, q, model_->nq);
    mju_copy(data_->qvel, v, model_->nv);

    // B * u 
    mju_copy(data_->ctrl, u, model_->nu);
    mj_fwdActuation(model_, data_);
    mju_scl(r, data_->qfrc_actuator, -1.0, model_->nv);

    // inverse dynamics 
    mju_copy(data_->qacc, a, model_->nv);
    mju_zero(data_->ctrl, model_->nu);
    mj_inverse(model_, data_);

    // qfrc_inverse - B * u
    mju_addTo(r, data_->qfrc_inverse, model_->nv);
  }
}

// residual (3) Jacobian wrt configuration, velocity, acceleration
void DirectEstimation::Residual3Jacobian() {
  // dimension
  int jacobian_row = model_->nv * (horizon_ - 1);
  int jacobian_col = (3 * model_->nv) * horizon_;

  // zero matrix
  mju_zero(residual3_state_jacobian_.data(), jacobian_row * jacobian_col);

  // set inverse dynamics Jacobians
  for (int t = 0; t < horizon_ - 1; t++) {
    mjpc::SetMatrixInMatrix(residual3_state_jacobian_.data(),
                            E_.data() + model_->nv * (horizon_ - 1) * t, 1.0,
                            jacobian_row, jacobian_col, model_->nv,
                            3 * model_->nv, model_->nv * t, 3 * model_->nv * t);
  }
}

// total cost
double DirectEstimation::TotalCost() {
  return this->Cost1() + this->Cost2() + this->Cost3();
}

// total cost gradient wrt configuration
void DirectEstimation::TotalCostGradient() {
  // dimensions 
  int dim1 = 3 * model_->nv * horizon_;
  int dim2 = 2 * model_->nq * horizon_;

  // sum cost gradients wrt state 
  mju_copy(total_cost_state_gradient_.data(), cost1_state_gradient_.data(), dim1);
  mju_addTo(total_cost_state_gradient_.data(), cost2_state_gradient_.data(), dim1);
  mju_addTo(total_cost_state_gradient_.data(), cost3_state_gradient_.data(), dim1);

  // configuration to state projection
  mju_mulMatTVec(total_cost_configuration_gradient_.data(),
              configuration_to_state_.data(), total_cost_state_gradient_.data(),
              dim1, dim2);
}

// total cost Hessian wrt configuration
void DirectEstimation::TotalCostHessian() {
  // dimensions
  int dim1 = 3 * model_->nv * horizon_;
  int dim2 = 2 * model_->nq * horizon_;

  // sum cost Hessians wrt state
  mju_copy(total_cost_state_hessian_.data(), cost1_state_hessian_.data(),
           dim1 * dim1);
  mju_addTo(total_cost_state_hessian_.data(), cost2_state_hessian_.data(),
            dim1 * dim1);
  mju_addTo(total_cost_state_hessian_.data(), cost3_state_hessian_.data(),
            dim1 * dim1);

  // M' H M
  mju_mulMatMat(total_cost_state_hessian_cache_.data(),
                total_cost_state_hessian_.data(),
                configuration_to_state_.data(), dim1, dim1, dim2);
  mju_mulMatTMat(total_cost_configuration_hessian_.data(),
                 configuration_to_state_.data(),
                 total_cost_state_hessian_cache_.data(), dim1, dim2, dim2);
}

}  // namespace mjpc
