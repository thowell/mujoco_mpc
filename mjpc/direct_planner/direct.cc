// Copyright 2023 DeepMind Technologies Limited
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

#include "mjpc/direct_planner/direct.h"

#include <absl/random/random.h>
#include <absl/strings/match.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>

#include "mjpc/direct/trajectory.h"
#include "mjpc/norm.h"
#include "mjpc/trajectory.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {


// initialize direct estimator
void Direct2::Initialize(mjModel* model) {
  // activations check
  if (model->na > 0) {
    mju_error("na > 0: activations not implemented");
  }

  // model
  this->model = model;

  // set discrete inverse dynamics
  this->model->opt.enableflags |= mjENBL_INVDISCRETE;

  // sensor start index
  sensor_start_ = GetNumberOrDefault(0, model, "direct_sensor_start");

  // number of sensors
  nsensor_ = GetNumberOrDefault(model->nsensor, model, "direct_number_sensor");

  // sensor dimension
  nsensordata_ = 0;
  for (int i = 0; i < nsensor_; i++) {
    nsensordata_ += model->sensor_dim[sensor_start_ + i];
  }

  // sensor start index
  sensor_start_index_ = 0;
  for (int i = 0; i < sensor_start_; i++) {
    sensor_start_index_ += model->sensor_dim[i];
  }

  // band dimension
  nband_ = 3 * model->nv;
}

// initialize direct estimator
void Direct2::Allocate() {
  // data
  data_.clear();
  for (int i = 0; i < kMaxTrajectoryHorizon; i++) {
    data_.push_back(MakeUniqueMjData(mj_makeData(model)));
  }

  // allocation dimension
  int nq = model->nq, nv = model->nv;
  int ntotal_max = nv * kMaxTrajectoryHorizon;
  int nsensor_max = nsensordata_ * kMaxTrajectoryHorizon;

  // force weights
  weight_force.resize(nv);

  // sensor weights
  weight_sensor.resize(nsensor_);

  // -- trajectories -- //
  qpos.Initialize(nq, kMaxTrajectoryHorizon);
  qvel.Initialize(nv, kMaxTrajectoryHorizon);
  qacc.Initialize(nv, kMaxTrajectoryHorizon);
  time.Initialize(1, kMaxTrajectoryHorizon);

  // sensor
  sensor_target.Initialize(nsensordata_, kMaxTrajectoryHorizon);
  sensor.Initialize(nsensordata_, kMaxTrajectoryHorizon);

  // force
  force_target.Initialize(nv, kMaxTrajectoryHorizon);
  force.Initialize(nv, kMaxTrajectoryHorizon);

  // residual
  residual_sensor_.resize(nsensor_max);
  residual_force_.resize(ntotal_max);

  // sensor Jacobian
  jac_sensor_qpos.Initialize(model->nsensordata * nv, kMaxTrajectoryHorizon);
  jac_sensor_qvel.Initialize(model->nsensordata * nv, kMaxTrajectoryHorizon);
  jac_sensor_qacc.Initialize(model->nsensordata * nv, kMaxTrajectoryHorizon);
  jac_qpos_sensor.Initialize(model->nsensordata * nv, kMaxTrajectoryHorizon);
  jac_qvel_sensor.Initialize(model->nsensordata * nv, kMaxTrajectoryHorizon);
  jac_qacc_sensor.Initialize(model->nsensordata * nv, kMaxTrajectoryHorizon);

  jac_sensor_qpos0.Initialize(nsensordata_ * nv, kMaxTrajectoryHorizon);
  jac_sensor_qpos1.Initialize(nsensordata_ * nv, kMaxTrajectoryHorizon);
  jac_sensor_qpos2.Initialize(nsensordata_ * nv, kMaxTrajectoryHorizon);
  jac_sensor_qpos012.Initialize(nsensordata_ * nband_, kMaxTrajectoryHorizon);

  jac_sensor_scratch.Initialize(
      std::max(nv, nsensordata_) * std::max(nv, nsensordata_), kMaxTrajectoryHorizon);

  // force Jacobian
  jac_force_qpos.Initialize(nv * nv, kMaxTrajectoryHorizon);
  jac_force_qvel.Initialize(nv * nv, kMaxTrajectoryHorizon);
  jac_force_qacc.Initialize(nv * nv, kMaxTrajectoryHorizon);

  jac_force_qpos0.Initialize(nv * nv, kMaxTrajectoryHorizon);
  jac_force_qpos1.Initialize(nv * nv, kMaxTrajectoryHorizon);
  jac_force_qpos2.Initialize(nv * nv, kMaxTrajectoryHorizon);
  jac_force_qpos012.Initialize(nv * nband_, kMaxTrajectoryHorizon);

  jac_force_scratch.Initialize(nv * nv, kMaxTrajectoryHorizon);

  // qvel Jacobian wrt qpos0, qpos1
  jac_qvel1_qpos0.Initialize(nv * nv, kMaxTrajectoryHorizon);
  jac_qvel1_qpos1.Initialize(nv * nv, kMaxTrajectoryHorizon);

  // qacc Jacobian wrt qpos0, qpos1, qpos2
  jac_qacc1_qpos0.Initialize(nv * nv, kMaxTrajectoryHorizon);
  jac_qacc1_qpos1.Initialize(nv * nv, kMaxTrajectoryHorizon);
  jac_qacc1_qpos2.Initialize(nv * nv, kMaxTrajectoryHorizon);

  // cost gradient
  gradient_sensor_.resize(ntotal_max);
  gradient_force_.resize(ntotal_max);
  gradient_.resize(ntotal_max);

  // cost Hessian
  hessian_band_sensor_.resize(ntotal_max * nband_);
  hessian_band_force_.resize(ntotal_max * nband_);
  hessian_band_.resize(ntotal_max * nband_);
  hessian_band_factor_.resize(ntotal_max * nband_);

  // cost norms
  norm_type_sensor.resize(nsensor_);

  // cost norm parameters
  norm_parameters_sensor.resize(nsensor_ * kMaxNormParameters);

  // norm
  norm_sensor_.resize(nsensor_ * kMaxTrajectoryHorizon);
  norm_force_.resize(kMaxTrajectoryHorizon);

  // norm gradient
  norm_gradient_sensor_.resize(nsensor_max);
  norm_gradient_force_.resize(ntotal_max);

  // norm Hessian
  norm_hessian_sensor_.resize(nsensordata_ * nsensor_max);
  norm_hessian_force_.resize(nv * ntotal_max);

  // scratch
  scratch_sensor_.resize(nband_ + nsensordata_ * nband_ + 9 * nv * nv);
  scratch_force_.resize(12 * nv * nv);
  scratch_expected_.resize(ntotal_max);

  // copy
  qpos_copy_.Initialize(nq, kMaxTrajectoryHorizon);

  // search direction
  search_direction_.resize(ntotal_max);

  // timer
  timer_.sensor_step.resize(kMaxTrajectoryHorizon);
  timer_.force_step.resize(kMaxTrajectoryHorizon);

  // pinned
  pinned.resize(kMaxTrajectoryHorizon);
}

// reset memory
void Direct2::Reset() {
  regularization_ = settings.regularization_initial;

  // -- force weight -- //
  double* wf = GetCustomNumericData(model, "direct_force_weight", model->nv);
  if (wf) {
    mju_copy(weight_force.data(), wf, model->nv);
  } else {
    std::fill(weight_force.begin(), weight_force.end(), 1.0);
  }

  // -- sensor weight -- //
  double* ws = GetCustomNumericData(model, "direct_sensor_weight", nsensor_);
  if (ws) {
    mju_copy(weight_sensor.data(), ws, nsensor_);
  } else {
    std::fill(weight_sensor.begin(), weight_sensor.end(), 1.0);
  }

  // TODO(taylor): parse
  std::fill(norm_type_sensor.begin(), norm_type_sensor.end(), kQuadratic);
  std::fill(norm_parameters_sensor.begin(), norm_parameters_sensor.end(), 0.0);

  // trajectories
  qpos.Reset();
  qvel.Reset();
  qacc.Reset();
  time.Reset();

  // sensor
  sensor_target.Reset();
  sensor.Reset();

  // force
  force_target.Reset();
  force.Reset();

  // residual
  std::fill(residual_sensor_.begin(), residual_sensor_.end(), 0.0);
  std::fill(residual_force_.begin(), residual_force_.end(), 0.0);

  // sensor Jacobian
  jac_sensor_qpos.Reset();
  jac_sensor_qvel.Reset();
  jac_sensor_qacc.Reset();
  jac_qpos_sensor.Reset();
  jac_qvel_sensor.Reset();
  jac_qacc_sensor.Reset();

  jac_sensor_qpos0.Reset();
  jac_sensor_qpos1.Reset();
  jac_sensor_qpos2.Reset();
  jac_sensor_qpos012.Reset();

  jac_sensor_scratch.Reset();

  // force Jacobian
  jac_force_qpos.Reset();
  jac_force_qvel.Reset();
  jac_force_qacc.Reset();

  jac_force_qpos0.Reset();
  jac_force_qpos1.Reset();
  jac_force_qpos2.Reset();
  jac_force_qpos012.Reset();

  jac_force_scratch.Reset();

  // qvel Jacobian
  jac_qvel1_qpos0.Reset();
  jac_qvel1_qpos1.Reset();

  // qacc Jacobian
  jac_qacc1_qpos0.Reset();
  jac_qacc1_qpos1.Reset();
  jac_qacc1_qpos2.Reset();

  // cost
  cost_sensor_ = 0.0;
  cost_force_ = 0.0;
  cost_ = 0.0;
  cost_initial_ = 0.0;
  cost_previous_ = 1.0e32;

  // cost gradient
  std::fill(gradient_sensor_.begin(), gradient_sensor_.end(), 0.0);
  std::fill(gradient_force_.begin(), gradient_force_.end(), 0.0);
  std::fill(gradient_.begin(), gradient_.end(), 0.0);

  // cost Hessian
  std::fill(hessian_band_sensor_.begin(), hessian_band_sensor_.end(), 0.0);
  std::fill(hessian_band_force_.begin(), hessian_band_force_.end(), 0.0);
  std::fill(hessian_.begin(), hessian_.end(), 0.0);
  std::fill(hessian_band_.begin(), hessian_band_.end(), 0.0);
  std::fill(hessian_band_factor_.begin(), hessian_band_factor_.end(), 0.0);

  // norm
  std::fill(norm_sensor_.begin(), norm_sensor_.end(), 0.0);
  std::fill(norm_force_.begin(), norm_force_.end(), 0.0);

  // norm gradient
  std::fill(norm_gradient_sensor_.begin(), norm_gradient_sensor_.end(), 0.0);
  std::fill(norm_gradient_force_.begin(), norm_gradient_force_.end(), 0.0);

  // norm Hessian
  std::fill(norm_hessian_sensor_.begin(), norm_hessian_sensor_.end(), 0.0);
  std::fill(norm_hessian_force_.begin(), norm_hessian_force_.end(), 0.0);

  std::fill(norm_hessian_sensor_.begin(), norm_hessian_sensor_.end(), 0.0);
  std::fill(norm_hessian_force_.begin(), norm_hessian_force_.end(), 0.0);

  // scratch
  std::fill(scratch_sensor_.begin(), scratch_sensor_.end(), 0.0);
  std::fill(scratch_force_.begin(), scratch_force_.end(), 0.0);
  std::fill(scratch_expected_.begin(), scratch_expected_.end(), 0.0);

  // candidate
  qpos_copy_.Reset();

  // search direction
  std::fill(search_direction_.begin(), search_direction_.end(), 0.0);

  // timer
  std::fill(timer_.sensor_step.begin(), timer_.sensor_step.end(), 0.0);
  std::fill(timer_.force_step.begin(), timer_.force_step.end(), 0.0);

  // timing
  ResetTimers();

  std::fill(pinned.begin(), pinned.end(), false);
}

// evaluate configurations
void Direct2::EvaluateConfigurations() {
  // finite-difference velocities, accelerations
  ConfigurationToVelocityAcceleration();

  // compute sensor and force predictions
  InverseDynamicsPrediction();
}

// configurations derivatives
void Direct2::ConfigurationDerivative() {
  // inverse dynamics derivatives
  InverseDynamicsDerivatives();

  // qvel, qacc derivatives
  VelocityAccelerationDerivatives();

  // -- Jacobians -- //
  auto timer_jacobian_start = std::chrono::steady_clock::now();

  // pool count
  int count_begin = pool_.GetCount();

  // individual derivatives
  JacobianSensor();
  JacobianForce();

  // wait
  pool_.WaitCount(count_begin + 2 * (qpos_horizon_ - 2));

  // reset count
  pool_.ResetCount();

  // timers
  timer_.jacobian_sensor +=
      mju_sum(timer_.sensor_step.data(), qpos_horizon_ - 2);
  timer_.jacobian_force += mju_sum(timer_.force_step.data(), qpos_horizon_ - 2);
  timer_.jacobian_total += GetDuration(timer_jacobian_start);
}

// sensor cost
double Direct2::CostSensor(double* gradient, double* hessian) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // update dimension
  int nv = model->nv, ns = nsensordata_;

  // residual
  ResidualSensor();

  // ----- cost ----- //

  // initialize
  double cost = 0.0;

  // zero memory
  if (gradient) mju_zero(gradient, ntotal_);
  if (hessian) mju_zero(hessian, ntotal_ * nband_);

  // time scaling
  double time_scale = 1.0;
  double time_scale2 = 1.0;
  if (settings.time_scaling_sensor) {
    time_scale = model->opt.timestep * model->opt.timestep;
    time_scale2 = time_scale * time_scale;
  }

  // matrix shift
  int shift_matrix = 0;

  // loop over predictions
  for (int t = 1; t < qpos_horizon_ - 1; t++) {
    // residual
    double* rt = residual_sensor_.data() + ns * t;

    // unpack Jacobian
    double* jac = jac_sensor_qpos012.Get(t);
    int jac_columns = nband_;

    // shift
    int shift_sensor = 0;

    // loop over sensors
    for (int i = 0; i < nsensor_; i++) {
      // start cost timer
      auto start_cost = std::chrono::steady_clock::now();

      // sensor stage
      int sensor_stage = model->sensor_needstage[sensor_start_ + i];

      // time scaling weight
      double time_weight = 1.0;
      if (sensor_stage == mjSTAGE_VEL) {
        time_weight = time_scale;
      } else if (sensor_stage == mjSTAGE_ACC) {
        time_weight = time_scale2;
      }

      // dimension
      int nsi = model->sensor_dim[sensor_start_ + i];

      // sensor residual
      double* rti = rt + shift_sensor;

      // weight
      double weight = time_weight * weight_sensor[i] / nsi / qpos_horizon_;

      // parameters
      double* pi = norm_parameters_sensor.data() + kMaxNormParameters * i;

      // norm
      NormType normi = norm_type_sensor[i];

      // norm gradient
      double* norm_gradient =
          norm_gradient_sensor_.data() + ns * t + shift_sensor;

      // norm Hessian
      double* norm_hessian = norm_hessian_sensor_.data() + shift_matrix;

      // ----- cost ----- //

      // norm
      norm_sensor_[nsensor_ * t + i] =
          Norm(gradient ? norm_gradient : NULL, hessian ? norm_hessian : NULL,
               rti, pi, nsi, normi);

      // weighted norm
      cost += weight * norm_sensor_[nsensor_ * t + i];

      // stop cost timer
      timer_.cost_sensor += GetDuration(start_cost);

      // gradient wrt qpos: dsidq012' * dndsi
      if (gradient) {
        // sensor Jacobian
        double* jaci = jac + jac_columns * shift_sensor;

        // scratch = dsidq012' * dndsi
        mju_mulMatTVec(scratch_sensor_.data(), jaci, norm_gradient, nsi,
                       jac_columns);

        // add
        mju_addToScl(gradient + nv * std::max(0, t - 1), scratch_sensor_.data(),
                     weight, jac_columns);
      }

      // Hessian (Gauss-Newton): dsidq012' * d2ndsi2 * dsidq012
      if (hessian) {
        // sensor Jacobian
        double* jaci = jac + jac_columns * shift_sensor;

        // step 1: tmp0 = d2ndsi2 * dsidq
        double* tmp0 = scratch_sensor_.data();
        mju_mulMatMat(tmp0, norm_hessian, jaci, nsi, nsi, jac_columns);

        // step 2: hessian = dsidq' * tmp
        double* tmp1 = scratch_sensor_.data() + nsensordata_ * nband_;
        mju_mulMatTMat(tmp1, jaci, tmp0, nsi, jac_columns, jac_columns);

        // set Jacobian in band Hessian
        SetBlockInBand(hessian, tmp1, weight, ntotal_, nband_, jac_columns,
                       nv * std::max(0, t - 1));
      }

      // shift by individual sensor dimension
      shift_sensor += nsi;
      shift_matrix += nsi * nsi;
    }
  }

  // stop timer
  timer_.cost_sensor_derivatives += GetDuration(start);

  return cost;
}

// sensor residual
void Direct2::ResidualSensor() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // loop over predictions
  for (int t = 1; t < qpos_horizon_ - 1; t++) {
    // terms
    double* rt = residual_sensor_.data() + t * nsensordata_;
    double* yt_sensor = sensor_target.Get(t);
    double* yt_model = sensor.Get(t);

    // sensor difference
    mju_sub(rt, yt_model, yt_sensor, nsensordata_);
  }

  // stop timer
  timer_.residual_sensor += GetDuration(start);
}

// sensor Jacobian (dsdq0, dsdq1, dsdq2)
void Direct2::BlockSensor(int index) {
  // dimensions
  int nv = model->nv, ns = nsensordata_;

  // shift
  int shift = sensor_start_index_ * nv;

  // -- timesteps [1,...,T - 1] -- //

  // dqds
  double* dsdq = jac_sensor_qpos.Get(index) + shift;

  // dvds
  double* dsdv = jac_sensor_qvel.Get(index) + shift;

  // dads
  double* dsda = jac_sensor_qacc.Get(index) + shift;

  // -- qpos previous: dsdq0 = dsdv * dvdq0 + dsda * dadq0 -- //

  // unpack
  double* dsdq0 = jac_sensor_qpos0.Get(index);
  double* tmp = jac_sensor_scratch.Get(index);

  // dsdq0 <- dvds' * dvdq0
  double* dvdq0 = jac_qvel1_qpos0.Get(index);
  mju_mulMatMat(dsdq0, dsdv, dvdq0, ns, nv, nv);

  // dsdq0 += dads' * dadq0
  double* dadq0 = jac_qacc1_qpos0.Get(index);
  mju_mulMatMat(tmp, dsda, dadq0, ns, nv, nv);
  mju_addTo(dsdq0, tmp, ns * nv);

  // -- qpos current: dsdq1 = dsdq + dsdv * dvdq1 + dsda * dadq1 --

  // unpack
  double* dsdq1 = jac_sensor_qpos1.Get(index);

  // dsdq1 <- dqds'
  mju_copy(dsdq1, dsdq, ns * nv);

  // dsdq1 += dvds' * dvdq1
  double* dvdq1 = jac_qvel1_qpos1.Get(index);
  mju_mulMatMat(tmp, dsdv, dvdq1, ns, nv, nv);
  mju_addTo(dsdq1, tmp, ns * nv);

  // dsdq1 += dads' * dadq1
  double* dadq1 = jac_qacc1_qpos1.Get(index);
  mju_mulMatMat(tmp, dsda, dadq1, ns, nv, nv);
  mju_addTo(dsdq1, tmp, ns * nv);

  // -- qpos next: dsdq2 = dsda * dadq2 -- //

  // unpack
  double* dsdq2 = jac_sensor_qpos2.Get(index);

  // dsdq2 = dads' * dadq2
  double* dadq2 = jac_qacc1_qpos2.Get(index);
  mju_mulMatMat(dsdq2, dsda, dadq2, ns, nv, nv);

  // -- assemble dsdq012 Jacobian -- //

  // unpack
  double* dsdq012 = jac_sensor_qpos012.Get(index);

  // set dfdq0
  SetBlockInMatrix(dsdq012, dsdq0, 1.0, ns, nband_, ns, nv, 0, 0 * nv);

  // set dfdq1
  SetBlockInMatrix(dsdq012, dsdq1, 1.0, ns, nband_, ns, nv, 0, 1 * nv);

  // set dfdq0
  SetBlockInMatrix(dsdq012, dsdq2, 1.0, ns, nband_, ns, nv, 0, 2 * nv);
}

// sensor Jacobian
// note: pool wait is called outside this function
void Direct2::JacobianSensor() {
  // loop over predictions
  for (int t = 1; t < qpos_horizon_ - 1; t++) {
    // schedule by time step
    pool_.Schedule([&direct = *this, t]() {
      // start Jacobian timer
      auto jacobian_sensor_start = std::chrono::steady_clock::now();

      // Jacobian term
      direct.BlockSensor(t);

      // stop Jacobian timer
      direct.timer_.sensor_step[t] = GetDuration(jacobian_sensor_start);
    });
  }
}

// force cost
double Direct2::CostForce(double* gradient, double* hessian) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // update dimension
  int nv = model->nv;

  // residual
  ResidualForce();

  // ----- cost ----- //

  // initialize
  double cost = 0.0;

  // zero memory
  if (gradient) mju_zero(gradient, ntotal_);
  if (hessian) mju_zero(hessian, ntotal_ * nband_);

  // time scaling
  double time_scale2 = 1.0;
  if (settings.time_scaling_force) {
    time_scale2 = model->opt.timestep * model->opt.timestep *
                  model->opt.timestep * model->opt.timestep;
  }

  // loop over predictions
  for (int t = 1; t < qpos_horizon_ - 1; t++) {
    // unpack Jacobian
    double* jac = jac_force_qpos012.Get(t);

    // start cost timer
    auto start_cost = std::chrono::steady_clock::now();

    // residual
    double* rt = residual_force_.data() + t * nv;

    // norm gradient
    double* norm_gradient = norm_gradient_force_.data() + t * nv;

    // norm block
    double* norm_hessian = norm_hessian_force_.data() + t * nv * nv;
    mju_zero(norm_hessian, nv * nv);

    // ----- cost ----- //

    // quadratic cost
    for (int i = 0; i < nv; i++) {
      // weight
      double weight = time_scale2 * weight_force[i] / nv / (qpos_horizon_ - 2);

      // gradient
      norm_gradient[i] = weight * rt[i];

      // Hessian
      norm_hessian[nv * i + i] = weight;
    }

    // norm
    norm_force_[t] = 0.5 * mju_dot(rt, norm_gradient, nv);

    // weighted norm
    cost += norm_force_[t];

    // stop cost timer
    timer_.cost_force += GetDuration(start_cost);

    // gradient wrt qpos: dfdq012' * dndf
    if (gradient) {
      // scratch = dfdq012' * dndf
      mju_mulMatTVec(scratch_force_.data(), jac, norm_gradient, nv, nband_);

      // add
      mju_addToScl(gradient + (t - 1) * nv, scratch_force_.data(), 1.0, nband_);
    }

    // Hessian (Gauss-Newton): drdq012' * d2ndf2 * dfdq012
    if (hessian) {
      // step 1: tmp0 = d2ndf2 * dfdq012
      double* tmp0 = scratch_force_.data();
      mju_mulMatMat(tmp0, norm_hessian, jac, nv, nv, nband_);

      // step 2: hessian = dfdq012' * tmp
      double* tmp1 = tmp0 + nv * nband_;
      mju_mulMatTMat(tmp1, jac, tmp0, nv, nband_, nband_);

      // set Jacobian in band Hessian
      SetBlockInBand(hessian, tmp1, 1.0, ntotal_, nband_, nband_, nv * (t - 1));
    }
  }

  // stop timer
  timer_.cost_force_derivatives += GetDuration(start);

  return cost;
}

// force residual
void Direct2::ResidualForce() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model->nv;

  // loop over predictions
  for (int t = 1; t < qpos_horizon_ - 1; t++) {
    // terms
    double* rt = residual_force_.data() + t * nv;
    double* ft_actuator = force_target.Get(t);
    double* ft_inverse = force.Get(t);

    // force difference
    mju_sub(rt, ft_inverse, ft_actuator, nv);
  }

  // stop timer
  timer_.residual_force += GetDuration(start);
}

// force Jacobian (dfdq0, dfdq1, dfdq2)
void Direct2::BlockForce(int index) {
  // dimensions
  int nv = model->nv;

  // dqdf
  double* dqdf = jac_force_qpos.Get(index);

  // dvdf
  double* dvdf = jac_force_qvel.Get(index);

  // dadf
  double* dadf = jac_force_qacc.Get(index);

  // -- qpos previous: dfdq0 = dfdv * dvdq0 + dfda * dadq0 -- //

  // unpack
  double* dfdq0 = jac_force_qpos0.Get(index);
  double* tmp = jac_force_scratch.Get(index);

  // dfdq0 <- dvdf' * dvdq0
  double* dvdq0 = jac_qvel1_qpos0.Get(index);
  mju_mulMatTMat(dfdq0, dvdf, dvdq0, nv, nv, nv);

  // dfdq0 += dadf' * dadq0
  double* dadq0 = jac_qacc1_qpos0.Get(index);
  mju_mulMatTMat(tmp, dadf, dadq0, nv, nv, nv);
  mju_addTo(dfdq0, tmp, nv * nv);

  // -- qpos current: dfdq1 = dfdq + dfdv * dvdq1 + dfda * dadq1 --

  // unpack
  double* dfdq1 = jac_force_qpos1.Get(index);

  // dfdq1 <- dqdf'
  mju_transpose(dfdq1, dqdf, nv, nv);

  // dfdq1 += dvdf' * dvdq1
  double* dvdq1 = jac_qvel1_qpos1.Get(index);
  mju_mulMatTMat(tmp, dvdf, dvdq1, nv, nv, nv);
  mju_addTo(dfdq1, tmp, nv * nv);

  // dfdq1 += dadf' * dadq1
  double* dadq1 = jac_qacc1_qpos1.Get(index);
  mju_mulMatTMat(tmp, dadf, dadq1, nv, nv, nv);
  mju_addTo(dfdq1, tmp, nv * nv);

  // -- qpos next: dfdq2 = dfda * dadq2 -- //

  // unpack
  double* dfdq2 = jac_force_qpos2.Get(index);

  // dfdq2 = dadf' * dadq2
  double* dadq2 = jac_qacc1_qpos2.Get(index);
  mju_mulMatTMat(dfdq2, dadf, dadq2, nv, nv, nv);

  // -- assemble dfdq012 Jacobian -- //

  // unpack
  double* dfdq012 = jac_force_qpos012.Get(index);

  // set dfdq0
  SetBlockInMatrix(dfdq012, dfdq0, 1.0, nv, nband_, nv, nv, 0, 0 * nv);

  // set dfdq1
  SetBlockInMatrix(dfdq012, dfdq1, 1.0, nv, nband_, nv, nv, 0, 1 * nv);

  // set dfdq0
  SetBlockInMatrix(dfdq012, dfdq2, 1.0, nv, nband_, nv, nv, 0, 2 * nv);
}

// force Jacobian
// note: pool wait is called outside this function
void Direct2::JacobianForce() {
  // loop over predictions
  for (int t = 1; t < qpos_horizon_ - 1; t++) {
    // schedule by time step
    pool_.Schedule([&direct = *this, t]() {
      // start Jacobian timer
      auto jacobian_force_start = std::chrono::steady_clock::now();

      // Jacobian term
      direct.BlockForce(t);

      // stop Jacobian timer
      direct.timer_.force_step[t] = GetDuration(jacobian_force_start);
    });
  }
}

// compute force
void Direct2::InverseDynamicsPrediction() {
  // compute sensor and force predictions
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model->nq, nv = model->nv, ns = nsensordata_;

  // pool count
  int count_before = pool_.GetCount();

  // loop over predictions
  for (int t = 1; t < qpos_horizon_ - 1; t++) {
    // schedule
    pool_.Schedule([&direct = *this, nq, nv, ns, t]() {
      // terms
      double* qt = direct.qpos.Get(t);
      double* vt = direct.qvel.Get(t);
      double* at = direct.qacc.Get(t);

      // data
      mjData* d = direct.data_[t].get();

      // set qt, vt, at
      mju_copy(d->qpos, qt, nq);
      mju_copy(d->qvel, vt, nv);
      mju_copy(d->qacc, at, nv);
      d->time = direct.time.Get(t)[0];

      // inverse dynamics
      mj_inverse(direct.model, d);

      // copy sensor
      double* st = direct.sensor.Get(t);
      mju_copy(st, d->sensordata + direct.sensor_start_index_, ns);

      // copy force
      double* ft = direct.force.Get(t);
      mju_copy(ft, d->qfrc_inverse, nv);
    });
  }

  // wait
  pool_.WaitCount(count_before + qpos_horizon_ - 2);
  pool_.ResetCount();

  // stop timer
  timer_.cost_prediction += GetDuration(start);
}

// compute inverse dynamics derivatives (via finite difference)
void Direct2::InverseDynamicsDerivatives() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model->nq, nv = model->nv;

  // pool count
  int count_before = pool_.GetCount();

  // loop over predictions
  for (int t = 1; t < qpos_horizon_ - 1; t++) {
    // schedule
    pool_.Schedule([&direct = *this, nq, nv, t]() {
      // unpack
      double* q = direct.qpos.Get(t);
      double* v = direct.qvel.Get(t);
      double* a = direct.qacc.Get(t);

      double* dsdq = direct.jac_sensor_qpos.Get(t);
      double* dsdv = direct.jac_sensor_qvel.Get(t);
      double* dsda = direct.jac_sensor_qacc.Get(t);
      double* dqds = direct.jac_qpos_sensor.Get(t);
      double* dvds = direct.jac_qvel_sensor.Get(t);
      double* dads = direct.jac_qacc_sensor.Get(t);
      double* dqdf = direct.jac_force_qpos.Get(t);
      double* dvdf = direct.jac_force_qvel.Get(t);
      double* dadf = direct.jac_force_qacc.Get(t);
      mjData* data = direct.data_[t].get();  // TODO(taylor): WorkerID

      // set state, qacc
      mju_copy(data->qpos, q, nq);
      mju_copy(data->qvel, v, nv);
      mju_copy(data->qacc, a, nv);
      data->time = direct.time.Get(t)[0];

      // finite-difference derivatives
      mjd_inverseFD(direct.model, data, direct.finite_difference.tolerance,
                    direct.finite_difference.flg_actuation, dqdf, dvdf, dadf,
                    dqds, dvds, dads, NULL);

      // transpose
      mju_transpose(dsdq, dqds, nv, direct.model->nsensordata);
      mju_transpose(dsdv, dvds, nv, direct.model->nsensordata);
      mju_transpose(dsda, dads, nv, direct.model->nsensordata);
    });
  }

  // wait
  pool_.WaitCount(count_before + qpos_horizon_ - 2);

  // reset pool count
  pool_.ResetCount();

  // stop timer
  timer_.inverse_dynamics_derivatives += GetDuration(start);
}

// update qpos trajectory
void Direct2::UpdateConfiguration(DirectTrajectory<double>& candidate,
                                  const DirectTrajectory<double>& qpos,
                                  const double* search_direction,
                                  double step_size, std::vector<bool>& pinned) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model->nq, nv = model->nv;

  // loop over configurations
  int tpin = 0;
  for (int t = 0; t < qpos_horizon_; t++) {
    if (pinned[t]) continue;

    // unpack
    const double* qt = qpos.Get(t);
    double* ct = candidate.Get(t);

    // copy
    mju_copy(ct, qt, nq);

    // search direction
    const double* dqt = search_direction + tpin * nv;

    // integrate
    mj_integratePos(model, ct, dqt, step_size);

    // increment search direction
    tpin++;
  }

  // stop timer
  timer_.configuration_update += GetDuration(start);
}

// convert sequence of configurations to velocities and accelerations
void Direct2::ConfigurationToVelocityAcceleration() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model->nv;

  // loop over configurations
  for (int t = 1; t < qpos_horizon_; t++) {
    // previous and current configurations
    const double* q0 = qpos.Get(t - 1);
    const double* q1 = qpos.Get(t);

    // compute qvel
    double* v1 = qvel.Get(t);
    mj_differentiatePos(model, v1, model->opt.timestep, q0, q1);

    // compute qacc
    if (t > 1) {
      // previous qvel
      const double* v0 = qvel.Get(t - 1);

      // compute qacc
      double* a1 = qacc.Get(t - 1);
      mju_sub(a1, v1, v0, nv);
      mju_scl(a1, a1, 1.0 / model->opt.timestep, nv);
    }
  }

  // stop time
  timer_.cost_config_to_velacc += GetDuration(start);
}

// compute finite-difference qvel, qacc derivatives
void Direct2::VelocityAccelerationDerivatives() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model->nv;

  // loop over configurations
  for (int t = 1; t < qpos_horizon_; t++) {
    // unpack
    double* q1 = qpos.Get(t - 1);
    double* q2 = qpos.Get(t);
    double* dv2dq1 = jac_qvel1_qpos0.Get(t);
    double* dv2dq2 = jac_qvel1_qpos1.Get(t);

    // compute qvel Jacobians
    DifferentiateDifferentiatePos(dv2dq1, dv2dq2, model, model->opt.timestep,
                                  q1, q2);

    // compute qacc Jacobians
    if (t > 1) {
      // unpack
      double* dadq0 = jac_qacc1_qpos0.Get(t - 1);
      double* dadq1 = jac_qacc1_qpos1.Get(t - 1);
      double* dadq2 = jac_qacc1_qpos2.Get(t - 1);

      // previous qvel Jacobians
      double* dv1dq0 = jac_qvel1_qpos0.Get(t - 1);
      double* dv1dq1 = jac_qvel1_qpos1.Get(t - 1);

      // dadq0 = -dv1dq0 / h
      mju_copy(dadq0, dv1dq0, nv * nv);
      mju_scl(dadq0, dadq0, -1.0 / model->opt.timestep, nv * nv);

      // dadq1 = dv2dq1 / h - dv1dq1 / h = (dv2dq1 - dv1dq1) / h
      mju_sub(dadq1, dv2dq1, dv1dq1, nv * nv);
      mju_scl(dadq1, dadq1, 1.0 / model->opt.timestep, nv * nv);

      // dadq2 = dv2dq2 / h
      mju_copy(dadq2, dv2dq2, nv * nv);
      mju_scl(dadq2, dadq2, 1.0 / model->opt.timestep, nv * nv);
    }
  }

  // stop timer
  timer_.velacc_derivatives += GetDuration(start);
}

// compute total cost
double Direct2::Cost(double* gradient, double* hessian) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // evaluate configurations
  EvaluateConfigurations();

  // derivatives
  if (gradient || hessian) {
    ConfigurationDerivative();
  }

  // start cost derivative timer
  auto start_cost_derivatives = std::chrono::steady_clock::now();

  // pool count
  int count_begin = pool_.GetCount();

  bool gradient_flag = (gradient ? true : false);
  bool hessian_flag = (hessian ? true : false);

  // -- individual cost derivatives -- //

  // sensor
  pool_.Schedule([&direct = *this, gradient_flag, hessian_flag]() {
    direct.cost_sensor_ = direct.CostSensor(
        gradient_flag ? direct.gradient_sensor_.data() : NULL,
        hessian_flag ? direct.hessian_band_sensor_.data() : NULL);
  });

  // force
  pool_.Schedule([&direct = *this, gradient_flag, hessian_flag]() {
    direct.cost_force_ = direct.CostForce(
        gradient_flag ? direct.gradient_force_.data() : NULL,
        hessian_flag ? direct.hessian_band_force_.data() : NULL);
  });

  // wait
  pool_.WaitCount(count_begin + 2);
  pool_.ResetCount();

  // total cost
  double cost = cost_sensor_ + cost_force_;

  // total gradient, hessian
  TotalGradient(gradient);
  TotalHessian(hessian);

  // counter
  cost_count_++;

  // -- stop timer -- //

  // cost time
  timer_.cost += GetDuration(start);

  // cost derivative time
  if (gradient || hessian) {
    timer_.cost_derivatives += GetDuration(start);
    timer_.cost_total_derivatives += GetDuration(start_cost_derivatives);
  }

  // total cost
  return cost;
}

// compute total gradient
void Direct2::TotalGradient(double* gradient) {
  if (!gradient) return;

  // start gradient timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model->nv;

  // zero memory
  mju_zero(gradient, ntotal_);

  // skip pinned qpos
  int tpin = 0;
  for (int t = 0; t < qpos_horizon_; t++) {
    if (pinned[t]) {
      mju_zero(gradient_sensor_.data() + t * nv, nv);
      mju_zero(gradient_force_.data() + t * nv, nv);
      continue;
    }
    mju_addTo(gradient + tpin * nv, gradient_sensor_.data() + t * nv, nv);
    mju_addTo(gradient + tpin * nv, gradient_force_.data() + t * nv, nv);
    tpin++;
  }

  // stop gradient timer
  timer_.cost_gradient += GetDuration(start);
}

// compute total Hessian
void Direct2::TotalHessian(double* hessian) {
  if (!hessian) return;

  // start Hessian timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model->nv;

  // zero memory
  mju_zero(hessian, ntotal_ * nband_);

  // skip pinned qpos
  int tpin = 0;
  for (int t = 0; t < qpos_horizon_; t++) {
    if (pinned[t]) {
      // zero row
      mju_zero(hessian_band_sensor_.data() + t * nv * nband_, nv * nband_);
      mju_zero(hessian_band_force_.data() + t * nv * nband_, nv * nband_);

      // zero diagonal (/)
      if (t + 1 < qpos_horizon_) {
        double* blk_sensor = hessian_band_sensor_.data() + (t + 1) * nv * nband_;
        double* blk_force = hessian_band_force_.data() + (t + 1) * nv * nband_;
        for (int i = 0; i < nv; i++) {
          mju_zero(blk_sensor + i * nband_ + nv, nv);
          mju_zero(blk_force + i * nband_ + nv, nv);
        }
      }
      if (t + 2 < qpos_horizon_) {
        double* blk_sensor = hessian_band_sensor_.data() + (t + 2) * nv * nband_;
        double* blk_force = hessian_band_force_.data() + (t + 2) * nv * nband_;
        for (int i = 0; i < nv; i++) {
          mju_zero(blk_sensor + i * nband_, nv);
          mju_zero(blk_force + i * nband_, nv);
        }
      }
      continue;
    }
    mju_addTo(hessian + tpin * nv * nband_, hessian_band_sensor_.data() + t * nv * nband_, nv * nband_);
    mju_addTo(hessian + tpin * nv * nband_, hessian_band_force_.data() + t * nv * nband_, nv * nband_);
    tpin++;
  }

  // stop Hessian timer
  timer_.cost_hessian += GetDuration(start);
}

// optimize qpos trajectory
void Direct2::Optimize(int qpos_horizon) {
  // start timer
  auto start_optimize = std::chrono::steady_clock::now();

  // set horizon
  qpos_horizon_ = qpos_horizon;

  // set dimensions
  ntotal_ = model->nv * qpos_horizon_;
  nband_ = 3 * model->nv;

  // set status
  gradient_norm_ = 0.0;
  search_direction_norm_ = 0.0;
  solve_status_ = kUnsolved;

  // reset timers
  ResetTimers();

  // initial cost
  cost_count_ = 0;
  cost_ = cost_initial_ = Cost(NULL, NULL);

  // print initial cost
  PrintCost();

  // ----- smoother iterations ----- //

  // reset
  iterations_smoother_ = 0;
  iterations_search_ = 0;

  // iterations
  for (; iterations_smoother_ < settings.max_smoother_iterations;
       iterations_smoother_++) {
    // evalute cost derivatives
    Cost(gradient_.data(), hessian_band_.data());

    // start timer
    auto start_search = std::chrono::steady_clock::now();

    // -- gradient -- //
    double* gradient = gradient_.data();

    // gradient tolerance check
    gradient_norm_ = mju_norm(gradient, ntotal_) / ntotal_;
    if (gradient_norm_ < settings.gradient_tolerance) {
      break;
    }

    // ----- line / curve search ----- //

    // copy qpos
    mju_copy(qpos_copy_.Data(), qpos.Data(), model->nq * qpos_horizon_);

    // initialize
    double cost_candidate = cost_;
    int iteration_search = 0;
    regularization_ = settings.regularization_initial;
    // regularization_ /= settings.regularization_scaling;
    improvement_ = -1.0;

    // -- search direction -- //

    // check regularization
    if (regularization_ >= kMaxDirectRegularization - 1.0e-6) {
      // set solve status
      solve_status_ = kMaxRegularizationFailure;

      // failure
      mju_copy(qpos.Data(), qpos_copy_.Data(), model->nq * qpos_horizon_);
      return;
    }

    // compute initial search direction
    if (!SearchDirection()) {
      mju_copy(qpos.Data(), qpos_copy_.Data(), model->nq * qpos_horizon_);
      return;  // failure
    }

    // check small search direction
    if (search_direction_norm_ < settings.search_direction_tolerance) {
      // set solve status
      solve_status_ = kSmallDirectionFailure;

      // failure
      mju_copy(qpos.Data(), qpos_copy_.Data(), model->nq * qpos_horizon_);
      return;
    }

    // backtracking until cost decrease
    while (cost_candidate >= cost_) {
      // check for max iterations
      if (iteration_search > settings.max_search_iterations) {
        // set solve status
        solve_status_ = kMaxIterationsFailure;

        // failure
        mju_copy(qpos.Data(), qpos_copy_.Data(), model->nq * qpos_horizon_);
        return;
      }

      // search type
      if (iteration_search > 0) {
        // increase regularization
        IncreaseRegularization();

        // recompute search direction
        if (!SearchDirection()) {
          mju_copy(qpos.Data(), qpos_copy_.Data(), model->nq * qpos_horizon_);
          return;  // failure
        }

        // check small search direction
        if (search_direction_norm_ < settings.search_direction_tolerance) {
          // set solve status
          solve_status_ = kSmallDirectionFailure;

          // failure
          mju_copy(qpos.Data(), qpos_copy_.Data(), model->nq * qpos_horizon_);
          return;
        }
      }

      // candidate configurations
      UpdateConfiguration(qpos, qpos_copy_, search_direction_.data(), -1.0,
                          pinned);

      // cost
      cost_candidate = Cost(NULL, NULL);

      // improvement
      improvement_ = cost_ - cost_candidate;

      // update iteration
      iteration_search++;
    }

    // increment
    iterations_search_ += iteration_search;

    // update cost
    cost_previous_ = cost_;
    cost_ = cost_candidate;

    // check cost difference
    cost_difference_ = std::abs(cost_ - cost_previous_);
    if (cost_difference_ < settings.cost_tolerance) {
      // set status
      solve_status_ = kCostDifferenceFailure;

      // failure
      return;
    }

    // ----- curve search ----- //

    // expected = g' d + 0.5 d' H d

    // g' * d
    expected_ = mju_dot(gradient_.data(), search_direction_.data(), ntotal_);

    // tmp = H * d
    double* tmp = scratch_expected_.data();
    mju_bandMulMatVec(tmp, hessian_band_.data(), search_direction_.data(),
                      ntotal_, nband_, 0, 1, true);

    // expected += 0.5 d' tmp
    expected_ += 0.5 * mju_dot(search_direction_.data(), tmp, ntotal_);

    // check for no expected decrease
    if (expected_ <= 0.0) {
      // set status
      solve_status_ = kExpectedDecreaseFailure;

      // failure
      return;
    }

    // reduction ratio
    reduction_ratio_ = improvement_ / expected_;

    printf("reduction ratio: %f\n", reduction_ratio_);

    // update regularization
    if (reduction_ratio_ > 0.75) {
      // decrease
      regularization_ =
          mju_max(kMinDirectRegularization,
                  regularization_ / settings.regularization_scaling);
    } else if (reduction_ratio_ < 0.25) {
      // increase
      regularization_ =
          mju_min(kMaxDirectRegularization,
                  regularization_ * settings.regularization_scaling);
    }

    // end timer
    timer_.search += GetDuration(start_search);

    // print cost
    PrintCost();
  }

  // stop timer
  timer_.optimize = GetDuration(start_optimize);

  // set solve status
  if (iterations_smoother_ >= settings.max_smoother_iterations) {
    solve_status_ = kMaxIterationsFailure;
  } else {
    solve_status_ = kSolved;
  }

  // status
  PrintOptimize();
}

// search direction
bool Direct2::SearchDirection() {
  // start timer
  auto search_direction_start = std::chrono::steady_clock::now();

  // -- band Hessian -- //
  int ntotal_pin = ntotal_;
  for (bool p: pinned) {
    if (p) {
      ntotal_pin -= model->nv;
    }
  }

  // unpack
  double* direction = search_direction_.data();
  double* gradient = gradient_.data();
  double* hessian_band = hessian_band_.data();
  double* hessian_band_factor = hessian_band_factor_.data();

  // -- linear system solver -- //

  // increase regularization until full rank
  double min_diag = 0.0;
  while (min_diag <= 0.0) {
    // failure
    // if (regularization_ >= kMaxDirectRegularization) {
    //   printf("min diag = %f\n", min_diag);
    //   mju_error("cost Hessian factorization failure: MAX REGULARIZATION\n");
    //   solve_status_ = kMaxRegularizationFailure;
    //   return false;
    // }

    // copy
    mju_copy(hessian_band_factor, hessian_band, ntotal_pin * nband_);

    // factorize
    min_diag = mju_cholFactorBand(hessian_band_factor, ntotal_pin, nband_, 0,
                                  regularization_, 0.0);

    // increase regularization
    if (min_diag <= 0.0) {
      IncreaseRegularization();
    }
  }

  // compute search direction
  mju_cholSolveBand(direction, hessian_band_factor, gradient, ntotal_pin,
                    nband_, 0);

  // search direction norm
  search_direction_norm_ = InfinityNorm(direction, ntotal_pin);

  // set regularization
  if (regularization_ > 0.0) {
    // configurations
    for (int i = 0; i < ntotal_pin; i++) {
      hessian_band[i * nband_ + nband_ - 1] += regularization_;
    }
  }

  // end timer
  timer_.search_direction += GetDuration(search_direction_start);
  return true;
}

// print Optimize status
void Direct2::PrintOptimize() {
  if (!settings.verbose_optimize) return;

  // title
  printf("Direct2::Optimize Status:\n\n");

  // timing
  printf("Timing:\n");

  printf("\n");
  printf("  cost : %.3f (ms) \n", 1.0e-3 * timer_.cost / cost_count_);
  printf("    - sensor: %.3f (ms) \n",
         1.0e-3 * timer_.cost_sensor / cost_count_);
  printf("    - force: %.3f (ms) \n", 1.0e-3 * timer_.cost_force / cost_count_);
  printf("    - qpos -> qvel, qacc: %.3f (ms) \n",
         1.0e-3 * timer_.cost_config_to_velacc / cost_count_);
  printf("    - prediction: %.3f (ms) \n",
         1.0e-3 * timer_.cost_prediction / cost_count_);
  printf("    - residual sensor: %.3f (ms) \n",
         1.0e-3 * timer_.residual_sensor / cost_count_);
  printf("    - residual force: %.3f (ms) \n",
         1.0e-3 * timer_.residual_force / cost_count_);
  printf("    [cost_count = %i]\n", cost_count_);
  printf("\n");
  printf("  cost derivatives [total]: %.3f (ms) \n",
         1.0e-3 * timer_.cost_derivatives);
  printf("    - inverse dynamics derivatives: %.3f (ms) \n",
         1.0e-3 * timer_.inverse_dynamics_derivatives);
  printf("    - vel., acc. derivatives: %.3f (ms) \n",
         1.0e-3 * timer_.velacc_derivatives);
  printf("    - jacobian [total]: %.3f (ms) \n",
         1.0e-3 * timer_.jacobian_total);
  printf("      < sensor: %.3f (ms) \n", 1.0e-3 * timer_.jacobian_sensor);
  printf("      < force: %.3f (ms) \n", 1.0e-3 * timer_.jacobian_force);
  printf("    - gradient, hessian [total]: %.3f (ms) \n",
         1.0e-3 * timer_.cost_total_derivatives);
  printf("      < sensor: %.3f (ms) \n",
         1.0e-3 * timer_.cost_sensor_derivatives);
  printf("      < force: %.3f (ms) \n", 1.0e-3 * timer_.cost_force_derivatives);
  printf("      < gradient assemble: %.3f (ms) \n ",
         1.0e-3 * timer_.cost_gradient);
  printf("      < hessian assemble: %.3f (ms) \n",
         1.0e-3 * timer_.cost_hessian);
  printf("\n");
  printf("  search [total]: %.3f (ms) \n", 1.0e-3 * timer_.search);
  printf("    - direction: %.3f (ms) \n", 1.0e-3 * timer_.search_direction);
  printf("    - cost: %.3f (ms) \n",
         1.0e-3 * (timer_.cost - timer_.cost / cost_count_));
  printf("      < sensor: %.3f (ms) \n",
         1.0e-3 * (timer_.cost_sensor - timer_.cost_sensor / cost_count_));
  printf("      < force: %.3f (ms) \n",
         1.0e-3 * (timer_.cost_force - timer_.cost_force / cost_count_));
  printf("      < qpos -> qvel, qacc: %.3f (ms) \n",
         1.0e-3 * (timer_.cost_config_to_velacc -
                   timer_.cost_config_to_velacc / cost_count_));
  printf(
      "      < prediction: %.3f (ms) \n",
      1.0e-3 * (timer_.cost_prediction - timer_.cost_prediction / cost_count_));
  printf(
      "      < residual sensor: %.3f (ms) \n",
      1.0e-3 * (timer_.residual_sensor - timer_.residual_sensor / cost_count_));
  printf(
      "      < residual force: %.3f (ms) \n",
      1.0e-3 * (timer_.residual_force - timer_.residual_force / cost_count_));
  printf("\n");
  printf("  TOTAL: %.3f (ms) \n", 1.0e-3 * (timer_.optimize));
  printf("\n");

  // status
  printf("Status:\n");
  printf("  search iterations: %i\n", iterations_search_);
  printf("  smoother iterations: %i\n", iterations_smoother_);
  printf("  regularization: %.6f\n", regularization_);
  printf("  gradient norm: %.6f\n", gradient_norm_);
  printf("  search direction norm: %.6f\n", search_direction_norm_);
  printf("  cost difference: %.6f\n", cost_difference_);
  printf("  solve status: %s\n", StatusString2(solve_status_).c_str());
  printf("  cost count: %i\n", cost_count_);
  printf("\n");

  // cost
  printf("Cost:\n");
  printf("  final: %.3f\n", cost_);
  printf("    - sensor: %.8f\n", cost_sensor_);
  printf("    - force: %.8f\n", cost_force_);
  printf("  <initial: %.8f>\n", cost_initial_);
  printf("\n");

  fflush(stdout);
}

// print cost
void Direct2::PrintCost() {
  if (settings.verbose_cost) {
    printf("cost (total): %.8f\n", cost_);
    printf("  sensor: %.8f\n", cost_sensor_);
    printf("  force: %.8f\n", cost_force_);
    printf("  [initial: %.8f]\n", cost_initial_);
    fflush(stdout);
  }
}

// reset timers
void Direct2::ResetTimers() {
  timer_.inverse_dynamics_derivatives = 0.0;
  timer_.velacc_derivatives = 0.0;
  timer_.jacobian_sensor = 0.0;
  timer_.jacobian_force = 0.0;
  timer_.jacobian_total = 0.0;
  timer_.cost_sensor_derivatives = 0.0;
  timer_.cost_force_derivatives = 0.0;
  timer_.cost_total_derivatives = 0.0;
  timer_.cost_gradient = 0.0;
  timer_.cost_hessian = 0.0;
  timer_.cost_derivatives = 0.0;
  timer_.cost = 0.0;
  timer_.cost_sensor = 0.0;
  timer_.cost_force = 0.0;
  timer_.cost_config_to_velacc = 0.0;
  timer_.cost_prediction = 0.0;
  timer_.residual_sensor = 0.0;
  timer_.residual_force = 0.0;
  timer_.search_direction = 0.0;
  timer_.search = 0.0;
  timer_.configuration_update = 0.0;
  timer_.optimize = 0.0;
  timer_.update_trajectory = 0.0;
}

// direct status string
std::string StatusString2(int code) {
  switch (code) {
    case kUnsolved:
      return "UNSOLVED";
    case kSearchFailure:
      return "SEACH_FAILURE";
    case kMaxIterationsFailure:
      return "MAX_ITERATIONS_FAILURE";
    case kSmallDirectionFailure:
      return "SMALL_DIRECTION_FAILURE";
    case kMaxRegularizationFailure:
      return "MAX_REGULARIZATION_FAILURE";
    case kCostDifferenceFailure:
      return "COST_DIFFERENCE_FAILURE";
    case kExpectedDecreaseFailure:
      return "EXPECTED_DECREASE_FAILURE";
    case kSolved:
      return "SOLVED";
    default:
      return "STATUS_CODE_ERROR";
  }
}

// increase regularization
void Direct2::IncreaseRegularization() {
  regularization_ = mju_min(kMaxDirectRegularization,
                            regularization_ * settings.regularization_scaling);
}

}  // namespace mjpc
