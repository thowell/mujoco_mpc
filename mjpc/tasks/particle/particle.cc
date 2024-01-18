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

#include "mjpc/tasks/particle/particle.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/estimators/estimator.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string Particle::XmlPath() const {
  return GetModelPath("particle/task_timevarying.xml");
}
std::string Particle::Name() const { return "Particle"; }

// -------- Residuals for particle task -------
//   Number of residuals: 3
//     Residual (0): position - goal_position
//     Residual (1): velocity
//     Residual (2): control
// --------------------------------------------
namespace {
void ResidualImpl(const mjModel* model, const mjData* data,
                  const double goal[2], double* residual) {
  // ----- residual (0) ----- //
  double* position = SensorByName(model, data, "position");
  mju_sub(residual, position, goal, model->nq);

  // ----- residual (1) ----- //
  double* velocity = SensorByName(model, data, "velocity");
  mju_copy(residual + 2, velocity, model->nv);

  // ----- residual (2) ----- //
  mju_copy(residual + 4, data->ctrl, model->nu);
}
}  // namespace

void Particle::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  // some Lissajous curve
  double goal[2]{0.25 * mju_sin(data->time), 0.25 * mju_cos(data->time / mjPI)};
  ResidualImpl(model, data, goal, residual);
}

void Particle::TransitionLocked(mjModel* model, mjData* data) {
  // some Lissajous curve
  double goal[2]{0.25 * mju_sin(data->time), 0.25 * mju_cos(data->time / mjPI)};

  // update mocap position
  data->mocap_pos[0] = goal[0];
  data->mocap_pos[1] = goal[1];
}

// draw task-related geometry in the scene
void Particle::ModifyScene(const mjModel* model, const mjData* data,
                           Estimator* estimator, mjvScene* scene) const {
  // color
  float color[4];
  color[0] = 1.0;
  color[1] = 0.0;
  color[2] = 1.0;
  color[3] = 0.25;

  // estimate
  double* state = estimator->State();
  double* covariance = estimator->Covariance();

  // covariance matrix
  double a = covariance[0];
  double b = covariance[1];
  double c = covariance[5];

  // ellipse dimension
  double tmp0 = 0.5 * (a + c);
  double tmp1 = mju_sqrt(0.25 * (a - c) * (a - c) + b * b);
  double l1 = tmp0 + tmp1;
  double l2 = tmp0 - tmp1;

  // ellipse rotation
  double theta;
  if (mju_abs(b) < 1.0e-8 && a >= c) {
    theta = 0.0;
  } else if (mju_abs(b) < 1.0e-8 && a < c) {
    theta = 0.5 * mjPI;
  } else {
    theta = mju_atan2(l1 - a, b);
  }

  // rotation matrix
  double mat[9] = {
    mju_cos(theta), -mju_sin(theta), 0.0,
    mju_sin(theta),  mju_cos(theta), 0.0,
    0.0           ,             0.0, 1.0,
  };

  double pos[3] = {state[0], state[1], 0.01};
  double scl = 1.0;
  double size[3] = {scl * mju_sqrt(l1), scl * mju_sqrt(l2), 0.011};

  AddGeom(scene, mjGEOM_ELLIPSOID, size, pos, mat, color);
}

std::string ParticleFixed::XmlPath() const {
  return GetModelPath("particle/task_timevarying.xml");
}
std::string ParticleFixed::Name() const { return "ParticleFixed"; }

void ParticleFixed::ResidualFn::Residual(const mjModel* model,
                                         const mjData* data,
                                         double* residual) const {
  double goal[2]{data->mocap_pos[0], data->mocap_pos[1]};
  ResidualImpl(model, data, goal, residual);
}

}  // namespace mjpc
