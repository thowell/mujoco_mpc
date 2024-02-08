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

#include "mjpc/tasks/particle3d/particle.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {

std::string Particle3::XmlPath() const {
  return GetModelPath("particle3d/task_timevarying.xml");
}
std::string Particle3::Name() const { return "Particle3"; }

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
  mju_copy(residual + 3, velocity, model->nv);

  // ----- residual (2) ----- //
  mju_copy(residual + 6, data->ctrl, model->nu);
}
}  // namespace

void Particle3::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  // some Lissajous curve
  double goal[3]{0.25 * mju_sin(data->time), 0.25 * mju_cos(data->time / mjPI), 0.125};
  ResidualImpl(model, data, goal, residual);
}

void Particle3::TransitionLocked(mjModel* model, mjData* data) {
  // some Lissajous curve
  double goal[3]{0.25 * mju_sin(data->time), 0.25 * mju_cos(data->time / mjPI), 0.125};

  // update mocap position
  data->mocap_pos[0] = goal[0];
  data->mocap_pos[1] = goal[1];
  data->mocap_pos[2] = goal[2];
}

}  // namespace mjpc
