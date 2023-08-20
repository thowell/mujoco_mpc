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

#include "mjpc/tasks/dam_leg/leg.h"

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string DamLeg::XmlPath() const { return GetModelPath("dam_leg/task.xml"); }
std::string DamLeg::Name() const { return "DAM Leg"; }

// ------- Residuals for DamLeg task ------
//     Vertical: Pole angle cosine should be -1
//     Centered: Cart should be at goal position
//     Velocity: Pole angular velocity should be small
//     Control:  Control should be small
// ------------------------------------------
void DamLeg::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                  double* residual) const {
  // counter
  int counter = 0;

  // ---------- Vertical ----------
  residual[counter] = std::cos(data->qpos[1]) - 1;
  counter++;

  // ---------- Centered ----------
  residual[counter] = data->qpos[0] - parameters_[0];
  counter++;

  // ---------- Velocity ----------
  residual[counter++] = data->qvel[0];
  residual[counter++] = data->qvel[1];
  residual[counter++] = data->qvel[2];
  residual[counter++] = data->qvel[3];

  // ---------- Control ----------
  mju_copy(residual + counter, data->actuator_force, model->nu);
  counter += model->nu;

  // mju_printMat(data->actuator_force, 1, model->nu);

  // printf("nq = %i\n", model->nq);
  // printf("nv = %i\n", model->nv);
  // printf("na = %i\n", model->na);
  // printf("nu = %i\n", model->nu);

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

}  // namespace mjpc
