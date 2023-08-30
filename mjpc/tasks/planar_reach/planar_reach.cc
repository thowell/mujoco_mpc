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

#include "mjpc/tasks/planar_reach/planar_reach.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string PlanarReach::XmlPath() const {
  return GetModelPath("planar_reach/task.xml");
}
std::string PlanarReach::Name() const { return "PlanarReach"; }

// ---------- Residuals for planar reach task ---------
//   Number of residuals: 6
//     Residual (0-1): Distance from tip to goal
//     Residual (2-3): Joint velocity
//     Residual (4-5):   Control
// -----------------------------------------------
void PlanarReach::ResidualFn::Residual(const mjModel* model, const mjData* data,
                       double* residual) const {
  // ---------- Residual (0-1) ----------
  mjtNum* tip_xpos = SensorByName(model, data, "position");
  residual[0] = parameters_[0] - tip_xpos[0];
  residual[1] = parameters_[1] - tip_xpos[2];

  // ---------- Residual (2-3) ----------
  residual[2] = data->qvel[0];
  residual[3] = data->qvel[1];

  // ---------- Residual (4-5) ----------
  residual[4] = data->ctrl[0];
  residual[5] = data->ctrl[1];
}

}  // namespace mjpc
