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

#ifndef MJPC_TASKS_HUMANOID_HUMANOID_H_
#define MJPC_TASKS_HUMANOID_HUMANOID_H_

#include <mujoco/mujoco.h>

namespace mjpc {
struct Humanoid {
  // -------------- Residuals for humanoid stand task ----------------
  //   Number of residuals: 6
  //     Residual (0): control
  //     Residual (1): COM_xy - average(feet position)_xy
  //     Residual (2): torso_xy - COM_xy
  //     Residual (3): head_z - feet^{(i)}_position_z - height_goal
  //     Residual (4): velocity COM_xy
  //     Residual (5): joint velocity
  //   Number of parameters: 1
  //     Parameter (0): height_goal
  // ----------------------------------------------------------------
  static void ResidualStand(const double* parameters, const mjModel* model,
                            const mjData* data, double* residual);

  // ------------------ Residuals for humanoid walk task ------------
  //   Number of residuals:
  //     Residual (0): torso height
  //     Residual (1): pelvis-feet aligment
  //     Residual (2): balance
  //     Residual (3): upright
  //     Residual (4): posture
  //     Residual (5): walk
  //     Residual (6): move feet
  //     Residual (7): control
  //   Number of parameters:
  //     Parameter (0): torso height goal
  //     Parameter (1): speed goal
  // ----------------------------------------------------------------
  static void ResidualWalk(const double* parameters, const mjModel* model,
                            const mjData* data, double* residual);

  // -------------- Residuals for humanoid tracking task ------------
  //   Number of residuals: TODO(hartikainen)
  //     Residual (0): TODO(hartikainen)
  //   Number of parameters: TODO(hartikainen)
  //     Parameter (0): TODO(hartikainen)
  // ----------------------------------------------------------------
  static void ResidualTrackSequence(const double* parameters, const mjModel* model,
                                    const mjData* data, double* residual);

  // ------------ Transition for humanoid tracking task -------------
  //   TODO(hartikainen)
  // ----------------------------------------------------------------
  static int TransitionTrackSequence(int state, const mjModel* model, mjData* data);

};
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_HUMANOID_H_
