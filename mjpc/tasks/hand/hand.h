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

#ifndef MJPC_TASKS_HAND_HAND_H_
#define MJPC_TASKS_HAND_HAND_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
class Hand : public Task {
 public:
  Hand() { 
    std::string path = GetModelPath("hand/transition_model.xml");

    // load transition model
    constexpr int kErrorLength = 1024;
    char load_error[kErrorLength] = "";
    transition_model_ = mj_loadXML(path.c_str(), nullptr, load_error,
                        kErrorLength);
    transition_data_ = mj_makeData(transition_model_);
    mju_zero(goal_, 6);
    printf("Hand loaded.\n"); 
  }
  std::string Name() const override;
  std::string XmlPath() const override;
  // ---------- Residuals for in-hand manipulation task ---------
  //   Number of residuals: 5
  //     Residual (0): cube_position - palm_position
  //     Residual (1): cube_orientation - cube_goal_orientation
  //     Residual (2): cube linear velocity
  //     Residual (3): cube angular velocity
  //     Residual (4): control
  // ------------------------------------------------------------
  void Residual(const mjModel* model, const mjData* data,
                double* residual) const override;
  // ----- Transition for in-hand manipulation task -----
  //   If cube is within tolerance or floor ->
  //   reset cube into hand.
  // -----------------------------------------------
  void Transition(const mjModel* model, mjData* data) override;
  mjModel* transition_model_;
  mjData* transition_data_;
  int num_scramble_ = 10;
  std::vector<int> face_;
  std::vector<int> direction_;
  double goal_[6];
  std::vector<double> goal_cache_;
  int goal_index_;
};
}  // namespace mjpc

#endif  // MJPC_TASKS_HAND_HAND_H_
