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

#include "mjpc/tasks/hand/hand.h"

#include <mujoco/mujoco.h>

#include <random>
#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string Hand::XmlPath() const {
  return GetModelPath("hand/task.xml");
}
std::string Hand::Name() const { return "Hand"; }

// ---------- Residuals for in-hand manipulation task ---------
//   Number of residuals: 4
//     Residual (0): cube_position - palm_position
//     Residual (1): cube_orientation - cube_goal_orientation
//     Residual (2): cube linear velocity
//     Residual (3): control
// ------------------------------------------------------------
void Hand::Residual(const mjModel* model, const mjData* data,
                    double* residual) const {
  int counter = 0;
  // ---------- Residual (0) ----------
  // goal position
  double* goal_position = SensorByName(model, data, "palm_position");

  // system's position
  double* position = SensorByName(model, data, "cube_position");

  // position error
  mju_sub3(residual + counter, position, goal_position);
  counter += 3;

  // ---------- Residual (1) ----------
  // goal orientation
  double* goal_orientation = SensorByName(model, data, "cube_goal_orientation");

  // system's orientation
  double* orientation = SensorByName(model, data, "cube_orientation");
  mju_normalize4(goal_orientation);

  // orientation error
  mju_subQuat(residual + counter, goal_orientation, orientation);
  counter += 3;

  // ---------- Residual (2) ----------
  double* cube_linear_velocity =
      SensorByName(model, data, "cube_linear_velocity");
  mju_copy(residual + counter, cube_linear_velocity, 3);
  counter += 3;

  // ---------- Residual (3) ----------
  mju_copy(residual + counter, data->actuator_force, model->nu);
  counter += model->nu;

  // ---------- Residual (3) ----------
  residual[counter + 0] = data->qpos[11] - goal_[0]; // red
  residual[counter + 1] = data->qpos[12] - goal_[1]; // orange
  residual[counter + 2] = data->qpos[13] - goal_[2]; // blue
  residual[counter + 3] = data->qpos[14] - goal_[3]; // green
  residual[counter + 4] = data->qpos[15] - goal_[4]; // white 
  residual[counter + 5] = data->qpos[16] - goal_[5]; // yellow
  counter += 6;
 
  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

// ----- Transition for in-hand manipulation task -----
//   If cube is within tolerance or floor ->
//   reset cube into hand.
// -----------------------------------------------
void Hand::Transition(const mjModel* model, mjData* data) {
  if (transition_model_) {
    if (mode == 0) { // wait 
      mju_copy(goal_, data->qpos + 11, 6);
    } else if (mode == 2) { // scramble
      printf("scramble!\n");

      // reset
      mju_copy(data->qpos, model->qpos0, model->nq);
      mj_resetData(transition_model_, transition_data_);

      // resize
      face_.resize(num_scramble_);
      direction_.resize(num_scramble_);
      goal_cache_.resize(6 * num_scramble_);

      // set transition model 
      for (int i = 0; i < num_scramble_; i++) {
        // copy goal face orientations
        mju_copy(goal_cache_.data() + i * 6, transition_data_->qpos, 6);

        // random face + direction
        std::random_device rd;     // Only used once to initialise (seed) engine
        std::mt19937 rng(rd());    // Random-number engine used (Mersenne-Twister in this case)
        
        std::uniform_int_distribution<int> uni_face(0, 5); // Guaranteed unbiased
        face_[i] = uni_face(rng);

        std::uniform_int_distribution<int> uni_direction(0, 1); // Guaranteed unbiased
        direction_[i] = uni_direction(rng);
        if (direction_[i] == 0) {
          direction_[i] = -1;
        }

        // set 
        for (int t = 0; t < 1500; t++) {
          transition_data_->ctrl[face_[i]] = direction_[i] * 1.57 * t / 1500;
          mj_step(transition_model_, transition_data_);
          mju_copy(data->qpos + 11, transition_data_->qpos, 86);
        }
      }

      // set face goal index 
      goal_index_ = num_scramble_ - 1;

      // set to wait
      mode = 0;
    }

    if (mode == 3) { // solve
      // set goal 
      mju_copy(goal_, goal_cache_.data() + 6 * goal_index_, 6);

      // check error
      double error[6];
      mju_sub(error, data->qpos + 11, goal_, 6);

      if (mju_norm(error, 6) < 0.1) {
        if (goal_index_ == 0) {
          // return to wait
          printf("solved!");
          mode = 0;
        } else {
          goal_index_--;
        }
      }
    }
  }

  // // find cube and floor
  // int cube = mj_name2id(model, mjOBJ_GEOM, "cube");
  // int floor = mj_name2id(model, mjOBJ_GEOM, "floor");
  // // look for contacts between the cube and the floor
  // bool on_floor = false;
  // for (int i=0; i < data->ncon; i++) {
  //   mjContact* g = data->contact + i;
  //   if ((g->geom1 == cube && g->geom2 == floor) ||
  //       (g->geom2 == cube && g->geom1 == floor)) {
  //     on_floor = true;
  //     break;
  //   }
  // }

  // double* cube_lin_vel = SensorByName(model, data, "cube_linear_velocity");
  // if (on_floor && mju_norm3(cube_lin_vel) < .001) {
  //   // reset box pose, adding a little height
  //   int cube_body = mj_name2id(model, mjOBJ_BODY, "cube");
  //   if (cube_body != -1) {
  //     int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[cube_body]];
  //     int jnt_veladr = model->jnt_dofadr[model->body_jntadr[cube_body]];
  //     mju_copy(data->qpos + jnt_qposadr, model->qpos0 + jnt_qposadr, 7);
  //     mju_zero(data->qvel + jnt_veladr, 6);
  //   }
  //   mj_forward(model, data);
  // }
}

// #include <random>

// std::random_device rd;     // Only used once to initialise (seed) engine
// std::mt19937 rng(rd());    // Random-number engine used (Mersenne-Twister in this case)
// std::uniform_int_distribution<int> uni(min,max); // Guaranteed unbiased

// auto random_integer = uni(rng);

}  // namespace mjpc
