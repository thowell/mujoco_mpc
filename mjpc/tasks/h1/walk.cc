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

#include "mjpc/tasks/h1/walk.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace h1 {

std::string Walk::XmlPath() const { return GetModelPath("h1/task_walk.xml"); }
std::string Walk::Name() const { return "H1 Walk"; }

void Walk::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                double* residual) const {
  // start counter
  int counter = 0;

  // ----- Height: head feet vertical error ----- //

  // feet sensor positions
  double* f1_position = SensorByName(model, data, "sp0");
  double* f2_position = SensorByName(model, data, "sp1");
  double* f3_position = SensorByName(model, data, "sp2");
  double* f4_position = SensorByName(model, data, "sp3");
  double* head_position = SensorByName(model, data, "head_position");
  double head_feet_error =
      head_position[2] - 0.25 * (f1_position[2] + f2_position[2] +
                                 f3_position[2] + f4_position[2]);
  residual[counter++] = head_feet_error - parameters_[ParameterIndex(model, "Height Goal")];
  
  // ----- Balance: CoM-feet xy error ----- //

  // capture point
  double* com_position = SensorByName(model, data, "torso_subtreecom");
  double* com_velocity = SensorByName(model, data, "torso_subtreelinvel");
  double kFallTime = 0.2;
  double capture_point[3] = {com_position[0], com_position[1], com_position[2]};
  mju_addToScl3(capture_point, com_velocity, kFallTime);

  // average feet xy position
  double fxy_avg[2] = {0.0};
  mju_addTo(fxy_avg, f1_position, 2);
  mju_addTo(fxy_avg, f2_position, 2);
  mju_addTo(fxy_avg, f3_position, 2);
  mju_addTo(fxy_avg, f4_position, 2);
  mju_scl(fxy_avg, fxy_avg, 0.25, 2);

  mju_subFrom(fxy_avg, capture_point, 2);
  double com_feet_distance = mju_norm(fxy_avg, 2);
  residual[counter++] = com_feet_distance;

  // ----- COM xy velocity should be 0 ----- //
  mju_copy(&residual[counter], com_velocity, 2);
  counter += 2;

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
  counter += model->nv - 6;

  // ----- action ----- //
  mju_copy(&residual[counter], data->actuator_force, model->nu);
  counter += model->nu;

  // ----- nominal ----- //
  // if (head_position[2] < 1.25) {
  //   mju_zero(residual + counter, 19);
  // } else {
  mju_sub(residual + counter, data->qpos + 7, model->key_qpos + 7, 19);
  // }
  counter += 19;

  // ---------- Gait ----------
  double step[kNumFoot];
  FootStep(step, GetPhase(data->time));
  double foot_pos[2][3];
  mju_copy3(foot_pos[0], SensorByName(model, data, "foot_left"));
  mju_copy3(foot_pos[1], SensorByName(model, data, "foot_right"));
  double foot_min_height[2];
  foot_min_height[0] = std::min(SensorByName(model, data, "sp0")[2],
                                SensorByName(model, data, "sp1")[2]);
  foot_min_height[1] = std::min(SensorByName(model, data, "sp2")[2],
                                SensorByName(model, data, "sp3")[2]);

  for (H1Foot foot : kFootAll) {
    double query[3] = {foot_pos[foot][0], foot_pos[foot][1], foot_pos[foot][2]};
    double ground_height = Ground(model, data, query);
    double height_target = ground_height + kFootRadius + step[foot];
    double height_difference = foot_min_height[foot] - height_target;
    residual[counter++] = step[foot] ? height_difference : 0;
  }

  // goal
  double* torso_position = SensorByName(model, data, "torso_position");
  mju_sub(residual + counter, torso_position, data->mocap_pos, 2);
  counter += 2;

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

//  ============  transition  ============
void Walk::TransitionLocked(mjModel* model, mjData* data) {
  // ---------- handle mjData reset ----------
  if (data->time < residual_.last_transition_time_ ||
      residual_.last_transition_time_ == -1) {
    residual_.last_transition_time_ = residual_.phase_start_time_ =
        residual_.phase_start_ = data->time;
  }

  // ---------- handle phase velocity change ----------
  double phase_velocity = 2 * mjPI * parameters[residual_.cadence_param_id_];
  if (phase_velocity != residual_.phase_velocity_) {
    residual_.phase_start_ = residual_.GetPhase(data->time);
    residual_.phase_start_time_ = data->time;
    residual_.phase_velocity_ = phase_velocity;
  }

  residual_.last_transition_time_ = data->time;
}

// draw task-related geometry in the scene
void Walk::ModifyScene(const mjModel* model, const mjData* data,
                       mjvScene* scene) const {}

//  ============  task-state utilities  ============
// save task-related ids
void Walk::ResetLocked(const mjModel* model) {
  // ----------  task identifiers  ----------
  residual_.cadence_param_id_ = ParameterIndex(model, "Cadence");
  residual_.amplitude_param_id_ = ParameterIndex(model, "Amplitude");
  residual_.duty_param_id_ = ParameterIndex(model, "Duty ratio");
}

// return phase as a function of time
double Walk::ResidualFn::GetPhase(double time) const {
  return phase_start_ + (time - phase_start_time_) * phase_velocity_;
}

// return normalized target step height
double Walk::ResidualFn::StepHeight(double time, double footphase,
                                    double duty_ratio) const {
  double angle = fmod(time + mjPI - footphase, 2 * mjPI) - mjPI;
  double value = 0;
  if (duty_ratio < 1) {
    angle *= 0.5 / (1 - duty_ratio);
    value = mju_cos(mju_clip(angle, -mjPI / 2, mjPI / 2));
  }
  return mju_abs(value) < 1e-6 ? 0.0 : value;
}

// compute target step height for all feet
void Walk::ResidualFn::FootStep(double step[kNumFoot], double time) const {
  double amplitude = parameters_[amplitude_param_id_];
  double duty_ratio = parameters_[duty_param_id_];
  double gait_phase[2] = {0.0, 0.5};
  for (H1Foot foot : kFootAll) {
    double footphase = 2 * mjPI * gait_phase[foot];
    step[foot] = amplitude * StepHeight(time, footphase, duty_ratio);
  }
}

}  // namespace h1
}  // namespace mjpc
