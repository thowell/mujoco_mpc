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

  // ---------- Upright ----------
  int pelvis_id = mj_name2id(model, mjOBJ_XBODY, "pelvis");
  double* pelvis_xmat = data->xmat + 9 * pelvis_id;
  residual[counter++] = pelvis_xmat[8] - 1;
  residual[counter++] = 0;
  residual[counter++] = 0;

  // ----- Height: head feet vertical error ----- //
  double height_goal = parameters_[ParameterIndex(model, "Height Goal")];

  // feet sensor positions
  double* f1_position = SensorByName(model, data, "sp0");
  double* f2_position = SensorByName(model, data, "sp1");
  double* f3_position = SensorByName(model, data, "sp2");
  double* f4_position = SensorByName(model, data, "sp3");
  double* torso_position = SensorByName(model, data, "torso_position");
  double torso_feet_error =
      torso_position[2] -
      0.25 *
          (f1_position[2] + f2_position[2] + f3_position[2] + f4_position[2]);
  residual[counter++] = torso_feet_error - height_goal;

  // ---------- Position -------
  double* target = data->mocap_pos;
  mju_sub3(residual + counter, torso_position, target);
  residual[counter + 2] *= 0.01;
  counter += 3;

  // ---------- Gait ----------
  double step[kNumFoot];
  FootStep(step, GetPhase(data->time));
  double foot_pos[2][3];
  mju_copy3(foot_pos[0], SensorByName(model, data, "foot_left"));
  mju_copy3(foot_pos[1], SensorByName(model, data, "foot_right"));

  for (H1Foot foot : kFootAll) {
    double query[3] = {foot_pos[foot][0], foot_pos[foot][1], foot_pos[foot][2]};
    double ground_height = Ground(model, data, query);
    double height_target = ground_height + 0 * kFootRadius + step[foot];
    double height_difference = foot_pos[foot][2] - height_target;
    residual[counter++] = step[foot] ? height_difference : 0;
  }
  residual[counter++] = 0.0;
  residual[counter++] = 0.0;

  // ---------- Balance ----------
  double* compos = SensorByName(model, data, "torso_subtreecom");
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  double capture_point[3];
  double avg_foot_pos[3];
  mju_add3(avg_foot_pos, foot_pos[0], foot_pos[1]);
  mju_scl3(avg_foot_pos, avg_foot_pos, 0.5);
  double fall_time = mju_sqrt(2.0 * height_goal / 9.81);
  mju_addScl3(capture_point, compos, comvel, fall_time);
  residual[counter++] = capture_point[0] - avg_foot_pos[0];
  residual[counter++] = capture_point[1] - avg_foot_pos[1];

  // ---------- Effort ----------
  mju_scl(residual + counter, data->actuator_force, 2e-2, model->nu);
  counter += model->nu;

  // ---------- Posture ----------
  double* home = KeyQPosByName(model, data, "home");
  mju_sub(residual + counter, data->qpos + 7, home + 7, model->nu);
  // for (H1Foot foot : kFootAll) {
  //   for (int joint = 0; joint < 3; joint++) {
  //     residual[counter + 3*foot + joint] *= kJointPostureGain[joint];
  //   }
  // }

  // loosen arms
  residual[counter + 11] *= 0.1;
  residual[counter + 12] *= 0.1;
  residual[counter + 13] *= 0.1;
  residual[counter + 14] *= 0.1;
  residual[counter + 15] *= 0.1;
  residual[counter + 16] *= 0.1;
  residual[counter + 17] *= 0.1;
  residual[counter + 18] *= 0.1;

  counter += model->nu;

  // ----- joint velocity -----
  mju_copy(residual + counter, data->qvel, model->nv);
  counter += model->nv;

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
