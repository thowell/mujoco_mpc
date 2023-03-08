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

#ifndef MJPC_TASKS_HUMANOID_GAIT_TASK_H_
#define MJPC_TASKS_HUMANOID_GAIT_TASK_H_

#include <mujoco/mujoco.h>

#include "mjpc/task.h"

namespace mjpc {
namespace humanoid {

class Gait : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  // ------------------ Residuals for humanoid gait task ------------
  //   Number of residuals: 
  //   Number of parameters: 
  // ----------------------------------------------------------------
  void Residual(const mjModel* model, const mjData* data,
                double* residual) const override;

  // transition
  void Transition(const mjModel* model, mjData* data) override;

  // reset humanoid task
  void Reset(const mjModel* model) override;

  // draw task-related geometry in the scene
  void ModifyScene(const mjModel* model, const mjData* data,
                   mjvScene* scene) const override;
  //  ============  enums  ============
  // stages
  enum HumanoidMode {
    kModeStand = 0,
    kModeWalk,
    kModeFlip,
    kModeHandstand,
    kNumMode,
  };

  // feet
  enum HumanoidFoot {
    kFootLeft  = 0,
    kFootRight,
    kNumFoot
  };

  // mode weights, set when switching modes
  constexpr static double kModeWeight[kNumMode][11] =
  {
    {3.0,  0.05, 2.5, 2.5, 0.075, 0.0,  0.0,   0.0,   0.0, 0.0, 0.1},      // stand
    {1.0, 0.035, 1.0, 1.0, 0.075, 0.05, 0.1, 0.125, 0.125, 1.0, 0.0},      // walk
    {1.0,  0.01, 1.0, 1.0, 0.075,  0.0, 0.0,   0.0,   0.0, 0.0, 0.0},      // flip
    {2.5, 0.001, 2.5, 1.0, 0.05,   0.0, 0.0,   0.0,   0.0, 1.0, 0.0},      // hand stand
  };

  // mode residual parameters, set when switching into modes
  constexpr static double kModeParameter[kNumMode][5] =
  {
    {1.3, 0.0, 0.0, 0.0, 0.0},      // stand
    {1.3, 1.0, 1.0, 0.1, 0.5},      // walk
    {1.3, 1.0, 1.0, 0.1, 0.5},      // flip
    {1.3, 0.0, 1.0, 0.1, 0.5},      // hand stand
  };

  // automatic gait switching: time constant for com speed filter
  constexpr static double kAutoGaitFilter = 0.2;    // second

  // automatic gait switching: minimum time between switches
  constexpr static double kAutoGaitMinTime = 1;     // second

  // target torso height over feet when humanoid
  constexpr static double kHeightHumanoid = 1.3;  // meter

  // target torso height over feet when bipedal
  constexpr static double kHeightHandstand = 0.645;    // meter

  // radius of foot
  constexpr static double kFootRadius = 0.025;  // meter

  // below this target yaw velocity, walk straight
  constexpr static double kMinAngvel = 0.01;        // radian/second

  // flip: crouching height, from which leap is initiated
  constexpr static double kCrouchHeight = 0.8;     // meter

  // flip: leap height, beginning of flight phase
  constexpr static double kLeapHeight = 1.1;       // meter

  // flip: maximum height of flight phase
  constexpr static double kMaxHeight = 1.35;         // meter

  //  ============  methods  ============
  // return internal phase clock
  double GetPhase(double time) const;

  // return normalized target step height
  double StepHeight(double time, double footphase, double duty_ratio) const;

  // compute target step height for all feet
  void FootStep(double* step, double time) const;

  // walk horizontal position given time
  void Walk(double pos[2], double time) const;

  // height during flip
  double FlipHeight(double time) const;

  // orientation during flip
  void FlipQuat(double quat[4], double time) const;

  //  ============  task state variables, managed by Transition  ============
  HumanoidMode current_mode_   = kModeStand;
  double last_transition_time_ = -1;

  // common stage states
  double mode_start_time_  = 0.0;
  double position_[3]       = {0};

  // walk states
  double heading_[2]        = {0};
  double speed_             = 0;
  double angvel_            = 0;

  // backflip states
  double ground_            = 0;
  double orientation_[4]    = {0};
  double save_gait_switch_  = 0;
  std::vector<double> save_weight_;

  // gait-related states
  double phase_start_       = 0;
  double phase_start_time_  = 0;
  double phase_velocity_    = 0;
  double com_vel_[2]        = {0};
  double gait_switch_time_  = 0;

  // handstand states 
  double hand_stand_phase_ = 0;
  double hand_stand_time_ = 0.0;

  //  ============  constants, computed in Reset()  ============
  int torso_body_id_         = -1;
  int head_site_id_          = -1;
  int goal_mocap_id_         = -1;
  int gait_param_id_         = -1;
  int gait_switch_param_id_  = -1;
  int flip_dir_param_id_     = -1;
  int walk_mode_param_id_    = -1;
  int torso_height_param_id_ = -1;
  int speed_param_id_        = -1;
  int cadence_param_id_      = -1;
  int amplitude_param_id_    = -1;
  int duty_param_id_         = -1;
  int upright_cost_id_       = -1;
  int balance_cost_id_       = -1;
  int height_cost_id_        = -1;
  int foot_geom_id_[kNumFoot];
  int qpos_reference_id_         = -1;
  int qpos_flip_crouch_id_       = -1;
  int qpos_handstand_id_         = -1;

  // derived kinematic quantities describing flip trajectory
  double gravity_           = 0;
  double jump_vel_          = 0;
  double flight_time_       = 0;
  double jump_acc_          = 0;
  double crouch_time_       = 0;
  double leap_time_         = 0;
  double jump_time_         = 0;
  double crouch_vel_        = 0;
  double land_time_         = 0;
  double land_acc_          = 0;
  double flight_rot_vel_    = 0;
  double jump_rot_vel_      = 0;
  double jump_rot_acc_      = 0;
  double land_rot_acc_      = 0;
};

}  // namespace humanoid
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_GAIT_TASK_H_
