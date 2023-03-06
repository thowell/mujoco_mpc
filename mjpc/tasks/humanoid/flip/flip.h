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

#ifndef MJPC_TASKS_HUMANOID_FLIP_TASK_H_
#define MJPC_TASKS_HUMANOID_FLIP_TASK_H_

#include <mujoco/mujoco.h>

#include "mjpc/task.h"

namespace mjpc {
namespace humanoid {

class Flip : public Task {
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
    kModeHumanoid = 0,
    kModeGait,
    kModeGetUp,
    kModeFlip,
    kModeHandStand,
    kNumMode,
  };

  // feet
  enum HumanoidFoot {
    kFootLeft  = 0,
    kFootRight,
    kNumFoot
  };

  // gaits
  enum HumanoidGait {
    kGaitStand = 0,
    kGaitWalk,
    kGaitHighWalk,
    kGaitRun,
    kNumGait,
  };

  //  ============  constants  ============
  constexpr static HumanoidFoot kFootAll[kNumFoot] = {kFootLeft, kFootRight};
  constexpr static HumanoidGait kGaitAll[kNumGait] = {kGaitStand, kGaitWalk,
                                                      kGaitHighWalk, kGaitRun};

  // gait phase signature (normalized)
  constexpr static double kGaitPhase[kNumGait][kNumFoot] =
  {
  // left     right
    {0.0,     0.0},   // stand
    {0.0,     0.5},   // walk
    {0.0,     0.5},   // high walk
    {0.0,     0.5},   // run
  };

  // gait parameters, set when switching into gait
  constexpr static double kGaitParam[kNumGait][6] =
  {
  // duty ratio  cadence  amplitude  balance   upright   height
  // unitless    Hz       meter      unitless  unitless  unitless
    {0.5,         1,       0.1,       1,        1,        1},      // stand
    {0.5,         1,       0.1,       1,        1,        1},      // walk
    {0.5,         1,       0.1,       1,        1,        1},      // high walk
    {0.5,         1,       0.1,       1,        1,      0.2},      // run
  };

  // velocity ranges for automatic gait switching, meter/second
  constexpr static double kGaitAuto[kNumGait] =
  {
    0.0,  // stand
    0.0,  // walk
    0.0,  // high walk
    0.0,  // run
  };
  // notes:
  // - walk is never triggered by auto-gait

  // automatic gait switching: time constant for com speed filter
  constexpr static double kAutoGaitFilter = 0.2;    // second

  // automatic gait switching: minimum time between switches
  constexpr static double kAutoGaitMinTime = 1;     // second

  // target torso height over feet when humanoid
  constexpr static double kHeightHumanoid = 1.3;    // meter

  // radius of foot
  constexpr static double kFootRadius = 0.025;      // meter

  // flip: crouching height, from which leap is initiated
  constexpr static double kCrouchHeight = 0.8;     // meter

  // flip: leap height, beginning of flight phase
  constexpr static double kLeapHeight = 1.1;       // meter

  // flip: maximum height of flight phase
  constexpr static double kMaxHeight = 1.35;         // meter

  //  ============  methods  ============
  // return internal phase clock
  double GetPhase(double time) const;

  // return current gait
  HumanoidGait GetGait() const;

  // return normalized target step height
  double StepHeight(double time, double footphase, double duty_ratio) const;

  // compute target step height for all feet
  void FootStep(double* step, double time) const;

  // height during flip
  double FlipHeight(double time) const;

  // orientation during flip
  void FlipQuat(double quat[4], double time) const;

  //  ============  task state variables, managed by Transition  ============
  HumanoidMode current_mode_   = kModeHumanoid;
  double last_transition_time_ = -1;

  // common stage states
  double mode_start_time_  = 0;
  double position_[3]       = {0};

  // backflip states
  double ground_            = 0;
  double orientation_[4]    = {0};
  double save_gait_switch_  = 0;
  std::vector<double> save_weight_;

  // gait-related states
  double current_gait_      = kGaitStand;
  double phase_start_       = 0;
  double phase_start_time_  = 0;
  double phase_velocity_    = 0;
  double com_vel_[2]        = {0};
  double gait_switch_time_  = 0;

  //  ============  constants, computed in Reset()  ============
  int torso_body_id_         = -1;
  int pelvis_body_id_        = -1;
  int head_site_id_          = -1;
  int goal_mocap_id_         = -1;
  int gait_param_id_         = -1;
  int gait_switch_param_id_  = -1;
  int flip_dir_param_id_     = -1;
  int torso_height_param_id_ = -1;
  int speed_param_id_        = -1;
  int velscl_param_id_       = -1;
  int cadence_param_id_      = -1;
  int amplitude_param_id_    = -1;
  int duty_param_id_         = -1;
  int upright_cost_id_       = -1;
  int balance_cost_id_       = -1;
  int height_cost_id_        = -1;
  int foot_geom_id_[kNumFoot];
  int shoulder_body_id_[kNumFoot];
  int qpos_reference_id_     = -1;
  int qpos_crouch_id_        = -1;

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

#endif  // MJPC_TASKS_HUMANOID_FLIP_TASK_H_
