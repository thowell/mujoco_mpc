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

#ifndef MJPC_TASKS_H1_TASK_H_
#define MJPC_TASKS_H1_TASK_H_

#include <memory>
#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace h1 {

class Stand : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Stand* task) : mjpc::BaseResidualFn(task) {}

    // ------------------ Residuals for h1 stand task ------------
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
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class Stand;

    // return internal phase clock
    double GetPhase(double time) const;

    // return normalized target step height
    double StepHeight(double time, double footphase, double duty_ratio) const;

    // compute target step height for all feet
    void FootStep(double* step, double time) const;

    double last_transition_time_ = -1;

    // gait-related states
    double phase_start_ = 0;
    double phase_start_time_ = 0;
    double phase_velocity_ = 0;

    // parameter id
    int cadence_param_id_ = -1;
    int amplitude_param_id_ = -1;
    int duty_param_id_ = -1;
  };

  Stand() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;
  // call base-class Reset, save task-related ids
  void ResetLocked(const mjModel* model) override;

  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:

  // residual function
  ResidualFn residual_;
};

}  // namespace h1
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_TASK_H_
