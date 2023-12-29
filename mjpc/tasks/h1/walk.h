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

#ifndef MJPC_TASKS_H1_WALK_H_
#define MJPC_TASKS_H1_WALK_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace h1 {

class Walk : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Walk* task) : mjpc::BaseResidualFn(task) {}
    ResidualFn(const ResidualFn&) = default;
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class Walk;
    //  ============  enums  ============

    // feet
    enum H1Foot { kFootL = 0, kFootR, kNumFoot };

    //  ============  constants  ============
    constexpr static H1Foot kFootAll[kNumFoot] = {kFootL, kFootR};

    // target head height over feet
    constexpr static double kHeightQuadruped = 1.68;  // meter

    // radius of foot geoms
    constexpr static double kFootRadius = 0.02;  // meter

    //  ============  methods  ============
    // return internal phase clock
    double GetPhase(double time) const;

    // return normalized target step height
    double StepHeight(double time, double footphase, double duty_ratio) const;

    // compute target step height for all feet
    void FootStep(double step[kNumFoot], double time) const;

    //  ============  task state variables, managed by Transition  ============
    double last_transition_time_ = -1;

    // gait-related states
    double phase_start_ = 0;
    double phase_start_time_ = 0;
    double phase_velocity_ = 0;

    //  ============  constants, computed in Reset()  ============
    int cadence_param_id_ = -1;
    int amplitude_param_id_ = -1;
    int duty_param_id_ = -1;
  };

  Walk() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

  // call base-class Reset, save task-related ids
  void ResetLocked(const mjModel* model) override;

  // draw task-related geometry in the scene
  void ModifyScene(const mjModel* model, const mjData* data,
                   mjvScene* scene) const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(residual_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  friend class ResidualFn;
  ResidualFn residual_;
};

}  // namespace h1
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_WALK_H_
