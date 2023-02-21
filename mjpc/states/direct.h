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

#ifndef MJPC_STATES_DIRECT_H_
#define MJPC_STATES_DIRECT_H_

#include <mujoco/mujoco.h>

#include <shared_mutex>
#include <vector>

#include "mjpc/states/state.h"

namespace mjpc {

// data and methods for state
class DirectEstimation : public State {
 public:
  friend class StateTest;

  // constructor
  DirectEstimation() = default;

  // destructor
  ~DirectEstimation() override = default;

  // ----- methods ----- //

  // initialize settings
  void Initialize(const mjModel* model) override;

  // allocate memory
  void Allocate(const mjModel* model) override;

  // reset memory to zeros
  void Reset() override;

  // set state from data
  void Set(const mjModel* model, const mjData* data) override;

  // copy into destination
  void CopyTo(double* dst_state, double* dst_mocap, double* dst_userdata,
              double* time) const override;

  const std::vector<double>& state() const override { return state_; }
  const std::vector<double>& mocap() const override{ return mocap_; }
  const std::vector<double>& userdata() const override{ return userdata_; }
  double time() const override{ return time_; }

  // model 
  const mjModel* model_;

  // data 
  // TODO(taylor): one per history 
  mjData* data_;

  // ----- memory for direct estimation ----- //
  // trajectories 
  std::vector<double> configurations_;
  std::vector<double> velocities_;
  std::vector<double> accelerations_;
  std::vector<double> measurements_;
  std::vector<double> actions_;
  std::vector<double> times_;

  // states for difference
  std::vector<double> state1_;
  std::vector<double> state2_;

  // Jacobians
  std::vector<double> A_;        // dynamics state Jacobian
  std::vector<double> C_;        // sensor state Jacobian
  std::vector<double> E_;        // inverse dynamics current state Jacobian

  // identity matrices
  std::vector<double> I_state_;         // identity (state dimension)
  std::vector<double> I_configuration_; // identity (configuration dimension)
  std::vector<double> I_velocity_;      // identity (velocity dimension)
  std::vector<double> I_acceleration_;  // identity (acceleration dimension)

  // ----- cost (1): dynamics ----- //
  double cost1_;
  std::vector<double> cost1_state_gradient_;
  std::vector<double> cost1_state_hessian_;
  std::vector<double> residual1_;
  std::vector<double> residual1_state_jacobian_;
  std::vector<double> P_;                        // weight matrix
  std::vector<double> cost1_scratch_;

  // ----- cost (2): sensors ----- //
  double cost2_;
  std::vector<double> cost2_state_gradient_;
  std::vector<double> cost2_state_hessian_;
  std::vector<double> residual2_;
  std::vector<double> residual2_state_jacobian_;
  std::vector<double> S_;                        // weight matrix
  std::vector<double> cost2_scratch_;

  // ----- cost (3): actions ----- //
  double cost3_;
  std::vector<double> cost3_state_gradient_;
  std::vector<double> cost3_state_hessian_;
  std::vector<double> residual3_;
  std::vector<double> residual3_state_jacobian_;
  std::vector<double> R_;                        // weight matrix
  std::vector<double> cost3_scratch_;

  // total cost 
  double total_cost_;
  std::vector<double> total_cost_configuration_gradient_;
  std::vector<double> total_cost_configuration_hessian_;  
  std::vector<double> total_cost_state_gradient_;
  std::vector<double> total_cost_state_hessian_;
  std::vector<double> total_cost_state_hessian_cache_;

  // configuration to state mapping
  std::vector<double> configuration_to_state_;

  // search direction 
  std::vector<double> search_direction_;

  // solver settings 
  int iterations_;
  double gradient_tolerance_;

  int num_sensor_;               // number of sensors
  int sensor_shift_;             // sensor shift
  int horizon_;                  // estimation horizon

  // ----- methods for direct estimation ----- //

  // cost (1)
  double Cost1();
  void Cost1Gradient();
  void Cost1Hessian();
  void Residual1();
  void Residual1Jacobian();

  // cost (2)
  double Cost2();
  void Cost2Gradient();
  void Cost2Hessian();
  void Residual2();
  void Residual2Jacobian();

  // cost (3)
  double Cost3();
  void Cost3Gradient();
  void Cost3Hessian();
  void Residual3();
  void Residual3Jacobian();

  // total cost 
  double TotalCost();
  void TotalCostGradient();
  void TotalCostHessian();

 private:
  std::vector<double> state_;     // (state dimension x 1)
  std::vector<double> mocap_;     // (mocap dimension x 1)
  std::vector<double> userdata_;  // (nuserdata x 1)
  double time_;
  mutable std::shared_mutex mtx_;
};

}  // namespace mjpc

#endif  // MJPC_STATES_DIRECT_H_
