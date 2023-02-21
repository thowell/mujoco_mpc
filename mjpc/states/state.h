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

#ifndef MJPC_STATES_STATE_H_
#define MJPC_STATES_STATE_H_

#include <vector>

#include <mujoco/mujoco.h>

namespace mjpc {

// virtual class
class State {
 public:
  // destructor
  virtual ~State() = default;

  // ----- methods ----- //

  // initialize settings
  virtual void Initialize(const mjModel* model) = 0;

  // allocate memory
  virtual void Allocate(const mjModel* model) = 0;

  // reset memory to zeros
  virtual void Reset() = 0;

  // set state from data
  virtual void Set(const mjModel* model, const mjData* data) = 0;

  // copy into destination
  virtual void CopyTo(double* dst_state, double* dst_mocap, double* dst_userdata, double* time) const = 0;

  virtual const std::vector<double>& state() const = 0;
  virtual const std::vector<double>& mocap() const = 0;
  virtual const std::vector<double>& userdata() const = 0;
  virtual double time() const = 0;
};

}  // namespace mjpc

#endif  // MJPC_STATES_STATE_H_
