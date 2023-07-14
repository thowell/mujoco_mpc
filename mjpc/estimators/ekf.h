// Copyright 2023 DeepMind Technologies Limited
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

#ifndef MJPC_ESTIMATORS_EKF_H_
#define MJPC_ESTIMATORS_EKF_H_

#include <mujoco/mujoco.h>

#include <mutex>
#include <vector>

#include "mjpc/utilities.h"

namespace mjpc {

class EKF {
 public:
  // constructor
  EKF() = default;
  EKF(mjModel* model) {
    Initialize(model);
    Reset();
  }

  // initialize
  void Initialize(mjModel* model);

  // reset memory
  void Reset();

  // update measurement
  void UpdateMeasurement(const double* ctrl, const double* sensor);

  // update time
  void UpdatePrediction();

  // get measurement timer (ms)
  double TimerMeasurement() const { return timer_measurement_; };

  // get prediction timer (ms)
  double TimerPrediction() const { return timer_prediction_; };

  // model
  mjModel* model;

  // state (nq + nv)
  std::vector<double> state;
  double time;

  // covariance
  std::vector<double> covariance;

  // process noise (2nv)
  std::vector<double> noise_process;

  // sensor noise (nsensordata)
  std::vector<double> noise_sensor;

  // settings
  struct Settings {
    double epsilon = 1.0e-6;
    bool flg_centered = false;
    bool auto_timestep = false;
  } settings;

  // data
  mjData* data_;

  // correction (2nv)
  std::vector<double> correction_;

 private:
  // dimensions
  int nstate_;
  int nvelocity_;

  // dynamics Jacobian (2nv x 2nv)
  std::vector<double> dynamics_jacobian_;

  // sensor Jacobian (nsensordata x 2nv)
  std::vector<double> sensor_jacobian_;

  // Kalman gain (2nv x nsensordata)
  // TODO(taylor): unused..
  std::vector<double> kalman_gain_;

  // sensor error (nsensordata)
  std::vector<double> sensor_error_;


  // timer (ms)
  double timer_measurement_;
  double timer_prediction_;

  // scratch
  std::vector<double> tmp0_;
  std::vector<double> tmp1_;
  std::vector<double> tmp2_;
  std::vector<double> tmp3_;
  std::vector<double> tmp4_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_EKF_H_
