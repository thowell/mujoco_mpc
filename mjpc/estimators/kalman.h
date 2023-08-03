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

#ifndef MJPC_ESTIMATORS_KALMAN_H_
#define MJPC_ESTIMATORS_KALMAN_H_

#include <mujoco/mujoco.h>

#include <mutex>
#include <vector>

#include "mjpc/estimators/estimator.h"
#include "mjpc/utilities.h"

namespace mjpc {

// maximum terms
inline constexpr int kMaxProcessNoise = 1028;
inline constexpr int kMaxSensorNoise = 1028;

// https://stanford.edu/class/ee363/lectures/kf.pdf
class Kalman {
 public:
  // constructor
  Kalman() = default;
  Kalman(const mjModel* model) {
    Initialize(model);
    Reset();
  }

  // destructor 
  ~Kalman() {
    if (data_) mj_deleteData(data_);
    if (model) mj_deleteModel(model);
  }

  // initialize
  void Initialize(const mjModel* model);

  // reset memory
  void Reset();

  // update measurement
  void UpdateMeasurement(const double* ctrl, const double* sensor);

  // update time
  void UpdatePrediction();

  // update 
  void Update(const double* ctrl, const double* sensor) {
    UpdateMeasurement(ctrl, sensor);
    UpdatePrediction();
  }

  // get state 
  double* State() { return state.data(); };

  // get covariance 
  double* Covariance() { return covariance.data(); };

  // get time 
  double& Time() { return time; };

  // get model 
  mjModel* Model() { return model; };

  // get data 
  mjData* Data() { return data_; };

  // get process noise 
  double* ProcessNoise() { return noise_process.data(); };

  // get sensor noise 
  double* SensorNoise() { return noise_sensor.data(); };

  // dimension process
  int DimensionProcess() const { return ndstate_; };

  // dimension sensor
  int DimensionSensor() const { return nsensordata_; };

  // set state
  void SetState(const double* state) {
    mju_copy(this->state.data(), state, ndstate_);
  };

  // set covariance
  void SetCovariance(const double* covariance) {
    mju_copy(this->covariance.data(), covariance, ndstate_ * ndstate_);
  };

  // get measurement timer (ms)
  double TimerMeasurement() const { return timer_measurement_; };

  // get prediction timer (ms)
  double TimerPrediction() const { return timer_prediction_; };

  // estimator-specific GUI elements
  void GUI(mjUI& ui, double* process_noise, double* sensor_noise,
           double& timestep, int& integrator);

  // estimator-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift);

  // model
  mjModel* model = nullptr;

  // state (nstate_)
  std::vector<double> state;
  double time;

  // covariance (ndstate_ x ndstate_)
  std::vector<double> covariance;

  // process noise (ndstate_)
  std::vector<double> noise_process;

  // sensor noise (nsensordata_)
  std::vector<double> noise_sensor;

  // settings
  struct Settings {
    double epsilon = 1.0e-6;
    bool flg_centered = false;
  } settings;

 private:
  // dimensions
  int nstate_;
  int ndstate_;  
  int nsensordata_;
  int nsensor_;

  // sensor indexing
  int sensor_start_;
  int sensor_start_index_;

  // data
  mjData* data_ = nullptr;

  // correction (ndstate_)
  std::vector<double> correction_;

  // sensor Jacobian (nsensordata x ndstate_)
  std::vector<double> sensor_jacobian_;

  // dynamics Jacobian (ndstate_ x ndstate_)
  std::vector<double> dynamics_jacobian_;

  // sensor error (nsensordata_)
  std::vector<double> sensor_error_;

  // timer (ms)
  double timer_measurement_;
  double timer_prediction_;

  // scratch
  std::vector<double> tmp0_;
  std::vector<double> tmp1_;
  std::vector<double> tmp2_;
  std::vector<double> tmp3_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_KALMAN_H_