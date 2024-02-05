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

#ifndef MJPC_DIRECT_DIRECT_PLANNER_H_
#define MJPC_DIRECT_DIRECT_PLANNER_H_

#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/direct/trajectory.h"
#include "mjpc/norm.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

// defaults
inline constexpr int kMinDirectHistory = 3;  // minimum qpos trajectory length

// solve status flags
enum DirectStatus : int {
  kUnsolved = 0,
  kSearchFailure,
  kMaxIterationsFailure,
  kSmallDirectionFailure,
  kMaxRegularizationFailure,
  kCostDifferenceFailure,
  kExpectedDecreaseFailure,
  kSolved,
};

// maximum / minimum regularization
inline constexpr double kMaxDirectRegularization = 1.0e12;
inline constexpr double kMinDirectRegularization = 1.0e-12;

// ----- direct optimization with MuJoCo inverse dynamics ----- //
class Direct2 {
 public:
  // constructor
  Direct2() : pool_(NumAvailableHardwareThreads()){};
  Direct2(const mjModel* model, int qpos_horizon);

  // destructor
  virtual ~Direct2() {
    if (model) mj_deleteModel(model);
  }

  // initialize
  void Initialize(const mjModel* model, int qpos_horizon);

  // reset memory
  void Reset();

  // evaluate configurations
  void EvaluateConfigurations();

  // compute total cost
  double Cost(double* gradient, double* hessian);

  // optimize trajectory estimate
  void Optimize();

  // convert sequence of configurations to velocities, accelerations
  void ConfigurationToVelocityAcceleration();

  // compute sensor and force predictions via inverse dynamics
  void InverseDynamicsPrediction();

  // compute finite-difference qvel, qacc derivatives
  void VelocityAccelerationDerivatives();

  // compute inverse dynamics derivatives (via finite difference)
  void InverseDynamicsDerivatives();

  // evaluate configurations derivatives
  void ConfigurationDerivative();

  // ----- sensor ----- //
  // cost
  double CostSensor(double* gradient, double* hessian);

  // residual
  void ResidualSensor();

  // Jacobian blocks (dsdq0, dsdq1, dsdq2)
  void BlockSensor(int index);

  // Jacobian
  void JacobianSensor();

  // ----- force ----- //
  // cost
  double CostForce(double* gradient, double* hessian);

  // residual
  void ResidualForce();

  // Jacobian blocks (dfdq0, dfdq1, dfdq2)
  void BlockForce(int index);

  // Jacobian
  void JacobianForce();

  // ----- total cost ----- //
  // compute total gradient
  void TotalGradient(double* gradient);

  // compute total Hessian
  void TotalHessian(double* hessian);

  // search direction, returns false if regularization maxes out
  bool SearchDirection();

  // update qpos trajectory
  void UpdateConfiguration(DirectTrajectory<double>& candidate,
                           const DirectTrajectory<double>& qpos,
                           const double* search_direction, double step_size,
                           std::vector<bool>& pinned);

  // reset timers
  void ResetTimers();

  // print optimize status
  void PrintOptimize();

  // print cost
  void PrintCost();

  // increase regularization
  void IncreaseRegularization();

  // model
  mjModel* model = nullptr;

  // force weight (ndstate_)
  std::vector<double> weight_force;

  // sensor weight (nsensor_)
  std::vector<double> weight_sensor;

  // trajectories
  DirectTrajectory<double> qpos;           // nq x T
  DirectTrajectory<double> qvel;           // nv x T
  DirectTrajectory<double> qacc;           // nv x T
  DirectTrajectory<double> time;           //  1 x T
  DirectTrajectory<double> sensor;         // ns x T
  DirectTrajectory<double> sensor_target;  // ns x T
  DirectTrajectory<double> force;          // nv x T
  DirectTrajectory<double> force_target;   // nv x T

  // norms
  std::vector<NormType> norm_type_sensor;  // num_sensor

  // norm parameters
  std::vector<double>
      norm_parameters_sensor;  // num_sensor x kMaxNormParameters

  // dimensions
  int nstate_;
  int ndstate_;
  int nsensordata_;
  int nsensor_;

  int ntotal_;  // total number of decision variable
  int nvel_;    // number of qpos (derivatives) variables
  int nband_;   // cost Hessian band dimension

  // sensor indexing
  int sensor_start_;
  int sensor_start_index_;

  // data
  std::vector<UniqueMjData> data_;

  // cost
  double cost_sensor_;
  double cost_force_;
  double cost_;
  double cost_initial_;
  double cost_previous_;

  // number of planning steps
  int qpos_horizon_;  // qpos horizon (= horizon + 2)

  // qpos copy
  DirectTrajectory<double> qpos_copy_;  // nq x qpos_horizon_

  // residual
  std::vector<double> residual_sensor_;  // ns x (T - 1)
  std::vector<double> residual_force_;   // nv x (T - 2)

  // sensor Jacobian blocks (dqds, dvds, dads), (dsdq0, dsdq1, dsdq2)
  DirectTrajectory<double> jac_sensor_qpos;  // (nsensordata * nv) x T
  DirectTrajectory<double> jac_sensor_qvel;  // (nsensordata * nv) x T
  DirectTrajectory<double> jac_sensor_qacc;  // (nsensordata * nv) x T
  DirectTrajectory<double> jac_qpos_sensor;  // (nv * nsensordata) x T
  DirectTrajectory<double> jac_qvel_sensor;  // (nv * nsensordata) x T
  DirectTrajectory<double> jac_qacc_sensor;  // (nv * nsensordata) x T

  DirectTrajectory<double> jac_sensor_qpos0;    // (ns * nv) x T
  DirectTrajectory<double> jac_sensor_qpos1;    // (ns * nv) x T
  DirectTrajectory<double> jac_sensor_qpos2;    // (ns * nv) x T
  DirectTrajectory<double> jac_sensor_qpos012;  // (ns * 3 * nv) x T

  DirectTrajectory<double> jac_sensor_scratch;  // max(nv, ns) x T

  // force Jacobian blocks (dqdf, dvdf, dadf), (dfdq0, dfdq1, dfdq2)
  DirectTrajectory<double> jac_force_qpos;  // (nv * nv) x T
  DirectTrajectory<double> jac_force_qvel;  // (nv * nv) x T
  DirectTrajectory<double> jac_force_qacc;  // (nv * nv) x T

  DirectTrajectory<double> jac_force_qpos0;    // (nv * nv) x T
  DirectTrajectory<double> jac_force_qpos1;    // (nv * nv) x T
  DirectTrajectory<double> jac_force_qpos2;    // (nv * nv) x T
  DirectTrajectory<double> jac_force_qpos012;  // (nv * 3 * nv) x T

  DirectTrajectory<double> jac_force_scratch;  // (nv * nv) x T

  // qvel Jacobian wrt qpos0, qpos1 (dv1dq0, dv1dq1)
  DirectTrajectory<double> jac_qvel1_qpos0;  // (nv * nv) x T
  DirectTrajectory<double> jac_qvel1_qpos1;  // (nv * nv) x T

  // qacc Jacobian wrt qpos0, qpos1, qpos2 (da1dq0, da1dq1, da1dq2)
  DirectTrajectory<double> jac_qacc1_qpos0;  // (nv * nv) x T
  DirectTrajectory<double> jac_qacc1_qpos1;  // (nv * nv) x T
  DirectTrajectory<double> jac_qacc1_qpos2;  // (nv * nv) x T

  // norm
  std::vector<double> norm_sensor_;  // num_sensor * qpos_horizon_
  std::vector<double> norm_force_;   // nv * qpos_horizon_

  // norm gradient
  std::vector<double> norm_gradient_sensor_;  // ns * qpos_horizon_
  std::vector<double> norm_gradient_force_;   // nv * qpos_horizon_

  // norm Hessian
  std::vector<double> norm_hessian_sensor_;  // (ns * ns) x qpos_horizon_
  std::vector<double> norm_hessian_force_;   // (nv * nv) x qpos_horizon_

  // cost gradient
  std::vector<double> gradient_sensor_;  // nv * qpos_horizon_
  std::vector<double> gradient_force_;   // nv * qpos_horizon_
  std::vector<double> gradient_;         // nv * qpos_horizon_

  // cost Hessian
  std::vector<double> hessian_band_sensor_;  // (nv * qpos_horizon_) * (3 * nv)
  std::vector<double> hessian_band_force_;   // (nv * qpos_horizon_) * (3 * nv)
  std::vector<double> hessian_;              // nv * qpos_horizon_
  std::vector<double> hessian_band_;         // (nv * qpos_horizon_) * (3 * nv)
  std::vector<double> hessian_band_factor_;  // (nv * qpos_horizon_) * (3 * nv)

  // cost scratch
  std::vector<double>
      scratch_sensor_;  // 3 * nv + nsensor_data * 3 * nv + 9 * nv * nv
  std::vector<double> scratch_force_;     // 12 * nv * nv
  std::vector<double> scratch_expected_;  // nv * qpos_horizon_

  // search direction
  std::vector<double> search_direction_;  // nv * qpos_horizon_

  // status
  int cost_count_;                // number of cost evaluations
  int iterations_smoother_;       // total smoother iterations after Optimize
  int iterations_search_;         // total line search iterations
  double gradient_norm_;          // norm of cost gradient
  double regularization_;         // regularization
  double search_direction_norm_;  // search direction norm
  DirectStatus solve_status_;     // solve status
  double cost_difference_;        // cost difference: abs(cost - cost_previous)
  double improvement_;            // cost improvement
  double expected_;               // expected cost improvement
  double reduction_ratio_;  // reduction ratio: cost_improvement / expected cost
                            // improvement

  // timers
  struct DirectTimers {
    double inverse_dynamics_derivatives;
    double velacc_derivatives;
    double jacobian_sensor;
    double jacobian_force;
    double jacobian_total;
    double cost_sensor_derivatives;
    double cost_force_derivatives;
    double cost_total_derivatives;
    double cost_gradient;
    double cost_hessian;
    double cost_derivatives;
    double cost;
    double cost_sensor;
    double cost_force;
    double cost_config_to_velacc;
    double cost_prediction;
    double residual_sensor;
    double residual_force;
    double search_direction;
    double search;
    double configuration_update;
    double optimize;
    double update_trajectory;
    std::vector<double> sensor_step;
    std::vector<double> force_step;
    double update;
  } timer_;

  // threadpool
  ThreadPool pool_;

  // settings
  struct DirectSettings {
    int max_search_iterations =
        1000;  // maximum number of line search iterations
    int max_smoother_iterations =
        100;  // maximum number of smoothing iterations
    double gradient_tolerance = 1.0e-10;  // small gradient tolerance
    bool verbose_optimize = false;        // flag for printing optimize status
    bool verbose_cost = false;            // flag for printing cost
    double regularization_initial = 1.0e-12;       // initial regularization
    double regularization_scaling = mju_sqrt(10);  // regularization scaling
    bool time_scaling_force = true;                // scale force costs
    bool time_scaling_sensor = true;               // scale sensor costs
    double search_direction_tolerance = 1.0e-8;    // search direction tolerance
    double cost_tolerance = 1.0e-8;                // cost difference tolernace
    double random_init = 0.0;
  } settings;

  // finite-difference settings
  struct FiniteDifferenceSettings {
    double tolerance = 1.0e-6;
    bool flg_actuation = 1;
  } finite_difference;

  // pinned time steps
  std::vector<bool> pinned;
};

// optimizer status string
std::string StatusString2(int code);

}  // namespace mjpc

#endif  // MJPC_DIRECT_DIRECT_PLANNER_H_
