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

#include <algorithm>
#include <cstdio>
#include <vector>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include "gtest/gtest.h"
#include "mjpc/direct_planner/direct.h"
#include "mjpc/test/load.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

// TEST(DirectPlanner, Particle2D) {
//   // load model
//   mjModel* model = LoadTestModel("estimator/particle/task.xml");
//   mjData* data = mj_makeData(model);

//   double w = GetNumberOrDefault(0.0, model, "direct_force_weight");

//   double* ww = GetCustomNumericData(model, "direct_force_weight", 2);

//   printf("weight: %f\n", w);

//   if (ww) printf("%f %f\n", ww[0], ww[1]);

// //   int qpos_horizon = 100;
// //   Direct2 direct = Direct2(model, qpos_horizon);

// //   // set time
// //   for (int t = 0; t < direct.qpos_horizon_; t++) {
// //     direct.time.Get(t)[0] = t * direct.model->opt.timestep;
// //   }

// //   for (int t = 0; t < direct.qpos_horizon_; t++) {
// //     printf("time (t = %i) = %f\n", t, direct.time.Get(t)[0]);
// //   }

// //   // pins
// //   double qpos0[2] = {model->qpos0[0], model->qpos0[0]};
// //   double qposT[2] = {model->qpos0[0] + 0.1, model->qpos0[0] + 0.1};

// //   mju_copy(direct.qpos.Get(0), qpos0, model->nq);
// //   mju_copy(direct.qpos.Get(1), qpos0, model->nq);

// //   mju_copy(direct.qpos.Get(direct.qpos_horizon_ - 2), qposT, model->nq);
// //   mju_copy(direct.qpos.Get(direct.qpos_horizon_ - 1), qposT, model->nq);

// //   printf("qpos initialization:\n");
// //   for (int t = 0; t < direct.qpos_horizon_; t++) {
// //     printf("t = %i\n", t);
// //     mju_printMat(direct.qpos.Get(t), 1, model->nq);
// //   }

// //   // pin q0, q1 and qT-1, qT
// //   direct.pinned[0] = true;
// //   direct.pinned[1] = true;

// //   // direct.pinned[direct.qpos_horizon_ - 2] = true;
// //   // direct.pinned[direct.qpos_horizon_ - 1] = true;

// //   for (int i = 0; i < direct.pinned.size(); i++) {
// //     printf("pin (%i): %i\n", i, int(direct.pinned[i]));
// //   }

// //   // weights
// //   std::fill(direct.weight_force.begin(), direct.weight_force.end(), 1.0);
// //   std::fill(direct.weight_sensor.begin(), direct.weight_sensor.end(), 1.0e-5);

// //   printf("force weights:");
// //   mju_printMat(direct.weight_force.data(), 1, direct.weight_force.size());

// //   printf("sensor weights:");
// //   mju_printMat(direct.weight_sensor.data(), 1, direct.weight_sensor.size());

// //   printf("model timestep: %f\n", direct.model->opt.timestep);
// //   printf("qpos horizon: %i\n", direct.qpos_horizon_);

// //   // sensor target
// //   printf("nsensordata: %i\n", model->nsensordata);
// //   double sensor_target[7] = {qposT[0], qposT[1], 0.0, 0.0, 0.0, 0.0, 0.0};
// //   for (int t = 0; t < direct.qpos_horizon_; t++) {
// //     mju_copy(direct.sensor_target.Get(t), sensor_target, model->nsensordata);
// //   }

// //   // verbose settings
// //   direct.settings.verbose_optimize = true;

// //   // optimize
// //   direct.Optimize();

// //   printf("qpos optimized:\n");
// //   for (int t = 0; t < direct.qpos_horizon_; t++) {
// //     printf("t = %i\n", t);
// //     mju_printMat(direct.qpos.Get(t), 1, model->nq);
// //   }

// //   printf("force optimized:\n");
// //   for (int t = 0; t < direct.qpos_horizon_; t++) {
// //     printf("t = %i\n", t);
// //     mju_printMat(direct.force.Get(t), 1, model->nv);
// //   }

//   // delete data + model
//   mj_deleteData(data);
//   mj_deleteModel(model);
// }

// sensor
extern "C" {
void sensor(const mjModel* m, mjData* d, int stage);
}

// sensor callback
void sensor(const mjModel* model, mjData* data, int stage) {
  double* residual = data->sensordata;
  if (stage == mjSTAGE_ACC) {
    int counter = 0;
  
    mju_copy(&residual[counter], data->ctrl, model->nu);
    counter += model->nu;

    // ---------- Residual (1) -----------
    double height = SensorByName(model, data, "torso_position")[2];
    residual[counter++] = height - 1.2;

    // ---------- Residual (2) ----------
    double torso_up = SensorByName(model, data, "torso_zaxis")[2];
    residual[counter++] = torso_up - 1.0;

    // ---------- Residual (3) ----------
    double com_vel = SensorByName(model, data, "torso_subtreelinvel")[0];
    residual[counter++] = com_vel - 0.0;
  }
}

TEST(DirectPlanner, Walker) {
  // load model
  mjModel* model = LoadTestModel("../../../mjpc/tasks/walker/task.xml");
  mjData* data = mj_makeData(model);

  // sensor callback
  mjcb_sensor = sensor;

  // data->qpos[0] = 1.0;
  
  // mj_forward(model, data);
  // printf("sensordata:\n");
  // mju_printMat(data->sensordata, 1, model->nq);

  printf("nq: %i\n", model->nq);
  printf("nu: %i\n", model->nu);

  int T = 101;
  Direct2 direct;
  direct.Initialize(model);
  direct.Allocate();
  // direct.Reset();

  direct.pinned[0] = true;
  direct.pinned[1] = true;
  // direct.pinned[2] = true;

  std::vector<double> qpos0(model->nq);
  mju_copy(qpos0.data(), model->qpos0, model->nq);
  // qpos0[0] += 1.0e-1;

  direct.qpos.Set(qpos0.data(), 0);
  direct.qpos.Set(qpos0.data(), 1);

  mju_copy(data->qpos, qpos0.data(), model->nq);
  mju_zero(data->qvel, model->nv);
  mju_zero(data->ctrl, model->nu);

  for (int t = 2; t < T; t++) {
    mj_step(model, data);
    direct.qpos.Set(data->qpos, t);
  }
  
  printf("qpos (initialized)\n");
  mju_printMat(direct.qpos.Data(), T, model->nq);

  printf("force weights\n");
  direct.weight_force[0] = 1.0e6;
  direct.weight_force[1] = 1.0e6;
  direct.weight_force[2] = 1.0e6;
  std::fill(direct.weight_force.begin() + 3, direct.weight_force.end(), 1.0);
  mju_printMat(direct.weight_force.data(), 1, direct.weight_force.size());


  printf("sensor weights\n");
  std::fill(direct.weight_sensor.begin(), direct.weight_sensor.end(), 1.0);
  mju_printMat(direct.weight_sensor.data(), 1, direct.weight_sensor.size());

  direct.settings.max_search_iterations = 100;
  direct.settings.max_smoother_iterations = 100;
  direct.Optimize(T);
  direct.settings.verbose_optimize = true;
  direct.PrintOptimize();

  printf("gradient\n");
  mju_printMat(direct.gradient_.data(), 1, model->nv * T);

  printf("qpos (optimized)\n");
  mju_printMat(direct.qpos.Data(), T, model->nq);

  printf("qvel (optimized)\n");
  mju_printMat(direct.qvel.Data(), T, model->nv);

  printf("qacc (optimized)\n");
  mju_printMat(direct.qacc.Data(), T, model->nv);

  printf("force (optimized)\n");
  mju_printMat(direct.force.Data(), T, model->nv);

  // mju_copy(data->qpos, model->qpos0, model->nq);
  // mju_zero(data->qvel, model->nv);
  // mj_forward(model, data);
  // printf("acc:\n");
  // mju_printMat(data->qacc, 1, model->nv);

  // printf("qfrc_bias:\n");
  // mju_printMat(data->qfrc_bias, 1, model->nv);
  // // mju_zero(data->qacc, model->na);

  // mj_inverse(model, data);

  // printf("force:\n");
  // mju_printMat(data->qfrc_inverse, 1, model->nv);

//   int qpos_horizon = 100;
//   Direct2 direct = Direct2(model, qpos_horizon);

//   // set time
//   for (int t = 0; t < direct.qpos_horizon_; t++) {
//     direct.time.Get(t)[0] = t * direct.model->opt.timestep;
//   }

//   for (int t = 0; t < direct.qpos_horizon_; t++) {
//     printf("time (t = %i) = %f\n", t, direct.time.Get(t)[0]);
//   }

//   // pins
//   double qpos0[2] = {model->qpos0[0], model->qpos0[0]};
//   double qposT[2] = {model->qpos0[0] + 0.1, model->qpos0[0] + 0.1};

//   mju_copy(direct.qpos.Get(0), qpos0, model->nq);
//   mju_copy(direct.qpos.Get(1), qpos0, model->nq);

//   mju_copy(direct.qpos.Get(direct.qpos_horizon_ - 2), qposT, model->nq);
//   mju_copy(direct.qpos.Get(direct.qpos_horizon_ - 1), qposT, model->nq);

//   printf("qpos initialization:\n");
//   for (int t = 0; t < direct.qpos_horizon_; t++) {
//     printf("t = %i\n", t);
//     mju_printMat(direct.qpos.Get(t), 1, model->nq);
//   }

//   // pin q0, q1 and qT-1, qT
//   direct.pinned[0] = true;
//   direct.pinned[1] = true;

//   // direct.pinned[direct.qpos_horizon_ - 2] = true;
//   // direct.pinned[direct.qpos_horizon_ - 1] = true;

//   for (int i = 0; i < direct.pinned.size(); i++) {
//     printf("pin (%i): %i\n", i, int(direct.pinned[i]));
//   }

//   // weights
//   std::fill(direct.weight_force.begin(), direct.weight_force.end(), 1.0);
//   std::fill(direct.weight_sensor.begin(), direct.weight_sensor.end(), 1.0e-5);

//   printf("force weights:");
//   mju_printMat(direct.weight_force.data(), 1, direct.weight_force.size());

//   printf("sensor weights:");
//   mju_printMat(direct.weight_sensor.data(), 1, direct.weight_sensor.size());

//   printf("model timestep: %f\n", direct.model->opt.timestep);
//   printf("qpos horizon: %i\n", direct.qpos_horizon_);

//   // sensor target
//   printf("nsensordata: %i\n", model->nsensordata);
//   double sensor_target[7] = {qposT[0], qposT[1], 0.0, 0.0, 0.0, 0.0, 0.0};
//   for (int t = 0; t < direct.qpos_horizon_; t++) {
//     mju_copy(direct.sensor_target.Get(t), sensor_target, model->nsensordata);
//   }

//   // verbose settings
//   direct.settings.verbose_optimize = true;

//   // optimize
//   direct.Optimize();

//   printf("qpos optimized:\n");
//   for (int t = 0; t < direct.qpos_horizon_; t++) {
//     printf("t = %i\n", t);
//     mju_printMat(direct.qpos.Get(t), 1, model->nq);
//   }

//   printf("force optimized:\n");
//   for (int t = 0; t < direct.qpos_horizon_; t++) {
//     printf("t = %i\n", t);
//     mju_printMat(direct.force.Get(t), 1, model->nv);
//   }

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

// TEST(DirectPlanner, Cartpole) {
//   // load model
//   mjModel* model = LoadTestModel("cartpole.xml");
//   mjData* data = mj_makeData(model);

//   data->ctrl[0] = 0.125;
//   mj_step(model, data);

//   printf("actuator gain: %f\n", model->actuator_gainprm[mjNGAIN * 0]);

//   printf("actuator force:\n");
//   mju_printMat(data->actuator_force, 1, model->nu);

//   printf("actuator_moment:\n");
//   mju_printMat(data->actuator_moment, model->nu, model->nv);

//   printf("qfrc_actuator:\n");
//   mju_printMat(data->qfrc_actuator, 1, model->nv);

//   double ctrl_force[1];
//   for (int i = 0; i < model->nu; i++) {
//     double gain = model->actuator_gainprm[mjNGAIN*i];
//     ctrl_force[i] = gain * data->ctrl[i];
//   }
//   double force[2];
//   mju_mulMatTVec(force, data->actuator_moment, ctrl_force, model->nu, model->nv);

//   printf("force:\n");
//   mju_printMat(force, 1, model->nv);

//   // delete data + model
//   mj_deleteData(data);
//   mj_deleteModel(model);
// }

}  // namespace
}  // namespace mjpc
