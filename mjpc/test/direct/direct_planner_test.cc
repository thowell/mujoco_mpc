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

namespace mjpc {
namespace {

// TEST(DirectPlanner, Particle2D) {
//   // load model
//   mjModel* model = LoadTestModel("estimator/particle/task.xml");
//   mjData* data = mj_makeData(model);

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

//   // delete data + model
//   mj_deleteData(data);
//   mj_deleteModel(model);
// }

TEST(DirectPlanner, Cartpole) {
  // load model
  mjModel* model = LoadTestModel("cartpole.xml");
  mjData* data = mj_makeData(model);

  data->ctrl[0] = 0.125;
  mj_step(model, data);

  printf("actuator gain: %f\n", model->actuator_gainprm[mjNGAIN * 0]);

  printf("actuator force:\n");
  mju_printMat(data->actuator_force, 1, model->nu);

  printf("actuator_moment:\n");
  mju_printMat(data->actuator_moment, model->nu, model->nv);

  printf("qfrc_actuator:\n");
  mju_printMat(data->qfrc_actuator, 1, model->nv);

  double ctrl_force[1];
  for (int i = 0; i < model->nu; i++) {
    double gain = model->actuator_gainprm[mjNGAIN*i];
    ctrl_force[i] = gain * data->ctrl[i];
  }
  double force[2];
  mju_mulMatTVec(force, data->actuator_moment, ctrl_force, model->nu, model->nv);

  printf("force:\n");
  mju_printMat(force, 1, model->nv);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
