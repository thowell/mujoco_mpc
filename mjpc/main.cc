// // Copyright 2021 DeepMind Technologies Limited
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// #include <atomic>
// #include <chrono>
// #include <cstdio>
// #include <cstdlib>
// #include <cstring>
// #include <iostream>
// #include <memory>
// #include <mutex>
// #include <sstream>
// #include <string>
// #include <string_view>
// #include <thread>

// #include <absl/flags/flag.h>
// #include <absl/flags/parse.h>
// #include <absl/strings/match.h>
// #include <mujoco/mujoco.h>
// #include "glfw_dispatch.h"
// #include "array_safety.h"
// #include "agent.h"
// #include "planners/include.h"
// #include "simulate.h"  // mjpc fork
// #include "tasks/tasks.h"
// #include "threadpool.h"
// #include "utilities.h"

// ABSL_FLAG(std::string, task, "", "Which model to load on startup.");

// namespace {
// namespace mj = ::mujoco;
// namespace mju = ::mujoco::util_mjpc;

// using ::mujoco::Glfw;

// // maximum mis-alignment before re-sync (simulation seconds)
// const double syncMisalign = 0.1;

// // fraction of refresh available for simulation
// const double simRefreshFraction = 0.7;

// // load error string length
// const int kErrorLength = 1024;

// // model and data
// mjModel* m = nullptr;
// mjData* d = nullptr;

// // control noise variables
// mjtNum* ctrlnoise = nullptr;

// // --------------------------------- callbacks ---------------------------------
// auto sim = std::make_unique<mj::Simulate>();

// // controller
// extern "C" {
// void controller(const mjModel* m, mjData* d);
// }

// // controller callback
// void controller(const mjModel* m, mjData* data) {
//   // if agent, skip
//   if (data != d) {
//     return;
//   }
//   // if simulation:
//   if (sim->agent.action_enabled) {
//     sim->agent.ActivePlanner().ActionFromPolicy(
//         data->ctrl, &sim->agent.ActiveState().state()[0],
//         sim->agent.ActiveState().time());
//   }
//   // if noise
//   if (!sim->agent.allocate_enabled && sim->uiloadrequest.load() == 0 &&
//       sim->ctrlnoisestd) {
//     for (int j = 0; j < sim->m->nu; j++) {
//       data->ctrl[j] += ctrlnoise[j];
//     }
//   }
// }

// // sensor
// extern "C" {
// void sensor(const mjModel* m, mjData* d, int stage);
// }

// // sensor callback
// void sensor(const mjModel* model, mjData* data, int stage) {
//   if (stage == mjSTAGE_ACC) {
//     if (!sim->agent.allocate_enabled && sim->uiloadrequest.load() == 0) {
//       // users sensors must be ordered first and sequentially
//       sim->agent.task().Residuals(model, data, data->sensordata);
//     }
//   }
// }

// //--------------------------------- simulation ---------------------------------

// mjModel* LoadModel(const char* file, mj::Simulate& sim) {
//   // this copy is needed so that the mju::strlen call below compiles
//   char filename[mj::Simulate::kMaxFilenameLength];
//   mju::strcpy_arr(filename, file);

//   // make sure filename is not empty
//   if (!filename[0]) {
//     return nullptr;
//   }

//   // load and compile
//   char loadError[kErrorLength] = "";
//   mjModel* mnew = 0;
//   if (mju::strlen_arr(filename) > 4 &&
//       !std::strncmp(
//           filename + mju::strlen_arr(filename) - 4, ".mjb",
//           mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4)) {
//     mnew = mj_loadModel(filename, nullptr);
//     if (!mnew) {
//       mju::strcpy_arr(loadError, "could not load binary model");
//     }
//   } else {
//     mnew = mj_loadXML(filename, nullptr, loadError,
//                       mj::Simulate::kMaxFilenameLength);
//     // remove trailing newline character from loadError
//     if (loadError[0]) {
//       int error_length = mju::strlen_arr(loadError);
//       if (loadError[error_length - 1] == '\n') {
//         loadError[error_length - 1] = '\0';
//       }
//     }
//   }

//   mju::strcpy_arr(sim.loadError, loadError);

//   if (!mnew) {
//     std::printf("%s\n", loadError);
//     return nullptr;
//   }

//   // compiler warning: print and pause
//   if (loadError[0]) {
//     // mj_forward() below will print the warning message
//     std::printf("Model compiled, but simulation warning (paused):\n  %s\n",
//                 loadError);
//     sim.run = 0;
//   }

//   return mnew;
// }

// // returns the index of a task, searching by name, case-insensitive.
// // -1 if not found.
// int TaskIdByName(std::string_view name) {
//   int i = 0;
//   for (const auto& task : mjpc::kTasks) {
//     if (absl::EqualsIgnoreCase(name, task.name)) {
//       return i;
//     }
//     i++;
//   }
//   return -1;
// }

// // simulate in background thread (while rendering in main thread)
// void PhysicsLoop(mj::Simulate& sim) {
//   // cpu-sim syncronization point
//   double syncCPU = 0;
//   mjtNum syncSim = 0;

//   // run until asked to exit
//   while (!sim.exitrequest.load()) {
//     if (sim.droploadrequest.load()) {
//       mjModel* mnew = LoadModel(sim.dropfilename, sim);
//       sim.droploadrequest.store(false);

//       mjData* dnew = nullptr;
//       if (mnew) dnew = mj_makeData(mnew);
//       if (dnew) {
//         sim.load(sim.dropfilename, mnew, dnew, true);

//         m = mnew;
//         d = dnew;
//         mj_forward(m, d);

//         // allocate ctrlnoise
//         free(ctrlnoise);
//         ctrlnoise = (mjtNum*)malloc(sizeof(mjtNum) * m->nu);
//         mju_zero(ctrlnoise, m->nu);
//       }
//     }

//     // ----- task reload ----- //
//     if (sim.uiloadrequest.load() == 1) {
//       // get new model + task
//       std::string filename =
//           mjpc::GetModelPath(mjpc::kTasks[sim.agent.task().id].xml_path);

//       // copy model + task file path into simulate
//       mju::strcpy_arr(sim.filename, filename.c_str());
//       mjModel* mnew = LoadModel(sim.filename, sim);
//       mjData* dnew = nullptr;
//       if (mnew) dnew = mj_makeData(mnew);
//       if (dnew) {
//         sim.load(sim.filename, mnew, dnew, true);
//         m = mnew;
//         d = dnew;
//         mj_forward(m, d);

//         // allocate ctrlnoise
//         free(ctrlnoise);
//         ctrlnoise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * m->nu));
//         mju_zero(ctrlnoise, m->nu);
//       }

//       // agent
//       {
//         std::ostringstream concatenated_task_names;
//         for (const auto& task : mjpc::kTasks) {
//           concatenated_task_names << task.name << '\n';
//         }
//         const auto& task = mjpc::kTasks[sim.agent.task().id];
//         sim.agent.Initialize(m, d, concatenated_task_names.str(),
//                              mjpc::kPlannerNames, task.residual,
//                              task.transition);
//       }
//       sim.agent.Allocate();
//       sim.agent.Reset();
//       sim.agent.PlotInitialize();
//     }

//     // reload model to refresh UI
//     if (sim.uiloadrequest.load() == 1) {
//       // get new model + task
//       std::string filename =
//           mjpc::GetModelPath(mjpc::kTasks[sim.agent.task().id].xml_path);

//       // copy model + task file path into simulate
//       mju::strcpy_arr(sim.filename, filename.c_str());
//       mjModel* mnew = LoadModel(sim.filename, sim);
//       mjData* dnew = nullptr;
//       if (mnew) dnew = mj_makeData(mnew);
//       if (dnew) {
//         sim.load(sim.filename, mnew, dnew, true);
//         m = mnew;
//         d = dnew;
//         mj_forward(m, d);

//         // allocate ctrlnoise
//         free(ctrlnoise);
//         ctrlnoise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * m->nu));
//         mju_zero(ctrlnoise, m->nu);
//       }

//       // set initial configuration via keyframe
//       double* qpos_key = mjpc::KeyFrameByName(sim.mnew, sim.dnew, "home");
//       if (qpos_key) {
//         mju_copy(sim.dnew->qpos, qpos_key, sim.mnew->nq);
//       }

//       // decrement counter
//       sim.uiloadrequest.fetch_sub(1);
//     }

//     // reload GUI
//     if (sim.uiloadrequest.load() == -1) {
//       sim.load(sim.filename, sim.m, sim.d, false);
//       sim.uiloadrequest.fetch_add(1);
//     }
//     // ----------------------- //

//     // sleep for 1 ms or yield, to let main thread run
//     //  yield results in busy wait - which has better timing but kills battery
//     //  life
//     if (sim.run && sim.busywait) {
//       std::this_thread::yield();
//     } else {
//       std::this_thread::sleep_for(std::chrono::milliseconds(1));
//     }

//     {
//       // lock the sim mutex
//       const std::lock_guard<std::mutex> lock(sim.mtx);

//       // run only if model is present
//       if (m) {
//         // running
//         if (sim.run) {
//           // record cpu time at start of iteration
//           double startCPU = Glfw().glfwGetTime();

//           // elapsed CPU and simulation time since last sync
//           double elapsedCPU = startCPU - syncCPU;
//           double elapsedSim = d->time - syncSim;

//           // inject noise
//           if (sim.ctrlnoisestd) {
//             // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
//             mjtNum rate = mju_exp(-m->opt.timestep / sim.ctrlnoiserate);
//             mjtNum scale = sim.ctrlnoisestd * mju_sqrt(1 - rate * rate);

//             for (int i = 0; i < m->nu; i++) {
//               // update noise
//               ctrlnoise[i] =
//                   rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

//               // apply noise
//               // d->ctrl[i] += ctrlnoise[i]; // noise is now added in controller
//               // callback
//             }
//           }

//           // requested slow-down factor
//           double slowdown = 100 / sim.percentRealTime[sim.realTimeIndex];

//           // misalignment condition: distance from target sim time is bigger
//           // than syncmisalign
//           bool misaligned =
//               mju_abs(elapsedCPU / slowdown - elapsedSim) > syncMisalign;

//           // out-of-sync (for any reason): reset sync times, step
//           if (elapsedSim < 0 || elapsedCPU < 0 || syncCPU == 0 || misaligned ||
//               sim.speedChanged) {
//             // re-sync
//             syncCPU = startCPU;
//             syncSim = d->time;
//             sim.speedChanged = false;

//             // clear old perturbations, apply new
//             mju_zero(d->xfrc_applied, 6 * m->nbody);
//             sim.applyposepertubations(0);  // move mocap bodies only
//             sim.applyforceperturbations();

//             // run single step, let next iteration deal with timing
//             mj_step(m, d);
//           } else {  // in-sync: step until ahead of cpu
//             bool measured = false;
//             mjtNum prevSim = d->time;
//             double refreshTime = simRefreshFraction / sim.refreshRate;

//             // step while sim lags behind cpu and within refreshTime
//             while ((d->time - syncSim) * slowdown <
//                        (Glfw().glfwGetTime() - syncCPU) &&
//                    (Glfw().glfwGetTime() - startCPU) < refreshTime) {
//               // measure slowdown before first step
//               if (!measured && elapsedSim) {
//                 sim.measuredSlowdown = elapsedCPU / elapsedSim;
//                 measured = true;
//               }

//               // clear old perturbations, apply new
//               mju_zero(d->xfrc_applied, 6 * m->nbody);
//               sim.applyposepertubations(0);  // move mocap bodies only
//               sim.applyforceperturbations();

//               // call mj_step
//               mj_step(m, d);

//               // break if reset
//               if (d->time < prevSim) {
//                 break;
//               }
//             }
//           }
//         } else {  // paused
//           // apply pose perturbation
//           sim.applyposepertubations(1);  // move mocap and dynamic bodies

//           // run mj_forward, to update rendering and joint sliders
//           mj_forward(m, d);
//         }

//         // transition
//         if (sim.agent.task().transition_status == 1 &&
//             sim.uiloadrequest.load() == 0) {
//           sim.agent.task().Transition(m, d);
//         }
//       }
//     }  // release sim.mtx

//     // state
//     if (sim.uiloadrequest.load() == 0) {
//       sim.agent.ActiveState().Set(m, d);
//     }
//   }
// }
// }  // namespace

// // ---------------------------- physics_thread ---------------------------------

// void PhysicsThread(mj::Simulate* sim, const char* filename) {
//   // request loadmodel if file given (otherwise drag-and-drop)
//   if (filename != nullptr) {
//     m = LoadModel(filename, *sim);
//     if (m) d = mj_makeData(m);
//     if (d) {
//       sim->load(filename, m, d, true);
//       mj_forward(m, d);

//       // allocate ctrlnoise
//       free(ctrlnoise);
//       ctrlnoise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * m->nu));
//       mju_zero(ctrlnoise, m->nu);
//     }
//   }

//   PhysicsLoop(*sim);

//   // delete everything we allocated
//   free(ctrlnoise);
//   mj_deleteData(d);
//   mj_deleteModel(m);
// }

// // ------------------------------- main ----------------------------------------

// // machinery for replacing command line error by a macOS dialog box
// // when running under Rosetta
// #if defined(__APPLE__) && defined(__AVX__)
// extern void DisplayErrorDialogBox(const char* title, const char* msg);
// static const char* rosetta_error_msg = nullptr;
// __attribute__((used, visibility("default")))
// extern "C" void _mj_rosettaError(const char* msg) {
//   rosetta_error_msg = msg;
// }
// #endif

// // run event loop
// int main(int argc, char** argv) {
//   // display an error if running on macOS under Rosetta 2
// #if defined(__APPLE__) && defined(__AVX__)
//   if (rosetta_error_msg) {
//     DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
//     std::exit(1);
//   }
// #endif

//   absl::ParseCommandLine(argc, argv);
//   std::printf("MuJoCo version %s\n", mj_versionString());
//   if (mjVERSION_HEADER != mj_version()) {
//     mju_error("Headers and library have Different versions");
//   }

//   // threads
//   printf("Hardware threads: %i\n", mjpc::NumAvailableHardwareThreads());

//   // init GLFW
//   if (!Glfw().glfwInit()) {
//     mju_error("could not initialize GLFW");
//   }

//   std::string task = absl::GetFlag(FLAGS_task);
//   if (!task.empty()) {
//     sim->agent.task().id = TaskIdByName(task);
//     if (sim->agent.task().id == -1) {
//       std::cerr << "Invalid --task flag: '" << task << "'. Valid values:\n";
//       for (const auto& task : mjpc::kTasks) {
//         std::cerr << '\t' << task.name << '\n';
//       }
//       mju_error("Invalid --task flag.");
//       return -1;
//     }
//   }

//   // default model + task
//   std::string filename =
//       mjpc::GetModelPath(mjpc::kTasks[sim->agent.task().id].xml_path);

//   // copy model + task file path into simulate
//   mju::strcpy_arr(sim->filename, filename.c_str());

//   // load model + make data
//   m = LoadModel(sim->filename, *sim);
//   if (m) d = mj_makeData(m);
//   sim->mnew = m;
//   sim->dnew = d;

//   sim->delete_old_m_d = true;
//   sim->loadrequest = 2;

//   // control noise
//   free(ctrlnoise);
//   ctrlnoise = (mjtNum*)malloc(sizeof(mjtNum) * m->nu);
//   mju_zero(ctrlnoise, m->nu);

//   // agent
//   {
//     std::ostringstream concatenated_task_names;
//     for (const auto& task : mjpc::kTasks) {
//       concatenated_task_names << task.name << '\n';
//     }
//     const auto& task = mjpc::kTasks[sim->agent.task().id];
//     sim->agent.Initialize(m, d, concatenated_task_names.str(),
//                           mjpc::kPlannerNames, task.residual,
//                           task.transition);
//   }
//   sim->agent.Allocate();
//   sim->agent.Reset();
//   sim->agent.PlotInitialize();

//   // planning threads
//   printf("Agent threads: %i\n", sim->agent.max_threads());

//   // set initial configuration via keyframe
//   double* qpos_key = mjpc::KeyFrameByName(sim->mnew, sim->dnew, "home");
//   if (qpos_key) {
//     mju_copy(sim->dnew->qpos, qpos_key, sim->mnew->nq);
//   }

//   // set control callback
//   mjcb_control = controller;

//   // set sensor callback
//   mjcb_sensor = sensor;

//   // start physics thread
//   mjpc::ThreadPool physics_pool(1);
//   physics_pool.Schedule([]() { PhysicsLoop(*sim.get()); });

//   // start plan thread
//   mjpc::ThreadPool plan_pool(1);
//   plan_pool.Schedule(
//       []() { sim->agent.Plan(sim->exitrequest, sim->uiloadrequest); });

//   // start simulation UI loop (blocking call)
//   sim->renderloop();
//   // terminate GLFW (crashes with Linux NVidia drivers)
// #if defined(__APPLE__) || defined(_WIN32)
//   Glfw().glfwTerminate();
// #endif

//   return 0;
// }



//

// Copyright 2021 DeepMind Technologies Limited
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

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/match.h>
#include <mujoco/mujoco.h>
#include "glfw_dispatch.h"
#include "array_safety.h"
#include "agent.h"
#include "planners/include.h"
#include "simulate.h"  // mjpc fork
#include "tasks/tasks.h"
#include "threadpool.h"
#include "utilities.h"

// CMA-ES
#include <absl/random/random.h>
#include <numeric>      
#include <algorithm>    

// ABSL_FLAG(std::string, task, "", "Which model to load on startup.");

// namespace {
// namespace mj = ::mujoco;
// namespace mju = ::mujoco::util_mjpc;

// using ::mujoco::Glfw;

// // maximum mis-alignment before re-sync (simulation seconds)
// const double syncMisalign = 0.1;

// // fraction of refresh available for simulation
// const double simRefreshFraction = 0.7;

// // load error string length
// const int kErrorLength = 1024;

// // model and data
// mjModel* m = nullptr;
// mjData* d = nullptr;

// // control noise variables
// mjtNum* ctrlnoise = nullptr;

// // --------------------------------- callbacks ---------------------------------
// auto sim = std::make_unique<mj::Simulate>();

// // controller
// extern "C" {
// void controller(const mjModel* m, mjData* d);
// }

// // controller callback
// void controller(const mjModel* m, mjData* data) {
//   // if agent, skip
//   if (data != d) {
//     return;
//   }
//   // if simulation:
//   if (sim->agent.action_enabled) {
//     sim->agent.ActivePlanner().ActionFromPolicy(
//         data->ctrl, &sim->agent.ActiveState().state()[0],
//         sim->agent.ActiveState().time());
//   }
//   // if noise
//   if (!sim->agent.allocate_enabled && sim->uiloadrequest.load() == 0 &&
//       sim->ctrlnoisestd) {
//     for (int j = 0; j < sim->m->nu; j++) {
//       data->ctrl[j] += ctrlnoise[j];
//     }
//   }
// }

// // sensor
// extern "C" {
// void sensor(const mjModel* m, mjData* d, int stage);
// }

// // sensor callback
// void sensor(const mjModel* model, mjData* data, int stage) {
//   if (stage == mjSTAGE_ACC) {
//     if (!sim->agent.allocate_enabled && sim->uiloadrequest.load() == 0) {
//       // users sensors must be ordered first and sequentially
//       sim->agent.task().Residuals(model, data, data->sensordata);
//     }
//   }
// }

// //--------------------------------- simulation ---------------------------------

// mjModel* LoadModel(const char* file, mj::Simulate& sim) {
//   // this copy is needed so that the mju::strlen call below compiles
//   char filename[mj::Simulate::kMaxFilenameLength];
//   mju::strcpy_arr(filename, file);

//   // make sure filename is not empty
//   if (!filename[0]) {
//     return nullptr;
//   }

//   // load and compile
//   char loadError[kErrorLength] = "";
//   mjModel* mnew = 0;
//   if (mju::strlen_arr(filename) > 4 &&
//       !std::strncmp(
//           filename + mju::strlen_arr(filename) - 4, ".mjb",
//           mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4)) {
//     mnew = mj_loadModel(filename, nullptr);
//     if (!mnew) {
//       mju::strcpy_arr(loadError, "could not load binary model");
//     }
//   } else {
//     mnew = mj_loadXML(filename, nullptr, loadError,
//                       mj::Simulate::kMaxFilenameLength);
//     // remove trailing newline character from loadError
//     if (loadError[0]) {
//       int error_length = mju::strlen_arr(loadError);
//       if (loadError[error_length - 1] == '\n') {
//         loadError[error_length - 1] = '\0';
//       }
//     }
//   }

//   mju::strcpy_arr(sim.loadError, loadError);

//   if (!mnew) {
//     std::printf("%s\n", loadError);
//     return nullptr;
//   }

//   // compiler warning: print and pause
//   if (loadError[0]) {
//     // mj_forward() below will print the warning message
//     std::printf("Model compiled, but simulation warning (paused):\n  %s\n",
//                 loadError);
//     sim.run = 0;
//   }

//   return mnew;
// }

// // returns the index of a task, searching by name, case-insensitive.
// // -1 if not found.
// int TaskIdByName(std::string_view name) {
//   int i = 0;
//   for (const auto& task : mjpc::kTasks) {
//     if (absl::EqualsIgnoreCase(name, task.name)) {
//       return i;
//     }
//     i++;
//   }
//   return -1;
// }

// // simulate in background thread (while rendering in main thread)
// void PhysicsLoop(mj::Simulate& sim) {
//   // cpu-sim syncronization point
//   double syncCPU = 0;
//   mjtNum syncSim = 0;

//   // run until asked to exit
//   while (!sim.exitrequest.load()) {
//     if (sim.droploadrequest.load()) {
//       mjModel* mnew = LoadModel(sim.dropfilename, sim);
//       sim.droploadrequest.store(false);

//       mjData* dnew = nullptr;
//       if (mnew) dnew = mj_makeData(mnew);
//       if (dnew) {
//         sim.load(sim.dropfilename, mnew, dnew, true);

//         m = mnew;
//         d = dnew;
//         mj_forward(m, d);

//         // allocate ctrlnoise
//         free(ctrlnoise);
//         ctrlnoise = (mjtNum*)malloc(sizeof(mjtNum) * m->nu);
//         mju_zero(ctrlnoise, m->nu);
//       }
//     }

//     // ----- task reload ----- //
//     if (sim.uiloadrequest.load() == 1) {
//       // get new model + task
//       std::string filename =
//           mjpc::GetModelPath(mjpc::kTasks[sim.agent.task().id].xml_path);

//       // copy model + task file path into simulate
//       mju::strcpy_arr(sim.filename, filename.c_str());
//       mjModel* mnew = LoadModel(sim.filename, sim);
//       mjData* dnew = nullptr;
//       if (mnew) dnew = mj_makeData(mnew);
//       if (dnew) {
//         sim.load(sim.filename, mnew, dnew, true);
//         m = mnew;
//         d = dnew;
//         mj_forward(m, d);

//         // allocate ctrlnoise
//         free(ctrlnoise);
//         ctrlnoise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * m->nu));
//         mju_zero(ctrlnoise, m->nu);
//       }

//       // agent
//       {
//         std::ostringstream concatenated_task_names;
//         for (const auto& task : mjpc::kTasks) {
//           concatenated_task_names << task.name << '\n';
//         }
//         const auto& task = mjpc::kTasks[sim.agent.task().id];
//         sim.agent.Initialize(m, d, concatenated_task_names.str(),
//                              mjpc::kPlannerNames, task.residual,
//                              task.transition);
//       }
//       sim.agent.Allocate();
//       sim.agent.Reset();
//       sim.agent.PlotInitialize();
//     }

//     // reload model to refresh UI
//     if (sim.uiloadrequest.load() == 1) {
//       // get new model + task
//       std::string filename =
//           mjpc::GetModelPath(mjpc::kTasks[sim.agent.task().id].xml_path);

//       // copy model + task file path into simulate
//       mju::strcpy_arr(sim.filename, filename.c_str());
//       mjModel* mnew = LoadModel(sim.filename, sim);
//       mjData* dnew = nullptr;
//       if (mnew) dnew = mj_makeData(mnew);
//       if (dnew) {
//         sim.load(sim.filename, mnew, dnew, true);
//         m = mnew;
//         d = dnew;
//         mj_forward(m, d);

//         // allocate ctrlnoise
//         free(ctrlnoise);
//         ctrlnoise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * m->nu));
//         mju_zero(ctrlnoise, m->nu);
//       }

//       // set initial configuration via keyframe
//       double* qpos_key = mjpc::KeyFrameByName(sim.mnew, sim.dnew, "home");
//       if (qpos_key) {
//         mju_copy(sim.dnew->qpos, qpos_key, sim.mnew->nq);
//       }

//       // decrement counter
//       sim.uiloadrequest.fetch_sub(1);
//     }

//     // reload GUI
//     if (sim.uiloadrequest.load() == -1) {
//       sim.load(sim.filename, sim.m, sim.d, false);
//       sim.uiloadrequest.fetch_add(1);
//     }
//     // ----------------------- //

//     // sleep for 1 ms or yield, to let main thread run
//     //  yield results in busy wait - which has better timing but kills battery
//     //  life
//     if (sim.run && sim.busywait) {
//       std::this_thread::yield();
//     } else {
//       std::this_thread::sleep_for(std::chrono::milliseconds(1));
//     }

//     {
//       // lock the sim mutex
//       const std::lock_guard<std::mutex> lock(sim.mtx);

//       // run only if model is present
//       if (m) {
//         // running
//         if (sim.run) {
//           // record cpu time at start of iteration
//           double startCPU = Glfw().glfwGetTime();

//           // elapsed CPU and simulation time since last sync
//           double elapsedCPU = startCPU - syncCPU;
//           double elapsedSim = d->time - syncSim;

//           // inject noise
//           if (sim.ctrlnoisestd) {
//             // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
//             mjtNum rate = mju_exp(-m->opt.timestep / sim.ctrlnoiserate);
//             mjtNum scale = sim.ctrlnoisestd * mju_sqrt(1 - rate * rate);

//             for (int i = 0; i < m->nu; i++) {
//               // update noise
//               ctrlnoise[i] =
//                   rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

//               // apply noise
//               // d->ctrl[i] += ctrlnoise[i]; // noise is now added in controller
//               // callback
//             }
//           }

//           // requested slow-down factor
//           double slowdown = 100 / sim.percentRealTime[sim.realTimeIndex];

//           // misalignment condition: distance from target sim time is bigger
//           // than syncmisalign
//           bool misaligned =
//               mju_abs(elapsedCPU / slowdown - elapsedSim) > syncMisalign;

//           // out-of-sync (for any reason): reset sync times, step
//           if (elapsedSim < 0 || elapsedCPU < 0 || syncCPU == 0 || misaligned ||
//               sim.speedChanged) {
//             // re-sync
//             syncCPU = startCPU;
//             syncSim = d->time;
//             sim.speedChanged = false;

//             // clear old perturbations, apply new
//             mju_zero(d->xfrc_applied, 6 * m->nbody);
//             sim.applyposepertubations(0);  // move mocap bodies only
//             sim.applyforceperturbations();

//             // run single step, let next iteration deal with timing
//             mj_step(m, d);
//           } else {  // in-sync: step until ahead of cpu
//             bool measured = false;
//             mjtNum prevSim = d->time;
//             double refreshTime = simRefreshFraction / sim.refreshRate;

//             // step while sim lags behind cpu and within refreshTime
//             while ((d->time - syncSim) * slowdown <
//                        (Glfw().glfwGetTime() - syncCPU) &&
//                    (Glfw().glfwGetTime() - startCPU) < refreshTime) {
//               // measure slowdown before first step
//               if (!measured && elapsedSim) {
//                 sim.measuredSlowdown = elapsedCPU / elapsedSim;
//                 measured = true;
//               }

//               // clear old perturbations, apply new
//               mju_zero(d->xfrc_applied, 6 * m->nbody);
//               sim.applyposepertubations(0);  // move mocap bodies only
//               sim.applyforceperturbations();

//               // call mj_step
//               mj_step(m, d);

//               // break if reset
//               if (d->time < prevSim) {
//                 break;
//               }
//             }
//           }
//         } else {  // paused
//           // apply pose perturbation
//           sim.applyposepertubations(1);  // move mocap and dynamic bodies

//           // run mj_forward, to update rendering and joint sliders
//           mj_forward(m, d);
//         }

//         // transition
//         if (sim.agent.task().transition_status == 1 &&
//             sim.uiloadrequest.load() == 0) {
//           sim.agent.task().Transition(m, d);
//         }
//       }
//     }  // release sim.mtx

//     // state
//     if (sim.uiloadrequest.load() == 0) {
//       sim.agent.ActiveState().Set(m, d);
//     }
//   }
// }
// }  // namespace

// // ---------------------------- physics_thread ---------------------------------

// void PhysicsThread(mj::Simulate* sim, const char* filename) {
//   // request loadmodel if file given (otherwise drag-and-drop)
//   if (filename != nullptr) {
//     m = LoadModel(filename, *sim);
//     if (m) d = mj_makeData(m);
//     if (d) {
//       sim->load(filename, m, d, true);
//       mj_forward(m, d);

//       // allocate ctrlnoise
//       free(ctrlnoise);
//       ctrlnoise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * m->nu));
//       mju_zero(ctrlnoise, m->nu);
//     }
//   }

//   PhysicsLoop(*sim);

//   // delete everything we allocated
//   free(ctrlnoise);
//   mj_deleteData(d);
//   mj_deleteModel(m);
// }

// // ------------------------------- main ----------------------------------------

// // machinery for replacing command line error by a macOS dialog box
// // when running under Rosetta
// #if defined(__APPLE__) && defined(__AVX__)
// extern void DisplayErrorDialogBox(const char* title, const char* msg);
// static const char* rosetta_error_msg = nullptr;
// __attribute__((used, visibility("default")))
// extern "C" void _mj_rosettaError(const char* msg) {
//   rosetta_error_msg = msg;
// }
// #endif

// // run event loop
// int main(int argc, char** argv) {
//   // display an error if running on macOS under Rosetta 2
// #if defined(__APPLE__) && defined(__AVX__)
//   if (rosetta_error_msg) {
//     DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
//     std::exit(1);
//   }
// #endif

//   absl::ParseCommandLine(argc, argv);
//   std::printf("MuJoCo version %s\n", mj_versionString());
//   if (mjVERSION_HEADER != mj_version()) {
//     mju_error("Headers and library have Different versions");
//   }

//   // threads
//   printf("Hardware threads: %i\n", mjpc::NumAvailableHardwareThreads());

//   // init GLFW
//   if (!Glfw().glfwInit()) {
//     mju_error("could not initialize GLFW");
//   }

//   std::string task = absl::GetFlag(FLAGS_task);
//   if (!task.empty()) {
//     sim->agent.task().id = TaskIdByName(task);
//     if (sim->agent.task().id == -1) {
//       std::cerr << "Invalid --task flag: '" << task << "'. Valid values:\n";
//       for (const auto& task : mjpc::kTasks) {
//         std::cerr << '\t' << task.name << '\n';
//       }
//       mju_error("Invalid --task flag.");
//       return -1;
//     }
//   }

//   // default model + task
//   std::string filename =
//       mjpc::GetModelPath(mjpc::kTasks[sim->agent.task().id].xml_path);

//   // copy model + task file path into simulate
//   mju::strcpy_arr(sim->filename, filename.c_str());

//   // load model + make data
//   m = LoadModel(sim->filename, *sim);
//   if (m) d = mj_makeData(m);
//   sim->mnew = m;
//   sim->dnew = d;

//   sim->delete_old_m_d = true;
//   sim->loadrequest = 2;

//   // control noise
//   free(ctrlnoise);
//   ctrlnoise = (mjtNum*)malloc(sizeof(mjtNum) * m->nu);
//   mju_zero(ctrlnoise, m->nu);

//   // agent
//   {
//     std::ostringstream concatenated_task_names;
//     for (const auto& task : mjpc::kTasks) {
//       concatenated_task_names << task.name << '\n';
//     }
//     const auto& task = mjpc::kTasks[sim->agent.task().id];
//     sim->agent.Initialize(m, d, concatenated_task_names.str(),
//                           mjpc::kPlannerNames, task.residual,
//                           task.transition);
//   }
//   sim->agent.Allocate();
//   sim->agent.Reset();
//   sim->agent.PlotInitialize();

//   // planning threads
//   printf("Agent threads: %i\n", sim->agent.max_threads());

//   // set initial configuration via keyframe
//   double* qpos_key = mjpc::KeyFrameByName(sim->mnew, sim->dnew, "home");
//   if (qpos_key) {
//     mju_copy(sim->dnew->qpos, qpos_key, sim->mnew->nq);
//   }

//   // set control callback
//   mjcb_control = controller;

//   // set sensor callback
//   mjcb_sensor = sensor;

//   // start physics thread
//   mjpc::ThreadPool physics_pool(1);
//   physics_pool.Schedule([]() { PhysicsLoop(*sim.get()); });

//   // start plan thread
//   mjpc::ThreadPool plan_pool(1);
//   plan_pool.Schedule(
//       []() { sim->agent.Plan(sim->exitrequest, sim->uiloadrequest); });

//   // start simulation UI loop (blocking call)
//   sim->renderloop();
//   // terminate GLFW (crashes with Linux NVidia drivers)
// #if defined(__APPLE__) || defined(_WIN32)
//   Glfw().glfwTerminate();
// #endif
// 
//   return 0;
// }


// test fitness function 
double f(const double* x, int n) {
  // return 0.5 * mju_dot(x, x, n);
  return (1.0 - x[0]) * (1.0 - x[0]) + 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
}

// sort fitness 
void sort_indices(std::vector<int>& idx, const std::vector<double> &f) {

  // initialize original index locations
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in f
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when f contains elements of equal values 
  std::sort(idx.begin(), idx.end(),
       [&f](int idx1, int idx2) {return f[idx1] < f[idx2];});
}

// Cholesky solve
void mju_cholForward(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int n) {
  // copy if source and destination are different
  if (res!=vec) {
    mju_copy(res, vec, n);
  }

  // forward substitution: solve L*res = vec
  for (int i=0; i<n; i++) {
    if (i) {
      res[i] -= mju_dot(mat+i*n, res, i);
    }

    // diagonal
    res[i] /= mat[i*(n+1)];
  }

  // // backward substitution: solve L'*res = res
  // for (int i=n-1; i>=0; i--) {
  //   if (i<n-1) {
  //     for (int j=i+1; j<n; j++) {
  //       res[i] -= mat[j*n+i] * res[j];
  //     }
  //   }
  //   // diagonal
  //   res[i] /= mat[i*(n+1)];
  // }
}

int main(int argc, char** argv) {
  // CMA-ES
  printf("CMA-ES\n");

  // ----- threadpool ----- //
  mjpc::ThreadPool pool(10);

  // ----- test fitness function ----- // 

  // initial point
  const int dim = 2;
  double x[dim];
  mju_fill(x, 0.0, dim);

  // // fitness
  // double v = f(x, dim);

  // printf("Initial fitness: %f\n", v);

  // info 
  // printf("initial point: \n");
  // mju_printMat(x, 1, dim);
  // printf("value: %f\n", v);

  // number of samples
  int num_sample = 4 + mju_floor(3 * mju_log(dim));
  int num_elite = mju_floor(num_sample / 2);

  // printf("num_sample: %i\n", num_sample);
  // printf("num_elite:   %i\n", num_elite);
  
  // initialize mean
  std::vector<double> mu;
  mu.resize(dim);
  mju_copy(mu.data(), x, dim);

  // initial step size 
  double step_size = 1.0;

  // ----- compute weights ----- //
  double weight[num_sample];
  for (int i = 0; i < num_sample; i++) {
    weight[i] = mju_log(0.5 * (num_sample + 1)) - mju_log(i + 1);
  }

  // elite weight normalization
  double elite_sum = mju_sum(weight, num_elite);
  for (int i = 0; i < num_elite; i++) {
    weight[i] /= elite_sum;
  }

  // ----- parameters ----- //
  double mu_eff = 1.0 / mju_dot(weight, weight, num_elite);
  double c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0);
  double d_sigma = 1.0 + 2.0 * mju_max(0.0, mju_sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma;
  double c_Sigma = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim);
  double c1 = 2.0 / ((dim + 1.3) * (dim + 1.3) + mu_eff);
  double c_mu = mju_min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) * (dim + 2.0) + mu_eff));
  double E = mju_sqrt(dim) * (1.0 - 1.0 / (4 * dim) + 1.0 / (21.0 * dim * dim));
  double eps = 1.0e-3;

  // scale non-elite weights
  double nonelite_sum = mju_sum(weight + num_elite, num_sample - num_elite);
  for (int i = num_elite; i < num_sample; i++) {
    weight[i] *= -1.0 * (1.0 + c1 / c_mu) / nonelite_sum;
  }

  // info 
  // printf("mu_eff: %f\n", mu_eff);
  // printf("c_sigma: %f\n", c_sigma);
  // printf("d_sigma: %f\n", d_sigma);
  // printf("c_Sigma: %f\n", c_Sigma);
  // printf("c1: %f\n", c1);
  // printf("c_mu: %f\n", c_mu);
  // printf("E: %f\n", E);

  // printf("weights:\n");
  // mju_printMat(weight, 1, num_sample);

  // ----- allocate memory ----- //

  // create
  std::vector<double> p_sigma;
  std::vector<double> p_sigma_tmp;
  std::vector<double> p_Sigma;
  std::vector<double> Sigma; 
  std::vector<double> Sigma_tmp;
  std::vector<double> covariance;
  std::vector<double> covariance_lower;
  std::vector<double> fitness;
  std::vector<int> fitness_sort;
  std::vector<double> gaussian_noise;
  std::vector<double> sample;
  std::vector<double> delta_s;
  std::vector<double> delta_w;
  std::vector<double> weight_update;
  std::vector<double> C_delta_s;
  
  // resize
  p_sigma.resize(dim);
  p_sigma_tmp.resize(dim);
  p_Sigma.resize(dim);
  Sigma.resize(dim * dim);
  Sigma_tmp.resize(dim * dim);
  covariance.resize(dim * dim);
  covariance_lower.resize(dim * dim);
  fitness.resize(num_sample);
  fitness_sort.resize(num_sample);
  gaussian_noise.resize(dim * num_sample);
  sample.resize(dim * num_sample);
  delta_s.resize(dim * num_sample);
  delta_w.resize(dim);
  weight_update.resize(num_sample);
  C_delta_s.resize(dim);

  // initialize 
  mju_zero(p_sigma.data(), dim);
  mju_zero(p_Sigma.data(), dim);
  mju_eye(Sigma.data(), dim);

  // ----- sample and evaluate candidates ----- //
  int max_iteration = 1000;

  for (int k = 0; k < max_iteration; k++) {
    // printf("ITERATION: %i\n", k + 1);

    // copy covariance
    mju_copy(covariance.data(), Sigma.data(), dim * dim);

    // regularize covariance
    for (int i = 0; i < dim; i++) {
      covariance[i * dim + i] += eps;
    }

    // Cholesky factor (L L')
    int rank = mju_cholFactor(covariance.data(), dim, 0.0);
    if (rank < dim) {
      printf("Cholesky factorization failure\n");
    }

    // get lower triangular matrix
    mju_zero(covariance_lower.data(), dim * dim);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        if (j <= i) {
          covariance_lower[i * dim + j] = covariance[i * dim + j];
        }
      }
    }


    // printf("Cholesky factor\n");
    // mju_printMat(covariance.data(), dim, dim);

    // printf("Cholesky factor lower\n");
    // mju_printMat(covariance_lower.data(), dim, dim);

    // printf("step size: %f\n", step_size);

    // ----- sample ----- //
    int count_before = pool.GetCount();
    for (int i = 0; i < num_sample; i++) {
      pool.Schedule([&gaussian_noise, &sample, &covariance_lower, step_size, &mu, &fitness, i]() {
        // standard normal sample 
        absl::BitGen gen_;

        for (int j = 0; j < dim; j++) {
          gaussian_noise[i * dim + j] = absl::Gaussian<double>(gen_, 0.0, 1.0);
        }

        // step_size * L * gaussian_noise
        mju_mulMatVec(mjpc::DataAt(sample, i * dim), covariance_lower.data(), mjpc::DataAt(gaussian_noise, i * dim), dim, dim);
        mju_scl(mjpc::DataAt(sample, i * dim), mjpc::DataAt(sample, i * dim), step_size, dim);

        // sample = L * gaussian_noise + mu
        mju_addTo(mjpc::DataAt(sample, i * dim), mu.data(), dim);

        // compute fitness
        fitness[i] = f(mjpc::DataAt(sample, i * dim), dim);
      });
    }
    pool.WaitCount(count_before + num_sample);
    pool.ResetCount();

    // fitness sort 
    sort_indices(fitness_sort, fitness);

    // info

    // printf("sample: \n");
    // mju_printMat(sample.data(), num_sample, dim);

    // printf("fitness: \n");
    // mju_printMat(fitness.data(), 1, num_sample);

    // printf("fitness sort: \n");
    // for (int i = 0; i < num_sample; i++) {
    //   printf("%i ", fitness_sort[i]);
    // }

    // printf("  best fitness: %f\n", fitness[fitness_sort[0]]);

    // ----- selection and mean update ----- //

    // TODO(taylorhowell): move into pool
    // (sample - mu) / step_size
    for (int i = 0; i < num_sample; i++) {
      mju_sub(mjpc::DataAt(delta_s, i * dim), mjpc::DataAt(sample, i * dim), mu.data(), dim);
      mju_scl(mjpc::DataAt(delta_s, i * dim), mjpc::DataAt(delta_s, i * dim), 1.0 / step_size, dim);
    }

    // sum(weight[i] * delta_s[rank[i]])
    mju_zero(delta_w.data(), dim);
    for (int i = 0; i < num_elite; i++) {
      int idx = fitness_sort[i];
      mju_addToScl(delta_w.data(), mjpc::DataAt(delta_s, idx * dim), weight[i], dim);
    }

    // update m += step_size * delta_w 
    mju_addToScl(mu.data(), delta_w.data(), step_size, dim);


    // printf("delta_s: \n");
    // mju_printMat(delta_s.data(), num_sample, dim);

    // printf("delta w:\n");
    // mju_printMat(delta_w.data(), 1, dim);

    // printf("mu update:\n");
    // mju_printMat(mu.data(), 1, dim);

    // ----- step-size control ----- //

    mju_cholForward(p_sigma_tmp.data(), covariance.data(), delta_w.data(), dim);

    // printf("Sigma^-0.5 delta_w: \n");
    // mju_printMat(p_sigma_tmp.data(), 1, dim);

    mju_scl(p_sigma_tmp.data(), p_sigma_tmp.data(), mju_sqrt(c_sigma * (2.0 - c_sigma) * mu_eff), dim);
    
    // printf("scale * Sigma^-0.5 delta_w: \n");
    // mju_printMat(p_sigma_tmp.data(), 1, dim);
    
    mju_scl(p_sigma.data(), p_sigma.data(), 1.0 - c_sigma, dim);
    mju_addTo(p_sigma.data(), p_sigma_tmp.data(), dim);

    // printf("p_sigma: \n");
    // mju_printMat(p_sigma.data(), 1, dim);

    step_size *= mju_exp(c_sigma / d_sigma * (mju_norm(p_sigma.data(), dim) / E - 1.0));

    // printf("new step size: %f\n", step_size);

    // ----- covariance adaptation ----- //

    int h_sigma = (mju_norm(p_sigma.data(), dim) / mju_sqrt(1.0 - mju_pow(1.0 - c_sigma, 2 * (k + 1))) < (1.4 + 2.0 / (dim + 1)) * E) ? 1 : 0;

    // printf("h_sigma: %i\n", h_sigma);

    mju_scl(p_Sigma.data(), p_Sigma.data(), 1.0 - c_Sigma, dim);
    mju_addToScl(p_Sigma.data(), delta_w.data(), h_sigma * mju_sqrt(c_Sigma * (2.0 - c_Sigma) * mu_eff), dim);  

    // printf("p_Sigma: \n");
    // mju_printMat(p_Sigma.data(), 1, dim);

    for (int i = 0; i < num_sample; i++) {
      if (weight[i] >= 0.0) {
        weight_update[i] = weight[i];
      } else {
        int idx = fitness_sort[i];
        mju_cholForward(C_delta_s.data(), covariance.data(), mjpc::DataAt(delta_s, idx * dim), dim);
        weight_update[i] = dim * weight[i] / mju_dot(C_delta_s.data(), C_delta_s.data(), dim);
      }
    }
    // printf("weight update: \n");
    // mju_printMat(weight_update.data(), 1, num_sample);

    mju_scl(Sigma.data(), Sigma.data(), (1.0 - c1 - c_mu) + (c1 * (1.0 - h_sigma) * c_Sigma * (2.0 - c_Sigma)), dim * dim);
    
    mju_mulMatMatT(Sigma_tmp.data(), p_Sigma.data(), p_Sigma.data(), dim, 1, dim);
    mju_scl(Sigma_tmp.data(), Sigma_tmp.data(), c1, dim * dim);

    mju_addTo(Sigma.data(), Sigma_tmp.data(), dim * dim);

    for (int i = 0; i < num_sample; i++) {
      int idx = fitness_sort[i];
      mju_mulMatMatT(Sigma_tmp.data(), mjpc::DataAt(delta_s, idx * dim), mjpc::DataAt(delta_s, idx * dim), dim, 1, dim);
      mju_scl(Sigma_tmp.data(), Sigma_tmp.data(), c_mu * weight_update[i], dim * dim);
      mju_addTo(Sigma.data(), Sigma_tmp.data(), dim * dim);
    }

    mju_symmetrize(Sigma.data(), Sigma.data(), dim);

    // printf("new Sigma:\n");
    // mju_printMat(Sigma.data(), dim, dim);

    // printf("  best sample %i - best candidate: \n", fitness_sort[0]);
    // mju_printMat(mjpc::DataAt(sample, fitness_sort[0] * dim), 1, dim);

    // printf("\n\n\n");
  }

  printf("best value: %f\n", fitness[fitness_sort[0]]);
  printf("best candidate: \n");
  mju_printMat(mjpc::DataAt(sample, fitness_sort[0] * dim), 1, dim);

  return 0;
}


