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

// An implementation of the `Estimator` gRPC service.

#ifndef GRPC_ESTIMATOR_SERVICE_H
#define GRPC_ESTIMATOR_SERVICE_H

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include <memory>
#include <vector>

#include "grpc/estimator.grpc.pb.h"
#include "grpc/estimator.pb.h"
#include "mjpc/estimators/estimator.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace estimator_grpc {

class EstimatorService final : public estimator::Estimator::Service {
 public:
  explicit EstimatorService()
      : thread_pool_(mjpc::NumAvailableHardwareThreads()) {}
  ~EstimatorService();

  grpc::Status Init(grpc::ServerContext* context,
                    const estimator::InitRequest* request,
                    estimator::InitResponse* response) override;

  grpc::Status SetData(grpc::ServerContext* context,
                       const estimator::SetDataRequest* request,
                       estimator::SetDataResponse* response) override;

  grpc::Status GetData(grpc::ServerContext* context,
                       const estimator::GetDataRequest* request,
                       estimator::GetDataResponse* response) override;

  grpc::Status SetSettings(grpc::ServerContext* context,
                           const estimator::SetSettingsRequest* request,
                           estimator::SetSettingsResponse* response) override;

  grpc::Status GetSettings(grpc::ServerContext* context,
                           const estimator::GetSettingsRequest* request,
                           estimator::GetSettingsResponse* response) override;

  grpc::Status GetCosts(grpc::ServerContext* context,
                        const estimator::GetCostsRequest* request,
                        estimator::GetCostsResponse* response) override;

  grpc::Status SetWeights(grpc::ServerContext* context,
                          const estimator::SetWeightsRequest* request,
                          estimator::SetWeightsResponse* response) override;

  grpc::Status GetWeights(grpc::ServerContext* context,
                          const estimator::GetWeightsRequest* request,
                          estimator::GetWeightsResponse* response) override;

  grpc::Status ShiftTrajectories(
      grpc::ServerContext* context,
      const estimator::ShiftTrajectoriesRequest* request,
      estimator::ShiftTrajectoriesResponse* response) override;

  grpc::Status Reset(grpc::ServerContext* context,
                     const estimator::ResetRequest* request,
                     estimator::ResetResponse* response) override;

  grpc::Status Optimize(grpc::ServerContext* context,
                        const estimator::OptimizeRequest* request,
                        estimator::OptimizeResponse* response) override;

  grpc::Status GetStatus(grpc::ServerContext* context,
                         const estimator::GetStatusRequest* request,
                         estimator::GetStatusResponse* response) override;

 private:
  bool Initialized() const { return estimator_.model_ && estimator_.configuration_length_ >= 3; }  // TODO(taylor):

  // estimator
  mjpc::Estimator estimator_;
  mjpc::UniqueMjModel estimator_model_override_ = {nullptr, mj_deleteModel};

  // threadpool
  mjpc::ThreadPool thread_pool_;
};

}  // namespace estimator_grpc

#endif  // GRPC_ESTIMATOR_SERVICE_H