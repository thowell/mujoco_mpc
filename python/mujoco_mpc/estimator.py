# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python interface for interface with Estimator."""

import atexit
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
from typing import Literal, Optional

import grpc
import mujoco
import numpy as np
from numpy import typing as npt

# INTERNAL IMPORT
from mujoco_mpc.proto import estimator_pb2
from mujoco_mpc.proto import estimator_pb2_grpc


def find_free_port() -> int:
  """Find an available TCP port on the system.

    This function creates a temporary socket, binds it to an available port
    chosen by the operating system, and returns the chosen port number.

  Returns:
      int: An available TCP port number.
  """
  with socket.socket(family=socket.AF_INET6) as s:
    s.bind(("", 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return s.getsockname()[1]


class Estimator:
  """`Estimator` class to interface with MuJoCo MPC estimator.

  Attributes:
    port:
    channel:
    stub:
    server_process:
  """

  def __init__(
      self,
      model: mujoco.MjModel,
      configuration_length: int,
      buffer_length: Optional[int] = None,
      server_binary_path: Optional[str] = None,
      colab_logging: bool = False,
  ):
    # server
    if server_binary_path is None:
      binary_name = "estimator_server"
      server_binary_path = pathlib.Path(__file__).parent / "mjpc" / binary_name
    self._colab_logging = colab_logging
    self.port = find_free_port()
    self.server_process = subprocess.Popen(
        [str(server_binary_path), f"--mjpc_port={self.port}"],
        stdout=subprocess.PIPE if colab_logging else None,
    )
    os.set_blocking(self.server_process.stdout.fileno(), False)
    atexit.register(self.server_process.kill)

    credentials = grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP)
    self.channel = grpc.secure_channel(f"localhost:{self.port}", credentials)
    grpc.channel_ready_future(self.channel).result(timeout=10)
    self.stub = estimator_pb2_grpc.EstimatorStub(self.channel)

    # initialize
    self.init(
        model,
        configuration_length,
        buffer_length=buffer_length,
        send_as="xml",
    )

  def close(self):
    self.channel.close()
    self.server_process.kill()
    self.server_process.wait()

  def init(
      self,
      model: mujoco.MjModel,
      configuration_length: int,
      buffer_length: Optional[int] = None,
      send_as: Literal["xml"] = "xml",
  ):
    """Initialize the estimator for estimation horizon `configuration_length`.

    Args:
      model: optional `MjModel` instance, which, if provided, will be used as
        the underlying model for planning. If not provided, the default MJPC
        task xml will be used.
      configuration_length: estimation horizon.
      send_as: The serialization format for sending the model over gRPC; "xml".
    """

    # setup model
    def model_to_xml(model: mujoco.MjModel) -> str:
      tmp = tempfile.NamedTemporaryFile()
      mujoco.mj_saveLastXML(tmp.name, model)
      with pathlib.Path(tmp.name).open("rt") as f:
        xml_string = f.read()
      return xml_string

    if model is not None:
      model_message = estimator_pb2.MjModel(xml=model_to_xml(model))
    else:
      print("Failed to find xml.")
      model_message = None

    # initialize request
    init_request = estimator_pb2.InitRequest(
        model=model_message,
        configuration_length=configuration_length,
        buffer_length=buffer_length,
    )

    # initialize response
    self._wait(self.stub.Init.future(init_request))

  def data(
      self,
      index: int,
      configuration: Optional[npt.ArrayLike] = [],
      velocity: Optional[npt.ArrayLike] = [],
      acceleration: Optional[npt.ArrayLike] = [],
      time: Optional[npt.ArrayLike] = [],
      configuration_prior: Optional[npt.ArrayLike] = [],
      sensor_measurement: Optional[npt.ArrayLike] = [],
      sensor_prediction: Optional[npt.ArrayLike] = [],
      force_measurement: Optional[npt.ArrayLike] = [],
      force_prediction: Optional[npt.ArrayLike] = [],
  ) -> dict[str, np.ndarray]:
    # assemble inputs
    inputs = estimator_pb2.Data(
        configuration=configuration,
        velocity=velocity,
        acceleration=acceleration,
        time=time,
        configuration_prior=configuration_prior,
        sensor_measurement=sensor_measurement,
        sensor_prediction=sensor_prediction,
        force_measurement=force_measurement,
        force_prediction=force_prediction,
    )

    # data request
    request = estimator_pb2.DataRequest(data=inputs, index=index)

    # data response
    data = self._wait(self.stub.Data.future(request)).data

    # return all data
    return {
        "configuration": np.array(data.configuration),
        "velocity": np.array(data.velocity),
        "acceleration": np.array(data.acceleration),
        "time": np.array(data.time),
        "configuration_prior": np.array(data.configuration_prior),
        "sensor_measurement": np.array(data.sensor_measurement),
        "sensor_prediction": np.array(data.sensor_prediction),
        "force_measurement": np.array(data.force_measurement),
        "force_prediction": np.array(data.force_prediction),
    }

  def settings(
      self,
      configuration_length: Optional[int] = None,
      search_type: Optional[int] = None,
      prior_flag: Optional[bool] = None,
      sensor_flag: Optional[bool] = None,
      force_flag: Optional[bool] = None,
      smoother_iterations: Optional[int] = None,
      skip_prior_weight_update: Optional[bool] = None,
      time_scaling: Optional[bool] = None,
  ) -> dict[str, int | bool]:
    # assemble settings
    inputs = estimator_pb2.Settings(
        configuration_length=configuration_length,
        search_type=search_type,
        prior_flag=prior_flag,
        sensor_flag=sensor_flag,
        force_flag=force_flag,
        smoother_iterations=smoother_iterations,
        skip_prior_weight_update=skip_prior_weight_update,
        time_scaling=time_scaling,
    )

    # settings request
    request = estimator_pb2.SettingsRequest(
        settings=inputs,
    )

    # settings response
    settings = self._wait(self.stub.Settings.future(request)).settings

    # return all settings
    return {
        "configuration_length": settings.configuration_length,
        "search_type": settings.search_type,
        "prior_flag": settings.prior_flag,
        "sensor_flag": settings.sensor_flag,
        "force_flag": settings.force_flag,
        "smoother_iterations": settings.smoother_iterations,
        "skip_prior_weight_update": settings.skip_prior_weight_update,
        "time_scaling": settings.time_scaling,
    }

  def weight(
      self,
      prior: Optional[float] = None,
      sensor: Optional[npt.ArrayLike] = [],
      force: Optional[npt.ArrayLike] = [],
  ) -> dict[str, float | np.ndarray]:
    # assemble input weights
    inputs = estimator_pb2.Weight(
        prior=prior,
        sensor=sensor,
        force=force,
    )

    # weight request
    request = estimator_pb2.WeightsRequest(weight=inputs)

    # weight response
    weight = self._wait(self.stub.Weights.future(request)).weight

    # return all weights
    return {
        "prior": weight.prior,
        "sensor": np.array(weight.sensor),
        "force": np.array(weight.force),
    }

  def norm(
      self,
      sensor_type: Optional[npt.ArrayLike] = [],
      sensor_parameters: Optional[npt.ArrayLike] = [],
      force_type: Optional[npt.ArrayLike] = [],
      force_parameters: Optional[npt.ArrayLike] = [],
  ) -> dict[str, np.ndarray]:
    # assemble input norm data
    inputs = estimator_pb2.Norm(
        sensor_type=sensor_type,
        sensor_parameters=sensor_parameters,
        force_type=force_type,
        force_parameters=force_parameters,
    )

    # norm request
    request = estimator_pb2.NormRequest(norm=inputs)

    # norm response
    norm = self._wait(self.stub.Norms.future(request)).norm

    # return all norm data
    return {
        "sensor_type": norm.sensor_type,
        "sensor_parameters": np.array(norm.sensor_parameters),
        "force_type": np.array(norm.force_type),
        "force_parameters": np.array(norm.force_parameters),
    }

  def cost(self) -> dict[str, float]:
    # cost request
    request = estimator_pb2.CostRequest()

    # cost response
    cost = self._wait(self.stub.Cost.future(request)).cost

    # return all costs
    return {
        "total": cost.total,
        "prior": cost.prior,
        "sensor": cost.sensor,
        "force": cost.force,
        "initial": cost.initial,
    }

  def status(self) -> dict[str, int]:
    # status request
    request = estimator_pb2.StatusRequest()

    # status response
    status = self._wait(self.stub.Status.future(request)).status

    # return all status
    return {
        "search_iterations": status.search_iterations,
        "smoother_iterations": status.smoother_iterations,
        "step_size": status.step_size,
        "regularization": status.regularization,
        "gradient_norm": status.gradient_norm,
    }

  def shift(self, shift: int) -> int:
    # shift request
    request = estimator_pb2.ShiftRequest(shift=shift)

    # return head (for testing)
    return self._wait(self.stub.Shift.future(request)).head

  def reset(self):
    # reset request
    request = estimator_pb2.ResetRequest()

    # reset response
    self._wait(self.stub.Reset.future(request))

  def optimize(self):
    # optimize request
    print("hi!")
    request = estimator_pb2.OptimizeRequest()

    # optimize response
    self._wait(self.stub.Optimize.future(request))

  def cost_hessian(self) -> np.ndarray:
    # Hessian request
    request = estimator_pb2.CostHessianRequest()

    # Hessian response
    response = self._wait(self.stub.CostHessian.future(request))

    # reshape Hessian to (dimension, dimension)
    hessian = np.array(response.hessian).reshape(response.dimension, response.dimension)

    # return Hessian matrix
    return hessian

  def prior_matrix(self, prior: Optional[npt.ArrayLike] = None) -> np.ndarray:
    # prior request
    request = estimator_pb2.PriorMatrixRequest(
        prior=prior.flatten() if prior is not None else None
    )

    # prior response
    response = self._wait(self.stub.PriorMatrix.future(request))

    # reshape prior to (dimension, dimension)
    mat = np.array(response.prior).reshape(response.dimension, response.dimension)

    # return prior matrix
    return mat

  def buffer(
      self,
      index: int,
      sensor: Optional[npt.ArrayLike] = [],
      mask: Optional[npt.ArrayLike] = [],
      ctrl: Optional[npt.ArrayLike] = [],
      time: Optional[npt.ArrayLike] = [],
  ) -> dict[str, int | np.ndarray]:
    # assemble buffer
    inputs = estimator_pb2.Buffer(
        sensor=sensor,
        mask=mask,
        ctrl=ctrl,
        time=time,
    )

    # data request
    request = estimator_pb2.BufferDataRequest(index=index, buffer=inputs)

    # data response
    response = self._wait(self.stub.BufferData.future(request))

    # buffer
    buffer = response.buffer

    # return all buffer data at time index
    return {
        "sensor": np.array(buffer.sensor),
        "mask": np.array(buffer.mask),
        "ctrl": np.array(buffer.ctrl),
        "time": np.array(buffer.time),
        "length": response.length,
    }

  def update_buffer(
      self,
      sensor: npt.ArrayLike,
      mask: npt.ArrayLike,
      ctrl: npt.ArrayLike,
      time: npt.ArrayLike,
  ) -> int:
    # assemble buffer
    inputs = estimator_pb2.Buffer(
        sensor=sensor,
        mask=mask,
        ctrl=ctrl,
        time=time,
    )

    # update request
    request = estimator_pb2.UpdateBufferRequest(buffer=inputs)

    # return current buffer length
    return self._wait(self.stub.UpdateBuffer.future(request)).length

  def reset_buffer(self):
    # reset buffer request
    request = estimator_pb2.ResetBufferRequest()

    # reset buffer response
    self._wait(self.stub.ResetBuffer.future(request))

  def print_cost(self):
    # get costs
    cost = self.cost()

    # print
    print("cost:")
    print("  [total] = ", cost["total"])
    print("   - prior = ", cost["prior"])
    print("   - sensor = ", cost["sensor"])
    print("   - force = ", cost["force"])
    print("  (initial = ", cost["initial"], ")")

  def print_status(self):
    # get status
    status = self.status()

    # print
    print("status:")
    print("- search iterations = ", status["search_iterations"])
    print("- smoother iterations = ", status["smoother_iterations"])
    print("- step size = ", status["step_size"])
    print("- regularization = ", status["regularization"])
    print("- gradient norm = ", status["gradient_norm"])

  def _wait(self, future):
    """Waits for the future to complete, while printing out subprocess stdout."""
    if self._colab_logging:
      while True:
        line = self.server_process.stdout.readline()
        if line:
          sys.stdout.write(line.decode('utf-8'))
        if future.done():
          break
    return future.result()
