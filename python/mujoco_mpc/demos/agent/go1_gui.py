# Copyright 2022 DeepMind Technologies Limited
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

import pathlib
from mujoco_mpc import agent as agent_lib

# Run GUI
with agent_lib.Agent(
    server_binary_path=pathlib.Path(agent_lib.__file__).parent
    / "mjpc"
    / "ui_agent_server",
    task_id="Go1",
) as agent:
  while True:
    ## set state (from estimator)
    # agent.set_state(qpos=, qvel=, time=)

    ## ## get action (from planner)
    # ctrl = agent.get_action()
    None
