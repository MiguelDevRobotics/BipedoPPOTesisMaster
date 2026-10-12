'''Copyright 2026 Miguel de Jesus Santiago Vasquez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.'''

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Script para hacer inferencia del robot bípedo")
parser.add_argument("--num_envs", type=int, default=1, help="Numero de robots a spawnear")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

############################################################################
############################################################################
############################################################################

from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab_assets.robots.bipedo_mike import BIPED_CFG
import torch
import omni.appwindow 
import carb.input
from carb.input import KeyboardEventType
from isaaclab.utils.math import matrix_from_quat
import os
import io
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

keyboard_cfg = Se2KeyboardCfg(
    v_x_sensitivity=0.6,
    v_y_sensitivity=0.6,
    omega_z_sensitivity=1.0,
    )   

keyboard = Se2Keyboard(keyboard_cfg)


class BipedoSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(

        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
        )))
    
    
    light = AssetBaseCfg(
        prim_path = "/World/light",
        spawn = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    bipedo = BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Bipedo")


def get_observation(scene: InteractiveScene, last_action: torch.Tensor):

    lin_vel = scene["bipedo"].data.root_lin_vel_b
    ang_vel = scene["bipedo"].data.root_ang_vel_b
    joint_pos = scene["bipedo"].data.joint_pos - scene["bipedo"].data.default_joint_pos.clone()
    joint_vel = scene["bipedo"].data.joint_vel
    '''
        quat = root_state[:, 3:7]
        rot = matrix_from_quat(quat).transpose(-1, -2)
        v = torch.tensor([0.0, 0.0, -1.0], device="cuda:0")
        grav = rot @ v
        Esta es la forma de calcular el vector de poryeccion de gravedad 
    '''

    proyected_gravity = scene["bipedo"].data.projected_gravity_b

    cmd_raw = keyboard.advance().to(lin_vel.device).float()
    cmd = torch.zeros_like(proyected_gravity, device=lin_vel.device)
    cmd[:, :] = cmd_raw

    obs = torch.cat([lin_vel, ang_vel, proyected_gravity, cmd, joint_pos, joint_vel, last_action], dim=1)

    print(obs)
    return obs


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    # load the trained jit policy
    policy_path = os.path.abspath("/path/to/policy.pt")
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    
    materials = scene["bipedo"].root_physx_view.get_material_properties()
    env_ids = torch.arange(scene.num_envs, device=materials.device)
    materials[:] = materials  # opcional (solo para copiar estructura)

    # cambiar fricción global
    materials[..., 0] = 0.6  # static friction
    materials[..., 1] = 0.8  # dynamic friction
    materials[..., 2] = 0.0  # restitution

    scene["bipedo"].root_physx_view.set_material_properties(materials, env_ids)

    root_state = scene["bipedo"].data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins

    scene["bipedo"].write_root_pose_to_sim(root_state[:, :7])
    scene["bipedo"].write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos, joint_vel = (
        scene["bipedo"].data.default_joint_pos.clone(),
        scene["bipedo"].data.default_joint_vel.clone()
    )

    joint_pos = torch.tensor([[1.138, -1.138, 0.0, 0.0, -2.225, 2.225, 0.0, 0.0, 1.102, -1.102, 0.0, 0.0]], device='cuda:0')

    scene["bipedo"].write_joint_state_to_sim(joint_pos, joint_vel)
    scene.reset()

    default_action = torch.Tensor(joint_pos)
    scene["bipedo"].set_joint_position_target(default_action)

    target = scene["bipedo"].data.default_joint_pos.clone()
    warmup = True
    action = torch.zeros_like(default_action)
 
    while simulation_app.is_running():
        with torch.inference_mode():
            if warmup:
                if count > 200:
                    q = scene["bipedo"].data.joint_pos
                    alpha = torch.tensor(
                        [0.07, 0.07, 0.07, 0.07, 0.07, 0.07,
                        0.07, 0.07, 0.2, 0.2, 0.07, 0.07],
                        device=q.device
                    )
                    target_interp = q + alpha * (target - q)

                    scene["bipedo"].set_joint_position_target(target_interp)
                    error = torch.abs(scene["bipedo"].data.joint_pos - target)
                    if torch.all( error < 0.11, dim=1):
                        warmup = False
                scene.write_data_to_sim()
                sim.step()
                sim_time += 1
                count += 1
                scene.update(sim_dt)
                
            else:
                if count % 4 == 0:
                    obs = get_observation(scene,action)
                    if torch.any(obs[:,9:12] != 0):
                        action = policy(obs)
                    target = scene["bipedo"].data.default_joint_pos + action * 0.5
                    scene["bipedo"].set_joint_position_target(target)
                scene.write_data_to_sim()
                sim.step()
                sim_time += 1
                count += 1
                scene.update(sim_dt)

def main():
    sim_cfg = SimulationCfg(dt = 0.005)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([0.0, 2.0, 2],[0.0, 0.0, 0.0])
    scene_cfg = BipedoSceneCfg(args_cli.num_envs, env_spacing=0.2)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    run_simulator(sim, scene)



if __name__ == "__main__":
    main()
    simulation_app.close()
