from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    ActionsCfg,
)

from isaaclab_assets.robots.bipedo_mike import BIPED_CFG   
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen


import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation


ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols = 20,
    horizontal_scale = 0.1,
    vertical_scale = 0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.005, 0.01), noise_step=0.005, border_width=0.25
        )
    } 
)


# ------------------------
# Rewards personalizadas
# ------------------------

@configclass
class BipedoRewards:#(RewardsCfg):

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    #lin_vel_z_l2 = None
    
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    
    feet_air_time_positive_biped = RewTerm(
            func=mdp.feet_air_time_positive_biped,
            weight=2.5,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["link_6", "link_12"]
                ),
                "command_name": "base_velocity",
                "threshold": 0.6,
            },
        )
    
    stand_still_joint_deviation = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-0.1,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        
        )
    
      
    feet_air_time = None


# ------------------------
# Accion de velocidad para el bipedo
# ------------------------
@configclass
class BipedoActionsCfg(ActionsCfg):
    joint_vel = mdp.JointPositionActionCfg(asset_name="robot", 
                                           joint_names=[".*"], 
                                           scale=0.5, use_default_offset=True
                                           )


# ------------------------
# Env plana para bípedo
# ------------------------

@configclass
class BipedoFlatEnvCfg(LocomotionVelocityRoughEnvCfg):

    rewards: BipedoRewards = BipedoRewards()

    def __post_init__(self):
        super().__post_init__()

        # ------------------------
        # Robot
        # ------------------------
        self.scene.robot = BIPED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # ------------------------
        # Terreno plano
        # ------------------------
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.max_init_terrain_level = None

        # desactivar curriculum
        self.curriculum.terrain_levels = None

        nuevo_sensor_cfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(link_6|link_12)",
            update_period=0.0,
            track_contact_points=True,
            filter_prim_paths_expr=["/World/ground"],
            history_length=3 # Opcional
        )

        self.scene.zmp_contact_sensor = nuevo_sensor_cfg
        
        # ------------------------
        # Randomización OFF
        # ------------------------
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        
        
        # Desactivar ray caster
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # reset sin velocidad inicial
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        # ------------------------
        # Comandos simples
        # ------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)




        # activar flechas
        self.commands.base_velocity.debug_vis = True

        # ------------------------
        # Terminación correcta
        # ------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = [".*base_link"]


        # ------------------------
        # Simplificar escena
        # ------------------------
        self.scene.num_envs = 1024
        self.scene.env_spacing = 2.5

