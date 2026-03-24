from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from isaaclab_assets.robots.bipedo_mike import BIPED_CFG   # tu articulation cfg


# ------------------------
# Rewards personalizadas
# ------------------------

@configclass
class BipedoRewards(RewardsCfg):

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    
    feet_air_time_positive_biped = RewTerm(
            func=mdp.feet_air_time,
            weight=0.75,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["link_6", "link_12"]
                ),
                "command_name": "base_velocity",
                "threshold": 0.4,
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
    # menos penalización al inicio
    #dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    #action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)


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
        # Comandos simples (solo forward)
        # ------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        
        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7


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

        # episodios más cortos al inicio
        #self.episode_length_s = 15.0
