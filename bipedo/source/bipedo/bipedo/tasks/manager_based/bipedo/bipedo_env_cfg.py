from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    ActionsCfg,
)

from isaaclab_assets.robots.bipedo_mike import BIPED_CFG   # tu articulation cfg

'''import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg

def com_zmp_reward(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    # 1. References
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    
    # 2. CoM Acceleration (Estimated)
    dt = env.step_dt 
    current_com_vel = asset.data.root_com_vel_w[:, :3]

    if not hasattr(env, "prev_com_vel"):
        # Ensure it's on the same device (GPU/CPU)
        env.prev_com_vel = current_com_vel.clone()

    com_acc = (current_com_vel - env.prev_com_vel) / dt
    env.prev_com_vel[:] = current_com_vel

    # 3. CoM Data
    com_pos = asset.data.root_com_pos_w
    com_xy = com_pos[:, :2]
    z_com = com_pos[:, 2:3] 
    g = 9.81

    # 4. Center of Pressure (CoP) Calculation
    raw_contact_pos = contact_sensor.data.contact_pos_w[:, sensor_cfg.body_ids]
    
    # Average points per link and handle NaNs (when feet are in the air)
    contact_pos_per_link = torch.nanmean(raw_contact_pos, dim=2) 
    contact_pos_per_link = torch.nan_to_num(contact_pos_per_link, nan=0.0)

    # Net forces per link (Z-component)
    fz = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp(min=0.0)
    weights = fz + 1e-6 

    # CoP result
    cop_xy = torch.sum(contact_pos_per_link[:, :, :2] * weights.unsqueeze(-1), dim=1) / torch.sum(weights, dim=1, keepdim=True)

    # 5. ZMP Equation (LIPM)
    zmp_theoretical = com_xy - (z_com / g) * com_acc[:, :2]

    # 6. Reward (Quadratic Penalty)
    error = torch.norm(zmp_theoretical - cop_xy, dim=-1)
    
    return torch.square(error)'''





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
            weight=0.25,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["link_6", "link_12"]
                ),
                "command_name": "base_velocity",
                "threshold": 0.4,
            },
        )

    '''stand_still_joint_deviation = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-0.1,
            params={
                "command_name": "base_velocity",
                #"command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        
        )'''
    
    feet_silde = RewTerm(
        func = mdp.feet_slide,
        weight = -0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["link_6", "link_12"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["link_6", "link_12"])
            }
    )

    '''zmp = RewTerm(
        func=com_zmp_reward,
        weight = 0,#-1e-5,
        params={
            "sensor_cfg": SceneEntityCfg("zmp_contact_sensor", body_names=["link_6", "link_12"]),
            "asset_cfg": SceneEntityCfg("robot")
            }
    )'''
      
    feet_air_time = None
    # menos penalización al inicio
    #dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    #action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)


# ------------------------
# Accion de velocidad para el vipedo
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

        '''nuevo_sensor_cfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(link_6|link_12)",
            update_period=0.0,
            track_contact_points=True,
            filter_prim_paths_expr=["/World/ground"],
            history_length=3 # Opcional
        )

        self.scene.zmp_contact_sensor = nuevo_sensor_cfg'''
        
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

        self.rewards.flat_orientation_l2 = None
        self.rewards.dof_torques_l2=None
        self.rewards.dof_pos_limits=None


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
