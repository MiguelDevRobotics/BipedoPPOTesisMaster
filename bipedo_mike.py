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

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##


DcMotor = DCMotorCfg(
    joint_names_expr=[".*"],
    saturation_effort = 120,
    effort_limit=4.4,
    effort_limit_sim = 4.4,
    velocity_limit=2.0,
    stiffness={".*": 0},
    damping={".*": 5.0e6},
)

implicit = ImplicitActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit_sim=4.41,     
    velocity_limit_sim=5.82,  
    stiffness=50,
    damping=0.5,           
)

BIPED_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/bipedo",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/absolute/path/to/your/usd/bipedo.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=8,
        ),
        
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        #copy_from_source=False,
        #joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="velocity"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={
            "R1": 0.24307825538988628,
            "R2": 0.0,
            "R3": -1.1042811084848414,
            "R4": 0.0,
            "R5": 0.8612017789295812,
            "R6": 0.0,
            "R7": -0.24307825538988628,
            "R8": 0.0,
            "R9": 1.1042811084848414,
            "R10": 0.0,
            "R11": -0.8612017789295812,
            "R12": 0.0,
        },
        joint_vel = {".*": 0.0},
    ),
    actuators = {'body': implicit
        
    },
    soft_joint_pos_limit_factor=0.95,
)
