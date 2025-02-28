import omni.isaac.lab.sim as sim_utils
from zbot2.actuators import IdentifiedActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from zbot2.assets import ISAAC_ASSET_DIR
import math


ZBOT_BENT_KNEES_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": -math.radians(31.6),
    "left_knee": math.radians(65.6),
    "left_ankle": math.radians(31.6),
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": math.radians(31.6),
    "right_knee": -math.radians(65.6),
    "right_ankle": -math.radians(31.6),
    "left_shoulder_yaw": 0.0,
    "left_shoulder_pitch": 0.0,
    "left_elbow": 0.0,
    "left_gripper": 0.0,
    "right_shoulder_yaw": 0.0,
    "right_shoulder_pitch": 0.0,
    "right_elbow": 0.0,
    "right_gripper": 0.0,
}

ZBOT_STRAIGHT_KNEES_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": 0.0,
    "left_knee": 0.0,
    "left_ankle": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.0,
    "right_knee": 0.0,
    "right_ankle": 0.0,
    "left_shoulder_yaw": 0.0,
    "left_shoulder_pitch": 0.0,
    "left_elbow": 0.0,
    "left_gripper": 0.0,
    "right_shoulder_yaw": 0.0,
    "right_shoulder_pitch": 0.0,
    "right_elbow": 0.0,
    "right_gripper": 0.0,
}

# # ZBOT2_ACTUATOR_CFG = IdentifiedActuatorCfg(
# #    joint_names_expr=[".*"],
# #    effort_limit=1.9,            
# #    velocity_limit=1.0,
# #    saturation_effort=1.9,
# #    stiffness={".*": 21.1},
# #    damping={".*": 1.084},
# #    armature={".*": 0.045},
# #    friction_static=0.03,
# #    activation_vel=0.1,
# #    friction_dynamic=0.01,
# # )

# ZBOT2_ACTUATOR_CFG = IdentifiedActuatorCfg(
#    joint_names_expr=[".*"],
#    effort_limit=2.94,            
#    velocity_limit=1.0,
#    saturation_effort=3.0,
#    stiffness={".*": 150.0},
#    damping={".*": 5.0},
#    armature={".*": 0.045},
#    friction_static=0.03,
#    activation_vel=0.1,
#    friction_dynamic=0.01,
# )

STS3250_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit=2.90,           # N*m (from 16 kg*cm rated torque)
    velocity_limit=4.85,         # rad/s (from 60 deg in 0.133 s no-load speed)
    saturation_effort=4.90,      # N*m (from 50 kg*cm stall torque)
    stiffness={".*": 150.0},     # N*m/rad, chosen to yield ~0.01 rad error at full torque
    damping={".*": 5.084},         # N*m/(rad/s), a starting value for critically damped response
    armature={".*": 0.045},       # kg*m^2, estimated effective inertia (requires tuning)
    friction_static=0.03,
    activation_vel=0.1,
    friction_dynamic=0.01,
)

STS3215_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit=0.98,           # N*m (10 kg*cm continuous torque)
    velocity_limit=4.72,         # rad/s (from 60 deg/0.222 s no-load speed)
    saturation_effort=2.94,      # N*m (30 kg*cm stall torque)
    stiffness={".*": 100.0},     # N*m/rad, estimated based on torque scaling
    damping={".*": 3.0},         # N*m/(rad/s), scaled proportionally from a similar actuator
    armature={".*": 0.022},      # kg*m^2, estimated effective inertia (scaled by weight)
    friction_static=0.03,
    activation_vel=0.1,
    friction_dynamic=0.01,
)

ZBOT2_ACTUATOR_CFG = STS3250_ACTUATOR_CFG

ZBOT2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/zbot2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.39),
        joint_pos=ZBOT_BENT_KNEES_POS,
    ),
    actuators={"zbot2_actuators": ZBOT2_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
