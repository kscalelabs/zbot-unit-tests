import omni.isaac.lab.sim as sim_utils
from kbot.actuators import IdentifiedActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from kbot.assets import ISAAC_ASSET_DIR
import math

KBOT_04_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_04"],
    effort_limit=120.0,            # from forcerange "-120 120"
    velocity_limit=100.0,
    saturation_effort=340.0,       # double the effort_limit
    stiffness={
        # Hip pitch joints get a higher kp (250) than knee joints (200)
        ".*hip_pitch_04": 250.0,
        ".*knee_04": 200.0,
    },
    damping={
        # Corresponding damping values from kv in the XML:
        ".*hip_pitch_04": 30.0,
        ".*knee_04": 8.0,
    },
    armature={".*": 0.01},
    friction_static=0.0,
    activation_vel=0.1,
    friction_dynamic=0.0,
)

KBOT_03_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_03"],
    effort_limit=60.0,             # forcerange "-60 60"
    velocity_limit=100.0,
    saturation_effort=120.0,       # double the effort_limit
    stiffness={".*": 150.0},       # all _03 joints use kp=150
    damping={".*": 8.0},           # all _03 joints use kv=8
    armature={".*": 0.01},
    friction_static=0.0,
    activation_vel=0.1,
    friction_dynamic=0.0,
)

KBOT_02_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_02"],
    effort_limit=17.0,             # forcerange "-17 17"
    velocity_limit=100.0,
    saturation_effort=34.0,        # double the effort_limit
    stiffness={
        # For _02 joints, some (shoulder yaw and elbow) have kp=50, while others (wrist and ankle) use kp=20.
        ".*(shoulder|elbow)_02": 50.0,
        ".*(wrist|ankle)_02": 20.0,
    },
    damping={
        # Similarly, damping (kv) is 5 for shoulder/elbow and 2 for wrist/ankle joints.
        ".*(shoulder|elbow)_02": 5.0,
        ".*(wrist|ankle)_02": 2.0,
    },
    armature={".*": 0.01},
    friction_static=0.0,
    activation_vel=0.1,
    friction_dynamic=0.0,
)

KBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/kbot.usd",
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
        # TODO: try 4 steps 
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            "left_shoulder_pitch_03": math.radians(0),
            "left_shoulder_roll_03": 0.0,
            "left_shoulder_yaw_02": 0.0,
            "left_elbow_02": -math.radians(90), 
            # "left_elbow_02": 0.0,
            "left_wrist_02": 0.0,

            "right_shoulder_pitch_03": -math.radians(0),
            "right_shoulder_roll_03": 0.0,
            "right_shoulder_yaw_02": 0.0,
            "right_elbow_02": math.radians(90),
            # "right_elbow_02": 0.0,
            "right_wrist_02": 0.0,
            
            'left_hip_pitch_04': math.radians(20),
            # 'left_hip_pitch_04': 0.0,
            'left_hip_roll_03': 0.0,
            'left_hip_yaw_03': 0.0,
            'left_knee_04': math.radians(40),
            # 'left_knee_04': 0.0,
            'left_ankle_02': -math.radians(20),
            # 'left_ankle_02': 0.0,

            'right_hip_pitch_04': -math.radians(20),
            # 'right_hip_pitch_04': 0.0,
            'right_hip_roll_03': 0.0,
            'right_hip_yaw_03': 0.0,
            'right_knee_04': -math.radians(40),
            # 'right_knee_04': 0.0,
            'right_ankle_02': -math.radians(20),
            # 'right_ankle_02': 0.0,
        },
    ),
    actuators={
        "kbot_04": KBOT_04_ACTUATOR_CFG,
        "kbot_03": KBOT_03_ACTUATOR_CFG,
        "kbot_02": KBOT_02_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)