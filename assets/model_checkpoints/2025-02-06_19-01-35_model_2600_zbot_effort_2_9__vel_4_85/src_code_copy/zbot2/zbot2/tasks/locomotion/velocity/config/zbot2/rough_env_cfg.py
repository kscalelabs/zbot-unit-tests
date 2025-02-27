from omni.isaac.lab.utils import configclass

from zbot2.tasks.locomotion.velocity.velocity_env_cfg import \
    LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from zbot2.assets.zbot2 import ZBOT2_CFG


@configclass
class Zbot2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = ZBOT2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class Zbot2RoughEnvCfg_PLAY(Zbot2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
