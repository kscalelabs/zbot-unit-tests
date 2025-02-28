from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import DCMotorCfg

from .actuator_pd import IdentifiedActuator


@configclass
class IdentifiedActuatorCfg(DCMotorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    class_type: type = IdentifiedActuator

    friction_static: float = MISSING
    """ (in N-m)."""
    activation_vel: float = MISSING
    """ (in Rad/s)."""
    friction_dynamic: float = MISSING
    """ (in N-m-s/Rad)."""
