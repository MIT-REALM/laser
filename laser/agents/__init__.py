from typing import Union

from .fql import FQLAgent, FQLCfg
from .ifql import IFQLAgent, IFQLCfg
from .dsrl import DSRLAgent, DSRLCfg
from .reform import ReFORMAgent, ReFORMCfg
from .laser import LASERAgent, LASERCfg
from .qam import QAMAgent, QAMCfg
from .qam_e import QAMEAgent, QAMECfg

agents = dict(
    fql=FQLAgent,
    ifql=IFQLAgent,
    dsrl=DSRLAgent,
    reform=ReFORMAgent,
    laser=LASERAgent,
    qam=QAMAgent,
    qam_e=QAMEAgent,
)

agent_cfgs = dict(
    fql=FQLCfg,
    ifql=IFQLCfg,
    dsrl=DSRLCfg,
    reform=ReFORMCfg,
    laser=LASERCfg,
    qam=QAMCfg,
    qam_e=QAMECfg,
)

Agent = Union[FQLAgent, IFQLAgent, DSRLAgent, ReFORMAgent, LASERAgent, QAMAgent, QAMEAgent]
AgentCfg = Union[FQLCfg, IFQLCfg, DSRLCfg, ReFORMCfg, LASERCfg, QAMCfg, QAMECfg]
