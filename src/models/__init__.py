from .BaseModel import BaseModel, InvalidConfigError
from .S2SNoAttn import S2SNoAttn
from .Transfo import Transfo
from .S2SMultiHeadAttn import S2SMultiHeadAttn

models: dict[str, type[BaseModel]] = {
    "base": BaseModel,
    "no_attn": S2SNoAttn,
    "transfo": Transfo,
    "multi_head_attn": S2SMultiHeadAttn,
}

__all__ = [
    "BaseModel",
    "InvalidConfigError",
    "models",
    "S2SNoAttn",
    "Transfo",
]
