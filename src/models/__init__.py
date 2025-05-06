from .BaseModel import BaseModel, InvalidConfigError
from .S2SNoAttn import S2SNoAttn

models: dict[str, type[BaseModel]] = {
    "base": BaseModel,
    "no_attn": S2SNoAttn,
}

__all__ = [
    "BaseModel",
    "InvalidConfigError",
    "models",
    "S2SNoAttn",
]
