from .BaseModel import BaseModel, InvalidConfigError

models: dict[str, type[BaseModel]] = {
    "base": BaseModel,
}

__all__ = [
    "BaseModel",
    "InvalidConfigError",
    "models",
]
