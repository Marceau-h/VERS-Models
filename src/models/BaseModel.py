import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Set, Union

import torch
from numpy import ndarray
from torch import nn, Tensor
from torch.cuda import is_available
from torch.optim import Optimizer


class InvalidConfigError(Exception):
    def __init__(self, message: str):
        """
        Exception raised when the config file is invalid or conflicts with the parameters.
        :param message: The message to display.
        """
        self.message = f"Invalid config file!\n{message}"
        super().__init__(self.message)


class BaseModel(ABC, nn.Module):
    ROOT_DIR_NAME: str = "VERS"
    MODEL_ROOT_DIR_NAME: str = "models"
    LANGS_ROOT_DIR_NAME: str = "langs"
    EVALS_ROOT_DIR_NAME: str = "evals"
    ERRORS_ROOT_DIR_NAME: str = "errors"
    CHECKPOINTS_ROOT_DIR_NAME: str = "checkpoints"
    LOGS_ROOT_DIR_NAME: str = "logs"
    CONFIGS_ROOT_DIR_NAME: str = "configs"
    BANNED_KEYS: Set[str] = {
        "pretty_name",
        "device",
        "lang",
        "class_name",
    }
    MANDATORY_KEYS: Set[str] = set()

    @classmethod
    def get_root_dir(cls) -> tuple[Path, int]:
        """
        Get the root directory of the project and the number of directories to go up to reach it.
        """
        relative_to_root = 0
        cwd = Path.cwd()
        while cwd.name != cls.ROOT_DIR_NAME:
            relative_to_root += 1
            cwd = cwd.parent
        return cwd, relative_to_root

    def set_paths(self) -> None:
        """
        Uses get_root_dir and the class variables to set the paths for the model, and other directories.
        """
        clone_repo = "does not exist, have you cloned the repository correctly ?"
        twice = "already exists, you've probably executed the script twice withou realizing it."

        self.root_dir, self.relative_to_root = self.get_root_dir()

        # This first to exit if lang_dir does not exist without creating the other directories
        self.lang_dir = self.root_dir / Path(self.LANGS_ROOT_DIR_NAME) / self.lang
        assert self.lang_dir.exists(), f"Language directory {self.lang_dir} does not exist, if you are using a new language, please create it first with the ``--make_lang`` argument."

        # Dirs that sould always exist, crash if not
        self.eval_dir = self.root_dir / Path(self.EVALS_ROOT_DIR_NAME)
        assert self.eval_dir.exists(), f"Eval directory {self.eval_dir} {clone_repo}"
        self.errors_dir = self.root_dir / Path(self.ERRORS_ROOT_DIR_NAME)
        assert self.errors_dir.exists(), f"Errors directory {self.errors_dir} {clone_repo}"
        self.logs_dir = self.root_dir / Path(self.LOGS_ROOT_DIR_NAME)
        assert self.logs_dir.exists(), f"Logs directory {self.logs_dir} {clone_repo}"
        self.configs_dir = self.root_dir / Path(self.CONFIGS_ROOT_DIR_NAME)
        assert self.configs_dir.exists(), f"Configs directory {self.configs_dir} {clone_repo}"

        # Config file, should always exist
        self.config_file = self.configs_dir / Path(f"{self.pretty_name}.json")
        assert self.config_file.exists(), f"Config file {self.config_file} {clone_repo}"

        # Dirs to create
        self.model_dir = self.root_dir / Path(self.MODEL_ROOT_DIR_NAME) / self.pretty_name / self.start_datetime_str
        try:
            self.model_dir.mkdir(parents=True)
        except FileExistsError:
            raise FileExistsError(f"Model directory {self.model_dir} {twice}")
        self.checkpoints_dir = self.root_dir / Path(
            self.CHECKPOINTS_ROOT_DIR_NAME) / self.pretty_name / self.start_datetime_str
        try:
            self.checkpoints_dir.mkdir(parents=True)
        except FileExistsError:
            raise FileExistsError(f"Checkpoints directory {self.checkpoints_dir} {twice}")

    def read_config(self, **kwargs) -> dict[str, Any]:
        """
        Read the config file and update it with the kwargs.
        """
        # If pretrained, we don't want to read the config file
        if kwargs["pretrained"] is True:  # `is True` means really set to True and not to a truthy value
            return kwargs

        with open(self.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        for banned_key in self.BANNED_KEYS:
            if banned_key not in config:
                continue
            if config[banned_key] == "TO SPECIFY":
                continue
            raise InvalidConfigError(
                f"Key {banned_key} is not allowed in the config file, please remove it or set it to `TO SPECIFY`.")

        for key in self.MANDATORY_KEYS:
            if key not in config:
                raise InvalidConfigError(f"Key {key} is mandatory in the config file, please add it and set it")
            if config[key] == "TO SPECIFY":
                raise InvalidConfigError(
                    f"Key {key} is mandatory in the config file, please add it and set it to a valid value.")

        config.update(kwargs)

        for key, value in config.items():
            if value == "TO SPECIFY":
                raise InvalidConfigError(
                    f"Key {key} is not set by the config file, please set it to a valid value with the `--{key}` argument.")

        config["class_name"] = self.__class__.__name__

        return config

    @staticmethod
    def jsonify_types(obj):
        if isinstance(obj, Path):
            return obj.as_posix()
        elif isinstance(obj, ndarray):
            return obj.tolist()
        elif isinstance(obj, Tensor):
            return obj.tolist()
        else:
            raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__()

        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        self.optimizer: Optional[Optimizer] = None
        self.criterion: Optional[nn.Module] = None

        self.start_datetime: datetime = datetime.now()
        self.start_datetime_str: str = self.start_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        self.params: dict = kwargs
        self.pretty_name = kwargs.get("pretty_name", self.__class__.__name__)
        self.device: torch.device = torch.device(kwargs.get("device", "cuda" if is_available() else "cpu"))
        self.lang = kwargs["lang"]  # This should be set as a parameter rather than in the config file

        self.root_dir: Path = None
        self.relative_to_root: int = None
        self.lang_dir: Path = None
        self.eval_dir: Path = None
        self.errors_dir: Path = None
        self.logs_dir: Path = None
        self.configs_dir: Path = None
        self.config_file: Path = None
        self.model_dir: Path = None
        self.checkpoints_dir: Path = None
        self.set_paths()

        # Read the config file and update the params
        self.read_config(**kwargs)

    def save_model(self) -> tuple[Path, Path]:
        """
        Save the model (and parameters) to its directory.
        """
        torch.save(self.state_dict(), self.model_dir / Path("model.pth"))

        if self.optimizer is not None:
            self.optimizer.zero_grad()
            torch.save(self.optimizer.state_dict(), self.model_dir / Path("optimizer.pth"))

        if self.criterion is not None:
            torch.save(self.criterion.state_dict(), self.model_dir / Path("criterion.pth"))

        self.params["pretrained"] = True

        with open(self.model_dir / Path("params.json"), "w", encoding="utf-8") as f:
            json.dump(self.params, f, default=self.jsonify_types, indent=4)

        print("Model and parameters saved successfully")

        return self.model_dir / Path("model.pth"), self.model_dir / Path("params.json")

    @classmethod
    def load_model(
            cls,
            /,
            *args,
            pretty_name: Optional[str] = None,
            datetime_str: Optional[str] = None,
            default_to_latest: bool = True,
            device: Optional[Union[str, torch.device]] = None,
            **kwargs
    ) -> tuple[type["BaseModel"], Any, Path]:
        """
        Load the model from the given path.
        :param pretty_name: The name of the model to load. If None, use the class name.
        :param datetime_str: The datetime string to load the model from.
        :param default_to_latest: If True, load the latest model if datetime_str is not provided and multiple models exist.
        :param device: The device to load the model on. If None, tries to use cuda.
        :return: The loaded model, the state and the old vocab size.
        """
        if device is None:
            device = "cuda" if is_available() else "cpu"

        if pretty_name is None:
            pretty_name = cls.__name__

        model_root_dir = cls.get_root_dir()[0] / Path(cls.MODEL_ROOT_DIR_NAME) / pretty_name

        if not model_root_dir.exists():
            raise FileNotFoundError(f"Model directory {model_root_dir} does not exist, no model to load.")

        if datetime_str is None:
            if default_to_latest:
                lst_models = sorted(model_root_dir.iterdir(), key=lambda x: x.stat().st_mtime)
                if len(lst_models) == 0:
                    raise FileNotFoundError(f"Model directory {model_root_dir} is empty, no model to load.")
                model_dir = lst_models[-1]
            else:
                raise FileNotFoundError(f"`default_to_latest` was manually set to False, please specify a datetime string if you want to load a specific model or leave the `default_to_latest` to True.")
        else:
            model_dir = model_root_dir / Path(datetime_str)
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory {model_dir} does not exist, have you specified the correct datetime string ?")

        with open(model_dir / Path("params.json"), "r", encoding="utf-8") as f:
            params = json.load(f)

        if params["class_name"] != cls.__name__:
            raise InvalidConfigError(f"Model {params['class_name']} is not compatible with the current class {cls.__name__}, please load the model using the correct class.")

        params.device = device

        model = cls(**params)

        model.load_state_dict(torch.load(model_dir / Path("model.pth"), map_location=device))

        if (model_dir / Path("optimizer.pth")).exists():
            model.optimizer.load_state_dict(torch.load(model_dir / Path("optimizer.pth"), map_location=device))

        if (model_dir / Path("criterion.pth")).exists():
            model.criterion.load_state_dict(torch.load(model_dir / Path("criterion.pth"), map_location=device))

        return model, params, model_dir



