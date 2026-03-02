# SPDX-FileCopyrightText: 2025-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import json
from contextlib import suppress
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Set, Union, Iterable, Type

import torch
from numpy import ndarray
from torch import nn, Tensor
from torch.cuda import is_available
from torch.optim import Optimizer
from torch.utils.data import DataLoader

try:
    from ..Language import Language
except ImportError:
    from vers_models.Language import Language

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

class InvalidConfigError(Exception):
    def __init__(self, message: str):
        """
        Exception raised when the config file is invalid or conflicts with the parameters.
        :param message: The message to display.
        """
        self.message = f"Invalid config file!\n{message}"
        super().__init__(self.message)


class BaseModel(ABC, nn.Module):
    ROOT_DIR_NAME: str = "VERS-Models"
    MODEL_ROOT_DIR_NAME: str = "models"
    LANGS_ROOT_DIR_NAME: str = "langs"
    EVALS_ROOT_DIR_NAME: str = "evals"
    ERRORS_ROOT_DIR_NAME: str = "errors"
    CHECKPOINTS_ROOT_DIR_NAME: str = "checkpoints"
    LOGS_ROOT_DIR_NAME: str = "logs"
    CONFIGS_ROOT_DIR_NAME: str = "configs"
    BANNED_KEYS: Set[str] = {
        "cls_name",
        "device",
        "lang_name",
        "class_name",
        "input_size",
        "output_size",
        "max_input_length",
        "max_output_length",
    }
    MANDATORY_KEYS: Set[str] = set()
    ALLOWED_KEYS: Set[str]  # abstract

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

    @classmethod
    def solve_paths(cls) -> tuple[Path, int, Path, Path, Path, Path, Path, Path, Path]:
        root_dir, relative_to_root = cls.get_root_dir()
        (
            lang_root,
            eval_root,
            errors_root,
            logs_root,
            checkpoints_root,
            configs_root,
            model_root
        ) = [
            root_dir / dir_name
            for dir_name in [
                cls.LANGS_ROOT_DIR_NAME,
                cls.EVALS_ROOT_DIR_NAME,
                cls.ERRORS_ROOT_DIR_NAME,
                cls.LOGS_ROOT_DIR_NAME,
                cls.CHECKPOINTS_ROOT_DIR_NAME,
                cls.CONFIGS_ROOT_DIR_NAME,
                cls.MODEL_ROOT_DIR_NAME
            ]
        ]

        return (
            root_dir,
            relative_to_root,
            lang_root,
            eval_root,
            errors_root,
            logs_root,
            checkpoints_root,
            configs_root,
            model_root
        )



    def set_paths(self, raise_twice=True) -> None:
        """
        Uses get_root_dir and the class variables to set the paths for the model, and other directories.
        """
        clone_repo = "does not exist, have you cloned the repository correctly ?"
        twice = "already exists, you've probably executed the script twice without realizing it."

        (
            self.root_dir,
            self.relative_to_root,
            self.lang_root,
            self.eval_root,
            self.errors_root,
            self.logs_root,
            self.checkpoints_root,
            self.configs_root,
            self.model_root
        ) = self.solve_paths()

        # This first to exit if lang_dir does not exist without creating the other directories
        self.lang_dir = self.lang_root / self.lang
        assert self.lang_dir.exists(), f"Language directory {self.lang_dir} does not exist, if you are using a new language, please create it first with the ``--make_lang`` argument."

        # Dirs that sould always exist, crash if not
        self.eval_dir = self.eval_root
        assert self.eval_dir.exists(), f"Eval directory {self.eval_dir} {clone_repo}"
        self.eval_path = self.eval_dir / f"{self.cls_name}_{self.start_datetime_str}.json"
        self.latent_path = self.eval_dir / f"{self.cls_name}_{self.start_datetime_str}_latents.npy"
        self.errors_dir = self.errors_root
        assert self.errors_dir.exists(), f"Errors directory {self.errors_dir} {clone_repo}"
        self.logs_dir = self.logs_root
        assert self.logs_dir.exists(), f"Logs directory {self.logs_dir} {clone_repo}"
        self.configs_dir = self.configs_root
        assert self.configs_dir.exists(), f"Configs directory {self.configs_dir} {clone_repo}"

        # Config file, should always exist
        self.config_file = self.configs_dir / f"{self.cls_name}.json"
        assert self.config_file.exists(), f"Config file {self.config_file} {clone_repo}"

        # Dirs to create
        self.model_dir = self.model_root / self.cls_name / self.start_datetime_str
        try:
            self.model_dir.mkdir(parents=True)
        except FileExistsError:
            if raise_twice:
                raise FileExistsError(f"Model directory {self.model_dir} {twice}")
        self.checkpoints_dir = self.checkpoints_root / self.cls_name / self.start_datetime_str
        try:
            self.checkpoints_dir.mkdir(parents=True)
        except FileExistsError:
            if raise_twice:
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
        elif isinstance(obj, torch.device):
            return None
        else:
            raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__()

        try:
            0 in self.__class__.ALLOWED_KEYS  # Check if the class has mandatory keys or is unset
        except AttributeError:
            pass
            # raise NotImplementedError(
            #     f"{self.__class__.__name__} does not have any mandatory keys, please set them in the class.")

        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        self.optimizer: Optional[Optimizer] = None
        self.criterion: Optional[nn.Module] = None

        self.start_datetime: datetime = datetime.now()
        self.start_datetime_str: str = self.start_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        self.params: dict = kwargs
        self.cls_name = self.__class__.__name__
        self.device: torch.device = torch.device(kwargs.get("device", "cuda" if is_available() else "cpu"))
        self.lang = kwargs["lang_name"]  # This should be set as a parameter rather than in the config file

        self.root_dir: Path = None
        self.relative_to_root: int = None
        self.lang_dir: Path = None
        self.eval_dir: Path = None
        self.latent_path: Path = None
        self.eval_path: Path = None
        self.errors_dir: Path = None
        self.logs_dir: Path = None
        self.configs_dir: Path = None
        self.config_file: Path = None
        self.model_dir: Path = None
        self.checkpoints_dir: Path = None
        self.set_paths(raise_twice=kwargs.get("raise_twice", True))

        # Read the config file and update the params
        self.params = self.read_config(**kwargs)


    def save(self) -> tuple[Path, Path]:
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

    @staticmethod
    def ensure_compatibility(
            model_s: Union[Path, Iterable[Path]],
            lang_name: str,
    ) -> Optional[Path]:
        """
        Ensure that the model is compatible with the current class and language.
        :param model_s: The model to check.
        :param lang_name: The language name to check.
        :return: The model if any is compatible, None otherwise. (should be considered as a Result object)
        """
        if isinstance(model_s, Path):
            model_s = [model_s]

        assert all(isinstance(model, Path) for model in model_s), "model_s must be a Path or an iterable of Paths"
        assert len(model_s) > 0, "model_s must be a non-empty iterable of Paths"

        for model in model_s:
            params_file = model / "params.json"
            if not params_file.exists():
                continue
            with params_file.open(mode="r", encoding="utf-8") as f:
                params = json.load(f)
            if params["lang_name"] == lang_name:
                return model

        return None



    @classmethod
    def load(
            cls,
            /,
            lang_name: str,
            *args,
            datetime_str: Optional[str] = None,
            default_to_latest: bool = True,
            device: Optional[Union[str, torch.device]] = None,
            **kwargs
    ) -> tuple[type["BaseModel"], dict[str, Any], Path]:
        """
        Load the model from the given path.
        :param lang_name: The language name to load the model for.
        :param datetime_str: The datetime string to load the model from.
        :param default_to_latest: If True, load the latest model if datetime_str is not provided and multiple models exist.
        :param device: The device to load the model on. If None, tries to use cuda.
        :return: The loaded model, the state and the old vocab size.
        """
        if device is None:
            device = "cuda" if is_available() else "cpu"

        model_root_dir = cls.get_root_dir()[0] / Path(cls.MODEL_ROOT_DIR_NAME) / cls.__name__

        if not model_root_dir.exists():
            raise FileNotFoundError(f"Model directory {model_root_dir} does not exist, no model to load.")

        if datetime_str is None:
            if default_to_latest:
                lst_models = sorted(model_root_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
                if len(lst_models) == 0:
                    raise FileNotFoundError(f"Model directory {model_root_dir} is empty, no model to load.")
                model_dir = cls.ensure_compatibility(lst_models, lang_name)
                assert model_dir is not None, f"No compatible models were found in {model_root_dir} for {lang_name}, please double check the language name and the desired model class."
            else:
                raise FileNotFoundError(
                    f"`default_to_latest` was manually set to False, please specify a datetime string if you want to load a specific model or leave the `default_to_latest` to True.")
        else:
            model_dir = model_root_dir / Path(datetime_str)
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Model directory {model_dir} does not exist, have you specified the correct datetime string ?")
            model_dir = cls.ensure_compatibility(model_dir, lang_name)
            assert model_dir is not None, f"Model directory {model_dir} does exist but was trained on another lang than {lang_name}, please double check the language name and the desired model class."

        with open(model_dir / Path("params.json"), "r", encoding="utf-8") as f:
            params = json.load(f)

        if params["class_name"] != cls.__name__:
            raise InvalidConfigError(
                f"Model {params['class_name']} is not compatible with the current class {cls.__name__}, please load the model using the correct class.")

        params["device"] = device

        model = cls(**params).to(device)

        model.load_state_dict(torch.load(model_dir / Path("model.pth"), map_location=device))

        if (model_dir / Path("optimizer.pth")).exists():
            model.optimizer.load_state_dict(torch.load(model_dir / Path("optimizer.pth"), map_location=device))

        if (model_dir / Path("criterion.pth")).exists():
            model.criterion.load_state_dict(torch.load(model_dir / Path("criterion.pth"), map_location=device))

        return model, params, model_dir

    @classmethod
    def load_with_languages(
            cls,
            /,
            lang_name: str,
            input_lang: "Language",
            output_lang: "Language", 
            *args,
            datetime_str: Optional[str] = None,
            default_to_latest: bool = True,
            device: Optional[Union[str, torch.device]] = None,
            **kwargs
    ) -> tuple[type["BaseModel"], dict[str, Any], Path]:
        """
        Load the model and automatically set the current languages for fine-tuning.
        
        :param lang_name: The language name to load the model for.
        :param input_lang: The input language object
        :param output_lang: The output language object
        :param datetime_str: The datetime string to load the model from.
        :param default_to_latest: If True, load the latest model if datetime_str is not provided.
        :param device: The device to load the model on. If None, tries to use cuda.
        :return: The loaded model with languages set, the state and the model directory.
        """
        model, params, model_dir = cls.load(
            lang_name, *args, datetime_str=datetime_str, 
            default_to_latest=default_to_latest, device=device, **kwargs
        )
        
        # Set the current languages for proper fine-tuning
        model.set_current_languages(input_lang, output_lang)
        
        return model, params, model_dir

    def set_current_languages(self, input_lang: "Language", output_lang: "Language"):
        """
        Set the current input and output languages for the model.
        This is needed for proper vocabulary mapping during fine-tuning.
        
        :param input_lang: Current input language
        :param output_lang: Current output language
        """
        self._current_input_lang = input_lang
        self._current_output_lang = output_lang

    def to_tensor(self, src:Union[ndarray, list, Tensor]) -> Tensor:
        if isinstance(src, (ndarray, list)):
            return torch.tensor(src, dtype=torch.long, device=self.device)
        elif isinstance(src, Tensor):
            return src.to(self.device)
        else:
            raise TypeError("src must be a numpy array, list, or torch tensor")

    def handle_unknown_tokens(self, src:Union[ndarray, list, Tensor], lang:Language) -> Tensor:
        """
        Checks for tokens not in the vocabulary and replaces them with the UNK token
        :param src: Input tensor or array with token indices
        :param lang: Language object for the input
        :return: Tensor with unknown tokens replaced by UNK_ID
        """
        tensor = self.to_tensor(src)

        # Create mask for tokens that exceed the vocabulary size
        if tensor.dim() == 1:
            mask = tensor >= lang.n_tokens
            if mask.any():
                print(f"Warning: Found unknown tokens in input: {tensor[mask].tolist()}")
                tensor[mask] = lang.UNK_ID
        else:  # For batches
            mask = tensor >= lang.n_tokens
            if mask.any():
                print(f"Warning: Found {mask.sum().item()} unknown tokens in batch")
                tensor[mask] = lang.UNK_ID

        return tensor


    def _merge_vocabularies(self, old_lang: "Language", new_lang: "Language") -> "Language":
        """
        Merge vocabularies by building upon the old language and extending it with new tokens.
        All old tokens keep their original IDs, and new tokens from new_lang are added after.
        
        :param old_lang: Original language vocabulary (will be extended with new tokens)
        :param new_lang: New language vocabulary (new tokens will be added from this)
        :return: Extended language object based on old_lang
        """
        # Check if the new language already contains all old language tokens in the right order
        # This happens when the new language is the same as the old one or was built from the old one
        if new_lang.n_tokens >= old_lang.n_tokens:
            
            for i in range(old_lang.n_tokens):
                if (
                    i not in new_lang.index2token 
                    or new_lang.index2token[i] != old_lang.index2token[i]
                ):
                        break
            else:
                # No merge needed - the new language already contains all old tokens in correct order
                return new_lang
        
        # Create a new language object based on the old language
        merged_lang = Language(old_lang.name, old_lang.sep)
        
        # Copy all attributes from the old language
        merged_lang.token2index = old_lang.token2index.copy()
        merged_lang.index2token = old_lang.index2token.copy() 
        merged_lang.token2count = old_lang.token2count.copy()
        merged_lang.n_tokens = old_lang.n_tokens
        merged_lang.max_length = old_lang.max_length
        
        new_tokens = set(new_lang.token2index.keys()) - set(merged_lang.token2index.keys())
        for token in new_tokens:
            merged_lang.token2index[token] = merged_lang.n_tokens # n_tokens == next available index

            merged_lang.index2token[merged_lang.n_tokens] = token
            merged_lang.token2count[token] += new_lang.token2count.get(token, 1) # .get() to be extra safe

            merged_lang.n_tokens += 1

        # Keep the biggest max_length between the two languages
        merged_lang.max_length = max(old_lang.max_length, new_lang.max_length)
        
        return merged_lang

    def _create_vocab_mapping(self, old_lang: "Language", new_lang: "Language") -> dict:
        """
        Create a mapping from old vocabulary indices to new vocabulary indices.
        Now creates an identity mapping since vocabularies are merged to preserve old IDs.
        
        :param old_lang: Original language vocabulary
        :param new_lang: New language vocabulary (should be merged with old_lang)
        :return: Dictionary mapping old indices to new indices
        """
        return {
            old_idx: new_lang.token2index[token]
            for old_idx, token in old_lang.index2token.items()
            if token in new_lang.token2index # If token doesn't exist in new vocabulary, it will be unmapped (lost)
            # But since we are supposed to merge vocabularies, this should not happen, I could remove it we'll see
        }

    def _resize_embedding_layer(
            self, 
            embedding_layer: nn.Embedding, 
            new_vocab_size: int, 
            vocab_mapping: dict,
            init_std: float = 0.1
    ) -> nn.Embedding:
        """
        Resize an embedding layer to accommodate a new vocabulary size.
        
        :param embedding_layer: Original embedding layer
        :param new_vocab_size: New vocabulary size
        :param vocab_mapping: Mapping from old indices to new indices
        :param init_std: Standard deviation for initializing new embeddings
        :return: New embedding layer with preserved and new weights
        """
        old_vocab_size, embed_dim = embedding_layer.weight.shape
        
        if new_vocab_size <= old_vocab_size:
            # No resizing needed if new vocabulary is not larger
            return embedding_layer
            
        # Create new embedding layer
        new_embedding = nn.Embedding(new_vocab_size, embed_dim).to(embedding_layer.weight.device)
        
        # Initialize new embeddings with normal distribution
        nn.init.normal_(new_embedding.weight, mean=0.0, std=init_std)
        
        # Copy existing weights using the mapping
        with torch.no_grad():
            for old_idx, new_idx in vocab_mapping.items():
                if old_idx < old_vocab_size and new_idx < new_vocab_size:
                    new_embedding.weight[new_idx] = embedding_layer.weight[old_idx]
        
        return new_embedding

    def _resize_linear_layer(
            self, 
            linear_layer: nn.Linear, 
            new_output_size: int, 
            vocab_mapping: dict,
            init_std: float = 0.1,
            resize_input: bool = False
    ) -> nn.Linear:
        """
        Resize a linear layer's output (or input) dimension to accommodate new vocabulary.
        
        :param linear_layer: Original linear layer
        :param new_output_size: New output size (or input size if resize_input=True)
        :param vocab_mapping: Mapping from old indices to new indices
        :param init_std: Standard deviation for initializing new weights
        :param resize_input: Whether to resize input dimension instead of output
        :return: New linear layer with preserved and new weights
        """
        if resize_input:
            old_size = linear_layer.in_features
            new_size = new_output_size
            in_features = new_size
            out_features = linear_layer.out_features
        else:
            old_size = linear_layer.out_features
            new_size = new_output_size
            in_features = linear_layer.in_features
            out_features = new_size
        
        if new_size <= old_size:
            # No resizing needed if new size is not larger
            return linear_layer
            
        # Create new linear layer
        new_linear = nn.Linear(in_features, out_features).to(linear_layer.weight.device)
        
        # Initialize new weights with normal distribution
        nn.init.normal_(new_linear.weight, mean=0.0, std=init_std)
        if new_linear.bias is not None:
            nn.init.normal_(new_linear.bias, mean=0.0, std=init_std)
        
        # Copy existing weights using the mapping
        with torch.no_grad():
            if resize_input:
                # Resizing input dimension
                for old_idx, new_idx in vocab_mapping.items():
                    if old_idx < old_size and new_idx < new_size:
                        new_linear.weight[:, new_idx] = linear_layer.weight[:, old_idx]
            else:
                # Resizing output dimension
                for old_idx, new_idx in vocab_mapping.items():
                    if old_idx < old_size and new_idx < new_size:
                        new_linear.weight[new_idx] = linear_layer.weight[old_idx]
                        if new_linear.bias is not None and linear_layer.bias is not None:
                            new_linear.bias[new_idx] = linear_layer.bias[old_idx]
        
        return new_linear


    def to(self, device: Union[str, torch.device]) -> Type["BaseModel"]:
        """
        Move the model to the specified device.
        :param device: The device to move the model to.
        :return: The model itself.
        """
        self.device = torch.device(device)
        super().to(device)
        # if self.optimizer is not None:
        #     self.optimizer = self.optimizer.to(self.device)
        # if self.criterion is not None:
        #     self.criterion = self.criterion.to(self.device)
        return self

    @abstractmethod
    def partial_forward(self, src:Tensor) -> Tensor:
        """
        Perform a forward pass on the source tensor only, typically used for encoder-only models.
        :param src: The source tensor.
        :return: The output tensor.
        """
        raise NotImplementedError("Partial forward method not implemented")

    @abstractmethod
    def forward(self, src:Tensor, trg:Tensor) -> Tensor:
        raise NotImplementedError("Forward method not implemented")

    def predict(
            self,
            src:Union[ndarray, list, Tensor, DataLoader],
            lang_output:Language,
            batch_mode: bool = False
    ) -> Union[Iterable[str], Iterable[Iterable[str]]]:
        self.eval()

        if isinstance(src, DataLoader):
            return self._handle_dataloader_input(src, lang_output)

        if batch_mode:
            return self._predict_batch(src, lang_output)

        src = self.to_tensor(src)
        return self._predict_single(src, lang_output)

    @abstractmethod
    def _predict_single(self, src: Tensor, lang_output: Language) -> Iterable[str]:
        raise NotImplementedError("_predict_single method not implemented")

    def _predict_batch(self, src: Tensor, lang_output: Language) -> Iterable[Iterable[str]]:
        raise NotImplementedError("_predict_batch method not implemented")

    def _handle_dataloader_input(self, src: DataLoader, lang_output: Language):
        return [
            self.predict(batch, lang_output, batch_mode=True)
            for batch in src
        ]

    def _process_batch_input(self, src: Tensor) -> Tensor:
        src = self.to_tensor(src)
        if src.dim() == 1:
            src = src.unsqueeze(0)
        return src

    @abstractmethod
    def do_train(
            self,
            device:torch.device,
            dataloader:DataLoader,
            num_epochs:int = 10,
            eval_every: Optional[int] = None,
            eval_fn: Optional[callable] = None,
            eval_args: Optional[dict] = None,
            from_epoch: int = 0,
            **kwargs,
    ):
        raise NotImplementedError("Train method not implemented")

    @abstractmethod
    def finetune(
            self,
            new_input_lang: "Language",
            new_output_lang: "Language",
            preserve_weights: bool = True,
            init_std: float = 0.1
    ):
        """
        Fine-tune the model for new languages by adjusting layer sizes and preserving weights.
        
        :param new_input_lang: New input language with potentially larger vocabulary
        :param new_output_lang: New output language with potentially larger vocabulary  
        :param preserve_weights: Whether to preserve existing weights for common vocabulary
        :param init_std: Standard deviation for initializing new weights
        :return: Self for method chaining
        """
        raise NotImplementedError("Finetune method not implemented")

    def __del__(self):
        """
        Clean up the model directory and checkpoints directory if they are empty.
        """
        # To ignore errors during cleanup as the object might be partially initialized
        with suppress(AttributeError, OSError, KeyboardInterrupt):
            if (hasattr(self, 'model_dir') and self.model_dir.exists() and
                not any(self.model_dir.iterdir())):
                self.model_dir.rmdir()

            if (hasattr(self, 'checkpoints_dir') and self.checkpoints_dir.exists() and
                not any(self.checkpoints_dir.iterdir())):
                self.checkpoints_dir.rmdir()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.params})"

    def __hash__(self):
        return hash((self.__class__.__name__, frozenset(self.params.items())))

    def __eq__(self, other):
        if not isinstance(other, BaseModel):
            return False
        return self.__class__.__name__ == other.__class__.__name__ and self.params == other.params

    def __ne__(self, other):
        return not self.__eq__(other)

