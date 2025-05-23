# SPDX-FileCopyrightText: 2025-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path
from typing import Optional

import torch

try:
    from .Language import Language, read_data
    from .eval import random_predict, do_full_eval
    from .models import models
    from .train import auto_train
    from .profiler import profiler_wrapper
except ImportError:
    from vers_models.Language import Language, read_data
    from vers_models.eval import random_predict, do_full_eval
    from vers_models.models import models
    from vers_models.train import auto_train
    from vers_models.profiler import profiler_wrapper


def main(
        do_train: bool = False,
        num_epochs: int = 10,
        batch_size: Optional[int] = None,
        min_batch_size: Optional[int] = None,
        max_batch_size: Optional[int] = None,

        lang_input: str = "",
        lang_name: str = "",
        make_lang: bool = False,
        overwrite_lang: bool = False,

        full_eval: bool = False,
        nb_predictions: int = 10,

        model_class: str = "S2SNoAttn",
        model_args: dict = None,

        datetime_str: str = None,
        default_to_latest: bool = True,
        with_profiler: bool = False,
):

    train_func = profiler_wrapper(auto_train, profile_=with_profiler)
    full_eval_func = profiler_wrapper(do_full_eval, profile_=with_profiler)
    random_eval_func = profiler_wrapper(random_predict, profile_=with_profiler)

    assert lang_name, "lang_name must be provided"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_args["lang_name"] = lang_name
    model_args["device"] = device

    model_class = models[model_class]
    (
        root_dir,
        relative_to_root,
        lang_root,
        eval_root,
        errors_root,
        logs_root,
        checkpoints_root,
        configs_root,
        model_root
    ) = model_class.solve_paths()

    if make_lang:
        assert lang_input, "lang_input must be provided when make_lang is True"
        lang_input = Path(lang_input)
        assert lang_input.exists(), f"lang_input {lang_input} does not exist"

        if lang_input.suffix == ".json":
            X, y, l1, l2 = Language.read_data_from_json(lang_input)
        else:
            X, y, l1, l2 = Language.read_data_from_txt(lang_input)

        Language.save_data(X, y, l1, l2, lang_path=lang_root / lang_name, overwrite=overwrite_lang)

    if do_train:
        model_args["pretrained"] = False
        model, lang_input, lang_output, losses, evals, (X_train, X_dev, X_test, y_train, y_dev, y_test) = \
            train_func(
                model_class=model_class,
                model_args=model_args,
                num_epochs=num_epochs,
                lang_dir=lang_root / lang_name,
                batch_size=batch_size,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
            )

        model.save()

    else:
        model, state, old_vocab_size = model_class.load(
            lang_name=lang_name,
            datetime_str=datetime_str,
            default_to_latest=default_to_latest,
            device=device
        )
        X_train, X_dev, X_test, y_train, y_dev, y_test, lang_input, lang_output = \
            read_data(lang_path=lang_root / lang_name)
        print("Model, data, and parameters loaded successfully")

    if full_eval:
        full_eval_func(X_dev, y_dev, lang_input, lang_output, model, batch_size)
    else:
        random_eval_func(X_dev, y_dev, lang_input, lang_output, model, batch_size, nb_predictions=nb_predictions)
