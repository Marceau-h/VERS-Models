# SPDX-FileCopyrightText: 2028-present Marceau <git@marceau-h.fr>
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
except ImportError:
    from vers_models.Language import Language, read_data
    from vers_models.eval import random_predict, do_full_eval
    from vers_models.models import models
    from vers_models.train import auto_train


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
):
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
        (
            model,
            lang_input,
            lang_output,
            losses,
            evals,
            (X_train, X_dev, X_test, y_train, y_dev, y_test),
        ) = auto_train(
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
        model, state, old_vocab_size = model_class.load(datetime_str, default_to_latest, device)

        X_train, X_dev, X_test, y_train, y_dev, y_test, lang_input, lang_output = read_data(lang_path=lang_root / lang_name)
        print("Model, data, and parameters loaded successfully")

    # Test prediction
    random_predict(X_dev, y_dev, lang_input, lang_output, model, device=device, nb_predictions=nb_predictions)

    if full_eval:
        do_full_eval(X_dev, y_dev, lang_input, lang_output, model, device=device)
