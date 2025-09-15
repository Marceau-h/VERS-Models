# SPDX-FileCopyrightText: 2025-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterable, Union

import numpy as np
import torch
import polars as pl

try:
    from .Language import Language, read_data, read_data_1_lang
    from .eval import random_predict, do_full_eval, get_partial_output
    from .models import models, BaseModel
    from .train import auto_train
    from .profiler import profiler_wrapper
except ImportError:
    from vers_models.Language import Language, read_data, read_data_1_lang
    from vers_models.eval import random_predict, do_full_eval, get_partial_output
    from vers_models.models import models, BaseModel
    from vers_models.train import auto_train
    from vers_models.profiler import profiler_wrapper


def predict_one_or_many(
        sentence: Union[str, List[str]],
        lang_input_obj:Language,
        lang_output_obj:Language,
        model:BaseModel
) -> Union[str, List[str]]:
    if isinstance(sentence, str):
        return predict_one(sentence, lang_input_obj, lang_output_obj, model)
    elif isinstance(sentence, list):
        assert all(isinstance(s, str) for s in sentence), "All elements in the list must be strings"
        return [
            predict_one(s, lang_input_obj, lang_output_obj, model)
            for s in sentence
        ]



def predict_one(
        sentence: str,
        lang_input_obj:Language,
        lang_output_obj:Language,
        model:BaseModel
) -> str:
    try:
        token_ids = (
                [lang_input_obj.SOS_ID]
                + [
                    lang_input_obj.tindex(token)
                    for token in lang_input_obj.sent_iter(sentence)
                ]
                + [lang_input_obj.EOS_ID]
        )
    except Exception as e:
        print(f"Error processing sentence: {e}")
        return None

    pred_tokens = model.predict(token_ids, lang_output=lang_output_obj)
    if not isinstance(pred_tokens, list):
        pred_tokens = list(pred_tokens)
    sep = lang_output_obj.sep if lang_output_obj.sep else " | "
    if isinstance(sep, Iterable):
        sep = sep[0]  # type: ignore
    return sep.join(pred_tokens)


def main(
        do_train: bool = False,
        num_epochs: int = 10,
        batch_size: Optional[int] = None,
        min_batch_size: Optional[int] = None,
        max_batch_size: Optional[int] = None,

        lang_input: str = "",
        lang_name: str = "",
        from_lang: Optional[str] = None,
        make_lang: bool = False,
        overwrite_lang: bool = False,
        single_lang: bool = False,
        max_length: int = 1000,
        l1_sep: Optional[str] = None,
        l2_sep: Optional[str] = None,
        l1_extra_vocab: Optional[List[str]] = None,
        l2_extra_vocab: Optional[List[str]] = None,

        full_eval: bool = False,
        nb_predictions: int = 10,

        model_class: str = "S2SNoAttn",
        model_args: dict = None,

        datetime_str: str = None,
        default_to_latest: bool = True,
        with_profiler: bool = False,

        get_partial_forward: bool = False,

        user_inputs: Optional[List[str]] = None,
        user_df: Optional[pl.DataFrame] = None,
        user_df_input_col: Optional[str] = None,
        output_path: Optional[Path] = None,
) -> Optional[List[Dict[str, Any]]]:
    train_func = profiler_wrapper(auto_train, profile_=with_profiler)
    full_eval_func = profiler_wrapper(do_full_eval, profile_=with_profiler)
    random_eval_func = profiler_wrapper(random_predict, profile_=with_profiler)

    assert lang_name, "lang_name must be provided"

    if model_args is None:
        model_args = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_args["lang_name"] = lang_name
    model_args["device"] = device

    model_class_obj = models[model_class]
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
    ) = model_class_obj.solve_paths()

    if make_lang:
        assert lang_input, "lang_input must be provided when make_lang is True"
        lang_input_path = Path(lang_input)
        assert lang_input_path.exists(), f"lang_input {lang_input_path} does not exist"

        # Préparer le chemin pour from_lang s'il est spécifié
        from_lang_path = None
        if from_lang:
            from_lang_path = lang_root / from_lang / "lang.json"
            assert from_lang_path.exists(), f"from_lang path {from_lang_path} does not exist"
            print(f"Extending language from {from_lang_path}")

        if single_lang:
            extra_vocab = l1_extra_vocab if l1_extra_vocab else None
            X, l1 = Language.read_data_from_json_1_lang(
                lang_input_path,
                max_length=max_length,
                l_sep=l1_sep,
                from_lang=from_lang_path,
                extra_vocab=extra_vocab,
            )
            y, l2 = X, l1  # single langue = auto-encode
            Language.save_data_1_lang(X, l1, lang_path=lang_root / lang_name, overwrite=overwrite_lang)
        elif lang_input_path.suffix == ".json":
            extra_vocab_tuple = (l1_extra_vocab or [], l2_extra_vocab or [])
            X, y, l1, l2 = Language.read_data_from_json(
                lang_input_path,
                max_length=max_length,
                l1_sep=l1_sep,
                l2_sep=l2_sep,
                from_lang=from_lang_path,
                extra_vocab=extra_vocab_tuple,
            )
            Language.save_data(X, y, l1, l2, lang_path=lang_root / lang_name, overwrite=overwrite_lang)
        else:
            extra_vocab_tuple = (l1_extra_vocab or [], l2_extra_vocab or [])
            X, y, l1, l2 = Language.read_data_from_txt(
                lang_input_path,
                max_length=max_length,
                l1_sep=l1_sep,
                l2_sep=l2_sep,
                from_lang=from_lang_path,
                extra_vocab=extra_vocab_tuple,
            )
            Language.save_data(X, y, l1, l2, lang_path=lang_root / lang_name, overwrite=overwrite_lang)

    inference_mode = (user_inputs is not None) or (user_df is not None)

    if do_train:
        assert not inference_mode, "Inference mode et training are mutually exclusive"
        model_args["pretrained"] = False
        model, lang_input_obj, lang_output_obj, losses, evals, (X_train, X_dev, X_test, y_train, y_dev, y_test) = \
            train_func(
                model_class=model_class_obj,
                model_args=model_args,
                num_epochs=num_epochs,
                lang_dir=lang_root / lang_name,
                batch_size=batch_size,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                single_lang=single_lang,
            )
        model.save()
    else:
        model, state, model_dir = model_class_obj.load(
            lang_name=lang_name,
            datetime_str=datetime_str,
            default_to_latest=default_to_latest,
            device=device
        )
        if single_lang:
            X_train, X_dev, X_test, lang_input_obj = read_data_1_lang(lang_root / lang_name)
            y_train, y_dev, y_test, lang_output_obj = X_train, X_dev, X_test, lang_input_obj
        else:
            X_train, X_dev, X_test, y_train, y_dev, y_test, lang_input_obj, lang_output_obj = \
                read_data(lang_path=lang_root / lang_name)
        print("Model, data, and parameters loaded successfully")

    if inference_mode:
        if user_df is not None:
            df_input = user_df
        else:
            df_input = pl.DataFrame({"input": user_inputs})

        assert user_df_input_col in df_input.columns, (
            "The DataFrame must contain the input column "
            f"`{user_df_input_col}` (default: `input`)"
        )
        raw_inputs = df_input[user_df_input_col].to_list()
        print(f"Running in inference mode on {len(raw_inputs)} inputs")


        outputs = [
            predict_one_or_many(s, lang_input_obj, lang_output_obj, model)
            for s in raw_inputs
        ]

        df_output = df_input.with_columns(
            pl.Series("output", outputs)
        )

        # if output_path is not None:
        #     out_path = Path(output_path)
        #     out_path.parent.mkdir(parents=True, exist_ok=True)
        #     df_output.write_json(out_path)
        #     print(f"Résultats écrits dans {out_path}")
        # else:
        #     print(df_output)
        return df_output

    if full_eval:
        full_eval_func(X_dev, y_dev, lang_input_obj, lang_output_obj, model, batch_size)
    else:
        random_eval_func(X_dev, y_dev, lang_input_obj, lang_output_obj, model, batch_size, nb_predictions=nb_predictions)

    if get_partial_forward:
        get_partial_output(
            model,
            np.concatenate((X_train, X_dev)),
            np.concatenate((y_train, y_dev)),
            batch_size
        )

    return None
