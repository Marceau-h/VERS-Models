from pathlib import Path
from time import perf_counter_ns as ns
from argparse import ArgumentParser

import torch

from src.Language import Language, read_data
from src.eval import random_predict, do_full_eval
from src.models import models
from src.train import auto_train


def main(
        do_train: bool = False,
        num_epochs: int = 10,
        batch_size: int = 2048,

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
            (X_train, X_test, y_train, y_test),
        ) = auto_train(
            model_class=model_class,
            model_args=model_args,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lang_dir=lang_root / lang_name,
        )

        model.save()

    else:
        model, state, old_vocab_size = model_class.load(datetime_str, default_to_latest, device)

        X_train, X_test, y_train, y_test, lang_input, lang_output = read_data(lang_path=lang_root / lang_name)
        print("Model, data, and parameters loaded successfully")

    # Test prediction
    random_predict(X_test, y_test, lang_input, lang_output, model, device=device, nb_predictions=nb_predictions)

    if full_eval:
        do_full_eval(X_test, y_test, lang_input, lang_output, model, device=device)


# def load_and_do_one_sent(sentence, pho, suffix):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     params_path, model_path, og_lang_path, x_data, y_data, lang_path, eval_path = paths(pho, suffix)
#     model, state, old_vocab_size = load_model(params_path, model_path, device)
#     X_train, X_test, y_train, y_test, lang_input, lang_output = read_data(lang_path=lang_root / lang_name)
#     do_one_sent(model, sentence, lang_input, lang_output, device)


def pretty_time(ns: int) -> str:
    """
    Convert nanoseconds to a pretty string representation of time
    (hours, minutes, seconds, milliseconds)
    :param ns: The time in nanoseconds
    :return: The pretty string representation of time of the form "Xh Ym Zs Tms"
    """
    ns = ns // 1_000_000
    ms = ns % 1_000
    ns //= 1_000
    s = ns % 60
    ns //= 60
    m = ns % 60
    ns //= 60
    h = ns
    return f"{h}h {m}m {s}s {ms}ms"


if __name__ == '__main__':
    # parser = ArgumentParser()
    #
    # parser.add_argument("sentence", type=str, help="Sentence to predict (will bypass all other arguments)", nargs="?", default=None)
    #
    # parser.add_argument("--train", action="store_true", help="Train the model")
    # parser.add_argument("--pho", action="store_true", help="Use phonetic data")
    # parser.add_argument("--make_lang", action="store_true", help="Make language data")
    # parser.add_argument("--full_eval", action="store_true", help="Run full evaluation")
    # parser.add_argument("--suffix", type=str, default="", help="Suffix for file names (overrides `--pho`)")
    #
    # parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    # parser.add_argument("--embed_size", type=int, default=512, help="Embedding size")
    # parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size")
    # parser.add_argument("--num_layers", type=int, default=1, help="Number of layers")
    # parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    # parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    # parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5, help="Teacher forcing ratio")
    # parser.add_argument("--nb_predictions", type=int, default=10, help="Number of predictions to make")
    #
    # parser.add_argument("--fine_tune_from", type=str, default=None, help="Suffix to fine-tune from")
    #
    # args = parser.parse_args()
    #
    # # if args.sentence:
    # #     load_and_do_one_sent(args.sentence, pho=args.pho, suffix=args.suffix)
    # #     exit(0)
    #
    # start_time = ns()
    # # main(
    # #     do_train=args.train,
    # #     pho=args.pho,
    # #     suffix=args.suffix,
    # #     make_lang=args.make_lang,
    # #     full_eval=args.full_eval,
    # #     num_epochs=args.num_epochs,
    # #     embed_size=args.embed_size,
    # #     hidden_size=args.hidden_size,
    # #     num_layers=args.num_layers,
    # #     lr=args.lr,
    # #     batch_size=args.batch_size,
    # #     teacher_forcing_ratio=args.teacher_forcing_ratio,
    # #     nb_predictions=args.nb_predictions,
    # #     fine_tune_from=args.fine_tune_from
    # # )
    #
    # model_args = {
    #     "embed_size": args.embed_size,
    #     "hidden_size": args.hidden_size,
    #     "num_layers": args.num_layers,
    #     "lr": args.lr,
    #     "teacher_forcing_ratio": args.teacher_forcing_ratio
    # }
    #
    # model_args = {k: v for k, v in model_args.items() if v is not None}
    #
    # main(
    #     do_train=args.train,
    #     lang_input="data/phonetic_data.json",
    #     lang_name="phonetic_data",
    #     make_lang=args.make_lang,
    #     full_eval=args.full_eval,
    #     nb_predictions=args.nb_predictions,
    #     model_class="S2SNoAttn",
    #     model_args=model_args,
    #     datetime_str=None,
    #     default_to_latest=True,
    #     batch_size=args.batch_size,
    #     num_epochs=args.num_epochs,
    # )
    #
    # print(f"Done ! Took {pretty_time(ns() - start_time)}")

    parser = ArgumentParser()

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")

    parser.add_argument("--lang_input", type=str, default="", help="Path to the input language data")
    parser.add_argument("--lang_name", type=str, default="", help="Name of the language data")
    parser.add_argument("--make_lang", action="store_true", help="Make language data")
    parser.add_argument("--overwrite_lang", action="store_true", help="Overwrite existing language data if it exists")

    parser.add_argument("--full_eval", action="store_true", help="Run full evaluation")
    parser.add_argument("--nb_predictions", type=int, default=10, help="Number of predictions to make")

    parser.add_argument("--model_class", type=str, default="no_attn", help="Model class to use")

    parser.add_argument("--datetime_str", type=str, default=None, help="Datetime string for loading the model")
    parser.add_argument("--default_to_latest", action="store_true",
                        help="Use the latest model if datetime_str is not provided")

    parsed, unknown = parser.parse_known_args()
    print("Parsed arguments:", parsed)
    print("Unknown arguments:", unknown)
    start_time = ns()

    model_args = {}
    for arg in unknown:
        if arg.startswith("--"):
            key, value = arg[2:].split("=")
            model_args[key] = value

    # Convert numeric arguments to int or float
    for key, value in model_args.items():
        if value.isdigit():
            model_args[key] = int(value)
        else:
            try:
                model_args[key] = float(value)
            except ValueError:
                pass  # Keep it as a string if it can't be converted

    print("Model arguments:", model_args)
    print("Parsed arguments:", parsed)

    # Call the main function with the parsed arguments
    main(
        do_train=parsed.train,
        num_epochs=parsed.num_epochs,
        batch_size=parsed.batch_size,
        lang_input=parsed.lang_input,
        lang_name=parsed.lang_name,
        make_lang=parsed.make_lang,
        overwrite_lang=parsed.overwrite_lang,
        full_eval=parsed.full_eval,
        nb_predictions=parsed.nb_predictions,
        model_class=parsed.model_class,
        model_args=model_args,
        datetime_str=parsed.datetime_str,
        default_to_latest=parsed.default_to_latest
    )
    print(f"Done ! Took {pretty_time(ns() - start_time)}")
