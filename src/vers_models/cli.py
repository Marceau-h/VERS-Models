# SPDX-FileCopyrightText: 2028-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from time import perf_counter_ns as ns
from argparse import ArgumentParser


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


def main(*args, **kwargs):
    """
    Delays the import of the main function to validate the arguments first without wasting time on imports.
    """
    try:
        from .main import main
    except ImportError:
        from vers_models.main import main
    return main(*args, **kwargs)


def cli():
    parser = ArgumentParser()

    parser.add_argument(
        "--train", action="store_true",
        help="Train the model"
    )
    parser.add_argument(
        "--num_epochs", type=int,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int,
        help="Batch size for training, if specified would not search for the best batch size"
    )
    parser.add_argument(
        "--min_batch_size", type=int,
        help="Minimum batch size, if specified with max_batch_size would search for the best batch size"
    )
    parser.add_argument(
        "--max_batch_size", type=int,
        help="Maximum batch size, if specified with min_batch_size would search for the best batch size"
    )

    parser.add_argument(
        "--lang_input", type=str, default="",
        help="Path to the input language data"
    )
    parser.add_argument(
        "--lang_name", type=str, required=True,
        help="Name of the language data"
    )
    parser.add_argument(
        "--make_lang", action="store_true",
        help="Make language data"
    )
    parser.add_argument(
        "--overwrite_lang", action="store_true",
        help="Overwrite existing language data if it exists"
    )

    parser.add_argument(
        "--full_eval", action="store_true",
        help="Run full evaluation"
    )
    parser.add_argument(
        "--nb_predictions", type=int, default=10,
        help="Number of predictions to make"
    )

    parser.add_argument(
        "--model_class", type=str, required=True,
        help="Model class to use"
    )

    parser.add_argument(
        "--datetime_str", type=str, default=None,
        help="Datetime string for loading the model"
    )
    parser.add_argument(
        "--default_to_latest", action="store_false",
        help="Use the latest model if datetime_str is not provided"
    )



    parsed, unknown = parser.parse_known_args()
    print("Parsed arguments:", parsed)
    print("Unknown arguments:", unknown)

    if parsed.train:
        assert parsed.num_epochs is not None, "num_epochs must be specified when training"
        # assert parsed.batch_size is not None, "batch_size must be specified when training"
        if parsed.batch_size is None:
            assert parsed.min_batch_size is not None, "min_batch_size must be specified when training if batch_size is not specified"
            assert parsed.max_batch_size is not None, "max_batch_size must be specified when training if batch_size is not specified"

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

    start_time = ns()
    main(
        do_train=parsed.train,
        num_epochs=parsed.num_epochs,
        batch_size=parsed.batch_size,
        min_batch_size=parsed.min_batch_size,
        max_batch_size=parsed.max_batch_size,
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

if __name__ == "__main__":
    cli()
