# SPDX-FileCopyrightText: 2025-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from enum import Enum
from time import perf_counter_ns as ns
from argparse import ArgumentParser, Namespace
import sys
import json
from pathlib import Path
import select
from typing import Optional, List, Dict, Any, Iterable, assert_never, Tuple

import polars as pl

try:
    from .__about__ import __version__
    from .models import models
except ImportError:
    from vers_models.__about__ import __version__
    from vers_models.models import models


class TYPES_N_FUNC(Enum):
    CSV = 'csv'
    TSV = 'tsv'
    JSON = 'json'
    JSONL = 'jsonl'
    PARQUET = 'parquet'
    UNKNOWN = 'unknown'

    @staticmethod
    def from_str(label: str):
        label = label.lower().lstrip('.')
        if label == 'csv':
            return TYPES_N_FUNC.CSV
        if label == 'tsv':
            return TYPES_N_FUNC.TSV
        if label == 'json':
            return TYPES_N_FUNC.JSON
        if label in {'jsonl', 'ndjson'}:
            return TYPES_N_FUNC.JSONL
        if label == 'parquet':
            return TYPES_N_FUNC.PARQUET
        return TYPES_N_FUNC.UNKNOWN

    def as_suffix(self) -> str:
        assert self.value != 'unknown', "UNKNOWN type has no suffix"
        return f".{self.value}"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            other = TYPES_N_FUNC.from_str(other)

        if not isinstance(other, TYPES_N_FUNC):
            return NotImplemented("Can only compare TYPES_N_FUNC with str or TYPES_N_FUNC")

        if self.value == 'unknown' or other.value == 'unknown':
            raise ValueError("Cannot compare UNKNOWN type")

        return self.value == other.value

def read_input(input_path: Path) -> Tuple[pl.DataFrame, TYPES_N_FUNC]:
    type_ = TYPES_N_FUNC.from_str(input_path.suffix)
    if type_ == TYPES_N_FUNC.CSV:
        return pl.read_csv(input_path), type_
    if type_ == TYPES_N_FUNC.TSV:
        return pl.read_csv(input_path, separator='\t'), type_
    if type_ == TYPES_N_FUNC.JSON:
        return pl.read_json(input_path), type_
    if type_ == TYPES_N_FUNC.JSONL:
        return pl.read_ndjson(input_path), type_
    if type_ == TYPES_N_FUNC.PARQUET:
        return pl.read_parquet(input_path), type_
    raise ValueError(f"Unsupported input type: {type_}, cannot read {input_path.suffix} files")

def write_output(
        df: pl.DataFrame,
        output_path: Path,
        type_: TYPES_N_FUNC
) -> None:
    if type_ == TYPES_N_FUNC.CSV:
        df.write_csv(output_path)
    elif type_ == TYPES_N_FUNC.TSV:
        df.write_csv(output_path, separator='\t')
    elif type_ == TYPES_N_FUNC.JSON:
        df.write_json(output_path)
    elif type_ == TYPES_N_FUNC.JSONL:
        df.write_ndjson(output_path)
    elif type_ == TYPES_N_FUNC.PARQUET:
        df.write_parquet(output_path)
    else:
        raise ValueError(f"Unsupported output type: {type_}")

def _prompt_user() -> str:
    """
    Core tty reader
    """
    prompt = "No input provided. Please enter your input (end with an empty line):"
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()

def prompt_user(
        input: Iterable[Optional[str]],
) -> List[str]:
    """
    If
    """
    collected = []
    stdin_consumed = False
    for item in input:
        if item == '-' or item is None:
            if not stdin_consumed:
                if sys.stdin.isatty():
                    stdin_text = ''
                else:
                    r, _, _ = select.select([sys.stdin], [], [], 0)
                    if r:
                        stdin_text = sys.stdin.read()
                    else:
                        stdin_text = ''
                stdin_consumed = True
            else:
                stdin_text = ''

            stdin_text = stdin_text.strip()
            if not stdin_text:
                if sys.stdin.isatty():
                    stdin_text = _prompt_user()
                    if not stdin_text:
                        raise ValueError("No input provided interactively.")
                else:
                    raise ValueError("No input provided from stdin and unable to prompt interactively.")
            collected.extend([l for l in stdin_text.splitlines() if l.strip()])
        else:
            collected.append(item)

    return collected

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

def list_models() -> str:
    """
    List all available models
    :return: A string representation of all available models
    """
    return "Available models :\n\t" + "\n\t".join(
        f"{name} -> {model.__name__};"
        for name, model in models.items()
        if name != "base"
    )


def cli():
    models_str = list_models()

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

    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}",
        help="Show the version of the program"
    )
    parser.add_argument(
        "--list_models", action="version",
        version=models_str,
        help="List all available models"
    )
    parser.add_argument(
        "--with_profiler", action="store_true",
        help="Enable profiling of training and evaluation"
    )
    parser.add_argument(
        "--single_lang", action="store_true",
        help="Use single language mode (autoencoder)"
    )
    parser.add_argument(
        "--get_partial_forward", action="store_true",
        help=
            "Get the partial forward pass of the model as an output (useful for autoencoders or similar models)"
            "\nThe output will be the encoder output of the model."
            "\nThis functionality is not available for autoencoder models"
    )

    parser.add_argument(
        "--input_file", type=str, default=None,
        help="JSON file (list of dicts) with at least one 'input' column for batch inference"
    )
    parser.add_argument(
        "--input_file_col", type=str, default="input",
        help="Column name to use for input from the input_file (default: 'input')"
    )

    parser.add_argument(
        "--input", action='append', nargs='?', const='-',
        help="Direct input (repeatable). Use --input 'text' or --input without value to read from stdin"
             "\n(or prompt for interactive input if empty). Multiple --input allowed."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file (otherwise stdout)"
    )

    parsed, unknown = parser.parse_known_args()
    print("Parsed arguments:", parsed)
    print("Unknown arguments:", unknown)

    if parsed.train:
        assert parsed.num_epochs is not None, "num_epochs must be specified when training"
        if parsed.batch_size is None:
            assert parsed.min_batch_size is not None, "min_batch_size must be specified when training if batch_size is not specified"
            assert parsed.max_batch_size is not None, "max_batch_size must be specified when training if batch_size is not specified"

    inference_requested = parsed.input_file is not None or parsed.input is not None
    if inference_requested and parsed.train:
        raise ValueError("Please choose between training (--train) and inference (--input_file or --input), not both.")

    if inference_requested and parsed.full_eval:
        print("Warning: --full_eval is ignored during inference.")
        parsed.full_eval = False

    model_args = {}
    for arg in unknown:
        if arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            model_args[key] = value
        else:
            raise ValueError(f"Unknown argument format: {arg}.")

    # Convert numeric arguments to int or float
    for key, value in model_args.items():
        if isinstance(value, str) and value.isdigit():
            model_args[key] = int(value)
        else:
            if isinstance(value, str):
                try:
                    model_args[key] = float(value)
                except ValueError:
                    pass  # Keep it as a string if it can't be converted

    print("Model arguments:", model_args)

    # Inference logic
    user_inputs = None
    user_df = None
    if parsed.input_file is not None:
        input_path = Path(parsed.input_file)
        if not input_path.exists():
            raise ValueError(f"Input file {input_path} does not exist")
        user_df, type_ = read_input(input_path)
        print(f"Input file read as type {type_}")
        if not user_df.height:
            raise ValueError("The input file is empty")
        if parsed.input_file_col not in user_df.columns:
            if parsed.input_file_col == 'input':
                raise ValueError(
                    "The input file must contain an `input` column"
                    "\nor you must specify the correct column name with `--input_file_col`"
                )
            else:
                raise ValueError(
                    f"The input file does not contain the specified column '{parsed.input_file_col}'"
                    "\nPlease check the column name and reflect it with `--input_file_col`"
                )
    if parsed.input is not None:
        collected = prompt_user(parsed.input)
        if collected:
            if user_df is not None:
                raise ValueError("Cannot use both --input_file and --input options simultaneously.")
            user_inputs = collected
            user_df = pl.DataFrame({"input": user_inputs})
            user_inputs = None
    output_path = Path(parsed.output) if parsed.output is not None else None

    start_time = ns()
    res = main(
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
        default_to_latest=parsed.default_to_latest,
        with_profiler=parsed.with_profiler,
        single_lang=parsed.single_lang,
        get_partial_forward=parsed.get_partial_forward,
        user_inputs=user_inputs,
        user_df=user_df,
        user_df_input_col=parsed.input_file_col if user_df is not None else None,
        output_path=output_path,
    )
    print(f"Done ! Took {pretty_time(ns() - start_time)}")


    if res is not None:
        if output_path is not None:
            if output_path.is_dir():
                output_path = output_path / f"output{type_.as_suffix()}"
            else:
                output_path = output_path.with_suffix(type_.as_suffix())
            write_output(res, output_path, type_)
            print(f"Output written to {output_path}")
        else:
            print(res)

if __name__ == "__main__":
    cli()
