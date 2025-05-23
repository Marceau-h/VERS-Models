# SPDX-FileCopyrightText: 2025-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import re
from random import sample
from statistics import mean
from typing import List, Optional, Tuple

from jiwer import process_words, visualize_alignment
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import cowsay
except ImportError:
    cowsay = None

multi_stars = re.compile(r"\*+")


def dont(x):
    """
    Just don't (identity function)
    """
    return x


def align_words(ref, hyp):
    if ref == hyp:
        return None, 0

    computed = process_words([ref], [hyp], dont, dont)
    alignement = visualize_alignment(computed, show_measures=False)
    _, ref, hyp, *_ = multi_stars.sub("@", alignement).split("\n")
    alignement = f"{ref[5:]}\n{hyp[5:]}".replace(" ", " | ")

    return alignement, computed.wer


def predict(model, input_sentence, lang_input, lang_output):
    if torch.is_tensor(input_sentence) and input_sentence.dim() > 1:
        batch = input_sentence.to(model.device)
        input_sentence_lst = lang_input.index2token_sent(batch)
        # Predict each sample individually
        predicted_output_lst = [model.predict(row, lang_output=lang_output) for row in batch]
        return input_sentence_lst, predicted_output_lst

    if isinstance(input_sentence, list) and input_sentence and isinstance(input_sentence[0], (list, tuple)):
        batch = torch.tensor(input_sentence, dtype=torch.long, device=model.device)
        input_sentence_lst = lang_input.index2token_sent(batch)
        predicted_output_lst = [model.predict(list(seq), lang_output=lang_output) for seq in input_sentence]
        return input_sentence_lst, predicted_output_lst

    # Single sequence
    input_sentence_lst = [
        lang_input.index2token[token]
        for token in input_sentence
        if token != lang_input.PAD_ID
    ]
    predicted_output_lst = model.predict(input_sentence, lang_output=lang_output)
    return input_sentence_lst, predicted_output_lst


def do_one_sent(model, sentence, lang_input, lang_output):
    input_sentence = (
            [lang_input.SOS_ID]
            + [lang_input.token2index[token] for token in lang_input.sent_iter(sentence)]
            + [lang_input.EOS_ID]
    )

    input_sentence_lst, predicted_output_lst = predict(model, input_sentence, lang_input, lang_output)

    line1 = f"Input sentence: {sentence}"
    line2 = f"Predicted output: {' | '.join(predicted_output_lst)}"
    max_len = max(len(line1), len(line2))
    txt = f"{line1:^{max_len}}\n{line2:^{max_len}}"

    if cowsay is not None:
        cowsay.tux(txt)
    else:
        print("\n\t\t------------------\t\t\n\n" + txt + "\n\n\t\t------------------\t\t\n")

    return input_sentence_lst, predicted_output_lst



def eval_numbers(
        target: List[str],
        predicted: List[str],
        lang_output: "Language",
) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[bool], Optional[bool]]:
    lang_spesific_tokens = {lang_output.SOS_TOKEN, lang_output.EOS_TOKEN, lang_output.PAD_TOKEN}
    try:
        target = [
            int(token)
            for token in target[1:-1]
            if token not in lang_spesific_tokens
        ]
        predicted = [
            int(token)
            for token in predicted[1:-1]
            if token not in lang_spesific_tokens
        ]
    except ValueError:
        # raise
        return None, None, None, None, None

    if target == predicted:
        return 0, 0, 0, True, True

    sum_diff = sum(target) - sum(predicted)

    len_diff = len(target) - len(predicted)

    mean_diff = sum_diff / len(target)

    diffs = [t - p for t, p in zip(target, predicted)]
    same_diff = all(d == diffs[0] for d in diffs)

    same_diff_same_len = same_diff and not len_diff

    return sum_diff, len_diff, mean_diff, same_diff, same_diff_same_len



def core_eval(X_test, y_test, lang_input, lang_output, model, batch_size, nb_predictions=None, do_print=True):
    print(f"Evaluating.. {len(X_test) = }, {len(y_test) = }")
    # Prepare dataset and dataloader for evaluation
    dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
    # Subset for nb_predictions if specified
    if nb_predictions is None:
        eval_dataset = dataset
    elif isinstance(nb_predictions, int) and nb_predictions > 0:
        indices = sample(list(range(len(X_test))), min(nb_predictions, len(X_test)))
        from torch.utils.data import Subset
        eval_dataset = Subset(dataset, indices)
    else:
        raise ValueError("nb_predictions must be a positive integer or None")
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    pbar = tqdm(dataloader, desc="Evaluating", unit="batch") if do_print else dataloader

    model.eval()
    res = []
    with torch.no_grad():
        for input_tensor, target_tensor in pbar:
            # Batch to device
            input_tensor = input_tensor.to(model.device, non_blocking=True)
            target_tensor = target_tensor.to(model.device, non_blocking=True)
            # Predict batch
            input_lsts, pred_lsts = predict(model, input_tensor, lang_input, lang_output)
            # Convert targets
            target_lsts = lang_output.index2token_sent(target_tensor)
            # For each sample in batch
            for inp, tgt, pred in zip(input_lsts, target_lsts, pred_lsts):
                exact_match = tgt == pred
                sum_diff, len_diff, mean_diff, same_diff, same_diff_same_len = eval_numbers(tgt, pred, lang_output)
                aligned, wer = align_words(tgt, pred)
                res.append((inp, tgt, pred, aligned, exact_match, wer, sum_diff, len_diff, mean_diff, same_diff, same_diff_same_len))
    return res


def random_predict(X_test, y_test, lang_input, lang_output, model, batch_size, print_output=True, nb_predictions=10):
    # Evaluate random samples
    res = core_eval(X_test, y_test, lang_input, lang_output, model, batch_size, nb_predictions=nb_predictions, do_print=print_output)

    input_joiner = " " if lang_input.re_sep is not None else "" if lang_input.sep is None else lang_input.sep

    if print_output:
        for (
                input_sentence_lst,
                target_output_lst,
                predicted_output_lst,
                aligned,
                exact_match,
                wer,
                sum_diff,
                len_diff,
                mean_diff,
                same_diff,
                same_diff_same_len,
        ) in res:
            # mean_diff = mean_diff if mean_diff is not None else "N/A"
            print(f"""
Lengths (\\wo EOS) - Input: {len(input_sentence_lst) - 6}, Target: {len(target_output_lst) - 2}, Predicted: {len(predicted_output_lst) - 2}

Input sentence: {input_joiner.join(input_sentence_lst)}
Target output: {" | ".join(target_output_lst)}
Predicted output: {" | ".join(predicted_output_lst)}

Alignment:\n{aligned}

Exact match: {exact_match}
WER: {wer:.2f}
Sum diff: {sum_diff}
Len diff: {len_diff}
Mean diff: {mean_diff}
Same diff: {same_diff}
Same diff same len: {same_diff_same_len}
""")
        if nb_predictions > 1:
            print(f"Mean exact match ratio: {mean(r[4] for r in res):.3f}")
            print(f"Mean WER: {mean(r[5] for r in res):.3f}")
            if all(r[6] is not None for r in res):
                assert not any(r[6] is None for r in res)

                print(f"Mean sum diff: {mean(abs(r[6]) for r in res):.3f} (abs)")
                print(f"Mean len diff: {mean(abs(r[7]) for r in res):.3f} (abs)")
                print(f"Mean mean diff: {mean(abs(r[8]) for r in res):.3f} (abs)")
                print(f"Same diff ratio: {mean(r[9] for r in res):.3f}")
                print(f"Same diff same len ratio: {mean(r[10] for r in res):.3f}")


    return res


def evaluate(X_test, y_test, lang_input, lang_output, model, batch_size, do_print=True):
    num_mode = False

    res = core_eval(X_test, y_test, lang_input, lang_output, model, batch_size, nb_predictions=None, do_print=do_print)

    exact_match = mean(r[4] for r in res)
    wer_score = mean(r[5] for r in res)
    if all(r[6] is not None for r in res):
        num_mode = True
        sum_diff = mean(r[6] for r in res)
        len_diff = mean(r[7] for r in res)
        mean_diff = mean(r[8] for r in res)
        same_diff = mean(r[9] for r in res)
        same_diff_same_len = mean(r[10] for r in res)
    else:
        sum_diff = len_diff = mean_diff = same_diff = same_diff_same_len = None


    if do_print:
        print(
            f"Exact match ratio: {exact_match:.3f}\n"
            f"Mean WER: {wer_score:.3f}"
        )
        if num_mode:
            print(
                f"Mean sum diff: {sum_diff:.3f} (abs)\n"
                f"Mean len diff: {len_diff:.3f} (abs)\n"
                f"Mean mean diff: {mean_diff:.3f} (abs)\n"
                f"Same diff ratio: {same_diff:.3f}\n"
                f"Same diff same len ratio: {same_diff_same_len:.3f}"
            )


    res_for_save = [
        {
            "input": "".join(r[0]),
            "target": " | ".join(r[1]),
            "predicted": " | ".join(r[2]),
            "alignment": r[3],
            "target_length": len(r[1]) - 2,
            "predicted_length": len(r[2]) - 2,
            "exact_match": r[4],
            "wer": r[5],
            "sum_diff": r[6],
            "len_diff": r[7],
            "mean_diff": r[8],
            "same_diff": r[9],
            "same_diff_same_len": r[10],
        }
        for r in res
    ]

    return res, exact_match, wer_score, res_for_save


def do_full_eval(X_test, y_test, lang_input, lang_output, model, batch_size):
    import polars as pl

    res, accuracy, wer_score, res_for_save = evaluate(X_test, y_test, lang_input, lang_output, model, batch_size)

    df = pl.DataFrame(res_for_save, infer_schema_length=10_000_000)

    df = df[[s.name for s in df if not (s.null_count() == df.height)]]  # remove columns with only null values (numbers cols if there arent any)

    df.write_csv(model.eval_path.with_suffix(".csv"))
    df.write_ndjson(model.eval_path.with_suffix(".ndjson"))
    df.write_parquet(model.eval_path.with_suffix(".parquet"))
    df.write_json(model.eval_path.with_suffix(".json"))

    return df
