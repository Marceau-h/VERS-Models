# SPDX-FileCopyrightText: 2028-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import time
from pathlib import Path
from typing import Optional, Union, Type, Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .models import BaseModel
    from .Language import read_data
except ImportError:
    from vers_models.models import BaseModel
    from vers_models.Language import read_data


def expand_model_vocabulary(model, new_src_vocab_size, new_trg_vocab_size, device=None):
    """Expand model embedding layers to accommodate larger vocabularies."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get original embedding dimensions
    old_src_vocab_size = model.encoder_embedding.num_embeddings
    old_trg_vocab_size = model.decoder_embedding.num_embeddings
    embed_dim = model.encoder_embedding.embedding_dim

    # Create new embeddings with expanded size
    new_encoder_embed = nn.Embedding(new_src_vocab_size, embed_dim)
    new_decoder_embed = nn.Embedding(new_trg_vocab_size, embed_dim)

    # Initialize with normal distribution or zeros
    nn.init.normal_(new_encoder_embed.weight, mean=0, std=0.1)
    nn.init.normal_(new_decoder_embed.weight, mean=0, std=0.1)

    # Copy original embeddings to new ones
    with torch.no_grad():
        new_encoder_embed.weight[:old_src_vocab_size] = model.encoder_embedding.weight
        new_decoder_embed.weight[:old_trg_vocab_size] = model.decoder_embedding.weight

    # Replace embeddings in the model
    model.encoder_embedding = new_encoder_embed
    model.decoder_embedding = new_decoder_embed

    # If there's an output projection layer that depends on vocab size
    if hasattr(model, 'fc_out') and isinstance(model.fc_out, nn.Linear):
        old_fc = model.fc_out
        new_fc = nn.Linear(old_fc.in_features, new_trg_vocab_size)

        # Copy original weights for existing vocab
        with torch.no_grad():
            new_fc.weight[:old_trg_vocab_size] = old_fc.weight
            new_fc.bias[:old_trg_vocab_size] = old_fc.bias

        model.fc_out = new_fc

    return model.to(device)


def auto_train(
        model_class: type[BaseModel],
        model_args: dict,
        num_epochs: int,
        lang_dir: Union[str, Path],
        batch_size: Optional[int] = None,
        min_batch_size: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        eval_every: Optional[int] = None,
        eval_fn: "function" = None,
        eval_args: dict = None,

):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read data
    X_train, X_dev, X_test, y_train, y_dev, y_test, lang_input, lang_output = read_data(
        lang_dir
    )

    # Model setup
    model_args["input_size"] = lang_input.n_tokens
    model_args["max_input_length"] = lang_input.max_length
    model_args["output_size"] = lang_output.n_tokens
    model_args["max_output_length"] = lang_output.max_length

    model_args["num_epochs"] = num_epochs

    print(f"Model args: {model_args}")
    model = model_class(**model_args).to(device)

    # Training setup
    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if batch_size is None:
        if min_batch_size is None or max_batch_size is None:
            print(
                batch_size,
                min_batch_size,
                max_batch_size,
            )
            raise ValueError("Either batch_size or both min_batch_size and max_batch_size must be provided.")
        batch_size = find_best_batch_size(
            model_class=model_class,
            model_args=model_args,
            lang_dir=lang_dir,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            device=device
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train model
    model, losses, evals = model.do_train(
        device=device,
        dataloader=dataloader,
        num_epochs=num_epochs,
        eval_every=eval_every,
        eval_fn=eval_fn,
        eval_args={
            **(eval_args or {}),
            "lang_input": lang_input,
            "lang_output": lang_output,
            "model": model,
        }
    )

    return model, lang_input, lang_output, losses, evals, (X_train, X_dev, X_test, y_train, y_dev, y_test)

def find_best_batch_size(
    model_class: Type[BaseModel],
    model_args: Dict[str, Any],
    lang_dir: Union[str, Path],
    min_batch_size: int,
    max_batch_size: int,
    device: Optional[Union[str, torch.device]] = None,
    num_warmup: int = 10,
    num_iters: int = 10,
) -> int:
    """
    Binary‐search the largest batch size between [min_batch_size, max_batch_size] that fits
    on `device` and yields the fastest per‐batch forward pass.

    - On each midpoint, we measure average forward‐only time over `num_iters` batches.
    - If it OOMs or is slower than our current best, we move the upper bound down.
    - Otherwise we move the lower bound up and record the new best.

    Returns the best batch size found.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args["raise_twice"] = False

    X_train, X_dev, X_test, y_train, y_dev, y_test, lang_input, lang_output = read_data(lang_dir)
    tensor_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))

    def _test(bs: int) -> Optional[float]:
        loader = DataLoader(tensor_data, batch_size=bs, shuffle=False)
        nb_batches = len(loader)
        batch = next(iter(loader))
        src, trg = batch[0].to(device), batch[1].to(device)

        model = model_class(**{**model_args, "batch_size": bs}).to(device)
        model.eval()

        with torch.inference_mode():
            for _ in range(num_warmup):
                _ = model(src, trg)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = time.perf_counter()
        # with torch.no_grad():
            # for _ in range(num_iters):
            #     _ = model(src, trg)
        # If no grad we underestimate the memory usage
        for (src, trg), _ in zip(loader, range(num_iters)):
            src = src.to(device)
            trg = trg.to(device)
            _ = model(src, trg)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iters
        return elapsed * nb_batches

    best_time = None
    low, high = min_batch_size, max_batch_size

    # ensure min_bs fits
    try:
        t0 = _test(low)
        if t0 is None:
            raise RuntimeError(f"Minimum batch size {min_batch_size} OOMs immediately")
        best_time = t0
    except RuntimeError as e:
        if "out of memory" in str(e):
            raise RuntimeError(f"Even the minimum batch size {min_batch_size} OOMs")
        else:
            raise

    # binary search
    while high - low > 1:
        mid = (low + high) // 2
        try:
            t_mid = _test(mid)
        except RuntimeError as e:
            if "out of memory" in str(e):
                t_mid = None
            else:
                raise

        if t_mid is None or t_mid > best_time:
            # too big or slower
            high = mid
        else:
            # fits and faster
            low = mid
            best_time = t_mid
    print(f"Best batch size: {low} ({best_time:.2f}s)")
    return low
