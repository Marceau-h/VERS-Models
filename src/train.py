from pathlib import Path
from typing import Optional

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm.auto import trange

from src import BaseModel
from src.Language import read_data
from src.models import models


# def train(
#         model,
#         dataloader,
#         optimizer,
#         criterion,
#         device,
#         num_epochs=10,
#         teacher_forcing_ratio=0.5,
#         eval_every=None,
#         eval_fn=None,
#         eval_args=None,
#         from_epoch=0,
# ):
#     scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
#     model.train()
#
#     losses = []
#     evals = []
#
#     pbar = trange(1 + from_epoch, num_epochs + 1 + from_epoch, desc="Epochs", unit="epoch")
#     for epoch in pbar:
#         epoch_loss = 0
#
#         for src, trg in dataloader:
#             src, trg = src.to(device), trg.to(device)
#
#             optimizer.zero_grad()
#             if scaler is not None:
#                 with torch.amp.autocast("cuda"):
#                     output = model(src, trg, teacher_forcing_ratio)
#
#                     # Reshape for the loss function
#                     output_dim = output.shape[-1]
#                     output = output[:, 1:].reshape(-1, output_dim)
#                     trg = trg[:, 1:].reshape(-1)
#
#                     loss = criterion(output, trg)
#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 output = model(src, trg, teacher_forcing_ratio)
#
#                 # Reshape for the loss function
#                 output_dim = output.shape[-1]
#                 output = output[:, 1:].reshape(-1, output_dim)
#                 trg = trg[:, 1:].reshape(-1)
#
#                 loss = criterion(output, trg)
#                 loss.backward()
#                 optimizer.step()
#
#             epoch_loss += loss.item()
#
#         # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
#         pbar.set_postfix(loss=epoch_loss / len(dataloader))
#
#         if eval_every and eval_fn:
#             if epoch % eval_every == 0:
#                 losses.append(epoch_loss / len(dataloader))
#                 evals.append(eval_fn(**eval_args))
#                 model.train()
#
#     if not eval_every:
#         losses.append(epoch_loss / len(dataloader))
#     elif epoch % eval_every != 0:
#         losses.append(epoch_loss / len(dataloader))
#         if eval_fn:
#             evals.append(eval_fn(**eval_args))
#
#     return model, losses, evals


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
        batch_size: int,
        num_epochs: int,
        lang_dir: str,
        eval_every: Optional[int] = None,
        eval_fn: "function" = None,
        eval_args: dict = None,

):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read data
    X_train, X_test, y_train, y_test, lang_input, lang_output = read_data(
        lang_dir
    )

    # Model setup
    model_args["input_size"] = lang_input.n_tokens
    model_args["output_size"] = lang_output.n_tokens

    model = model_class(**model_args).to(device)

    # Training setup

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
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

    return model, lang_input, lang_output, losses, evals, (X_train, X_test, y_train, y_test)
