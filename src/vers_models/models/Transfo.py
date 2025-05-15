# SPDX-FileCopyrightText: 2028-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import math
from typing import Union, Optional, List

from numpy import ndarray
import torch
from torch import nn, Tensor
from torch.nn import Transformer
from torch.utils.data import DataLoader
from tqdm import trange

try:
    from .BaseModel import BaseModel
    from ..Language import Language, PAD_ID
except ImportError:
    from vers_models.models.BaseModel import BaseModel
    from vers_models.Language import Language, PAD_ID


class PositionalEncoding(nn.Module):
    """From the torch doc"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transfo(BaseModel):
    # def __init__(
    #         self,
    #         input_vocab_size: int,
    #         output_vocab_size: int,
    #         embed_size: int = 512,
    #         num_heads: int = 8,
    #         num_encoder_layers: int = 6,
    #         num_decoder_layers: int = 6,
    #         ff_dim: int = 2048,
    #         dropout: float = 0.1,
    #         max_length: int = 5000
    # ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.embed_size = self.params["embed_size"]
        self.input_size = self.params["input_size"]
        self.output_size = self.params["output_size"]
        self.max_input_length = self.params["max_input_length"]
        self.max_output_length = self.params["max_output_length"]
        self.dropout = self.params["dropout"]
        self.num_heads = self.params["num_heads"]
        self.num_encoder_layers = self.params["num_encoder_layers"]
        self.num_decoder_layers = self.params["num_decoder_layers"]
        self.ff_dim = self.params["ff_dim"]
        self.lr = self.params["lr"]

        self.src_tok_embed = nn.Embedding(self.input_size, self.embed_size)
        self.tgt_tok_embed = nn.Embedding(self.output_size, self.embed_size)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.dropout, self.max_input_length)

        self.transformer = Transformer(
            d_model=self.embed_size,
            nhead=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(self.embed_size, self.output_size)
        self.src_pad_idx = PAD_ID

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)


    def make_src_key_padding_mask(self, src: Tensor) -> Tensor:
        return src == self.src_pad_idx

    def make_tgt_mask(self, tgt: Tensor) -> Tensor:
        seq_len = tgt.size(1)
        return self.transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src_mask = self.make_src_key_padding_mask(src)
        tgt_mask = self.make_src_key_padding_mask(tgt)
        subsequent_mask = self.make_tgt_mask(tgt)

        embed_src = self.pos_encoder(self.src_tok_embed(src))
        embed_tgt = self.pos_encoder(self.tgt_tok_embed(tgt))

        out = self.transformer(
            src=embed_src,
            tgt=embed_tgt,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask,
            tgt_mask=subsequent_mask
        )
        return self.fc_out(out)

    def predict(self, src:Union[ndarray, list, Tensor], lang_output:Language) -> List[str]:
        self.eval()
        src = self.to_tensor(src)
        src = src.unsqueeze(0)
        with torch.inference_mode():
            src_mask = self.make_src_key_padding_mask(src)
            embed_src = self.pos_encoder(self.src_tok_embed(src))
            memory = self.transformer.encoder(embed_src, src_key_padding_mask=src_mask)

            outputs = [lang_output.SOS_ID]
            for _ in range(self.max_output_length):
                tgt = torch.tensor(outputs, dtype=torch.long, device=self.device).unsqueeze(0)
                tgt_mask = self.make_tgt_mask(tgt)
                embed_tgt = self.pos_encoder(self.tgt_tok_embed(tgt))
                dec = self.transformer.decoder(
                    embed_tgt,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_mask
                )
                logits = self.fc_out(dec)
                next_token = logits[0, -1].argmax().item()
                outputs.append(next_token)
                if next_token == lang_output.EOS_ID:
                    break

            return [lang_output.index2token[idx] for idx in outputs]


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
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        self.train()

        losses = []
        evals = []

        pbar = trange(1 + from_epoch, num_epochs + 1 + from_epoch, desc="Epochs", unit="epoch")
        for epoch in pbar:
            epoch_loss = 0

            for src, trg in dataloader:
                src, trg = src.to(device), trg.to(device)

                self.optimizer.zero_grad()

                # Forward pass
                with torch.amp.autocast(enabled=scaler is not None, device_type="cuda"):
                    output = self(src, trg)
                    output_dim = output.shape[-1]
                    out = output[:, 1:].reshape(-1, output_dim)
                    target = trg[:, 1:].reshape(-1)
                    loss = self.criterion(out, target)

                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss / len(dataloader))
            pbar.set_postfix({"loss": losses[-1]})

            if eval_every and eval_fn and eval_args:
                if epoch % eval_every == 0:
                    evals.append(eval_fn(self, **eval_args))

        return self, losses, evals

