# SPDX-FileCopyrightText: 2025-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import math
from typing import Union, Optional, List, Iterable

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

    def partial_forward(self, src:Tensor) -> Tensor:
        """
        Return encoder memory as latent representation for given input sequence.
        """
        self.eval()
        with torch.inference_mode():
            # src: [batch, seq_len]
            src_mask = self.make_src_key_padding_mask(src)
            embed_src = self.pos_encoder(self.src_tok_embed(src))
            memory = self.transformer.encoder(embed_src, src_key_padding_mask=src_mask)
        return memory

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

    def _predict_single(self, src: Tensor, lang_output: Language) -> Iterable[str]:
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

    def _predict_batch(self, src: Union[list, ndarray, Tensor], lang_output: Language) -> List[List[str]]:
        """Optimized batch prediction processing all sequences simultaneously."""
        src = self._process_batch_input(src)
        batch_size = src.size(0)

        with torch.inference_mode():
            src_mask = self.make_src_key_padding_mask(src)
            embed_src = self.pos_encoder(self.src_tok_embed(src))
            memory = self.transformer.encoder(embed_src, src_key_padding_mask=src_mask)

            batch_outputs = [[lang_output.SOS_ID] for _ in range(batch_size)]

            active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

            for step in range(self.max_output_length):
                if not active_mask.any():
                    break

                max_tgt_len = max(len(seq) for seq in batch_outputs)

                tgt_batch = torch.full(
                    (batch_size, max_tgt_len),
                    PAD_ID,
                    dtype=torch.long,
                    device=self.device
                )

                for i, seq in enumerate(batch_outputs):
                    tgt_batch[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)

                tgt_mask = self.make_tgt_mask(tgt_batch)
                tgt_key_padding_mask = self.make_src_key_padding_mask(tgt_batch)

                embed_tgt = self.pos_encoder(self.tgt_tok_embed(tgt_batch))

                decoder_output = self.transformer.decoder(
                    embed_tgt,
                    memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_mask
                )

                logits = self.fc_out(decoder_output)  # [batch_size, max_tgt_len, vocab_size]

                next_tokens = []
                for i in range(batch_size):
                    if active_mask[i]:
                        last_pos = len(batch_outputs[i]) - 1
                        next_token = logits[i, last_pos].argmax().item()
                        next_tokens.append(next_token)
                        batch_outputs[i].append(next_token)

                        if next_token == lang_output.EOS_ID:
                            active_mask[i] = False

            all_results = []
            for sequence_tokens in batch_outputs:
                result = [lang_output.index2token[idx] for idx in sequence_tokens]
                all_results.append(result)

            return all_results

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

    def finetune(
            self,
            new_input_lang: Language,
            new_output_lang: Language,
            preserve_weights: bool = True,
            init_std: float = 0.1
    ):
        """
        Fine-tune the model for new languages by adjusting layer sizes and preserving weights.
        All tokens from the original language will be preserved with their original IDs.
        """
        if not preserve_weights:
            # If not preserving weights, just update sizes and reinitialize
            self.input_size = new_input_lang.n_tokens
            self.output_size = new_output_lang.n_tokens
            self.params["input_size"] = self.input_size
            self.params["output_size"] = self.output_size
            
            # Reinitialize layers with new sizes
            self.src_tok_embed = nn.Embedding(self.input_size, self.embed_size).to(self.device)
            self.tgt_tok_embed = nn.Embedding(self.output_size, self.embed_size).to(self.device)
            self.fc_out = nn.Linear(self.embed_size, self.output_size).to(self.device)
            return self
        
        # Merge vocabularies to preserve old token IDs
        merged_input_lang = new_input_lang
        merged_output_lang = new_output_lang
        
        if hasattr(self, '_current_input_lang') and self._current_input_lang is not None:
            merged_input_lang = self._merge_vocabularies(self._current_input_lang, new_input_lang)
        
        if hasattr(self, '_current_output_lang') and self._current_output_lang is not None:
            merged_output_lang = self._merge_vocabularies(self._current_output_lang, new_output_lang)
        
        # Create vocabulary mappings with merged vocabularies
        if hasattr(self, '_current_input_lang') and self._current_input_lang is not None:
            input_mapping = self._create_vocab_mapping(self._current_input_lang, merged_input_lang)
        else:
            # Create identity mapping for indices that exist in both vocabularies
            min_size = min(self.input_size, merged_input_lang.n_tokens)
            input_mapping = {i: i for i in range(min_size)}
        
        if hasattr(self, '_current_output_lang') and self._current_output_lang is not None:
            output_mapping = self._create_vocab_mapping(self._current_output_lang, merged_output_lang)
        else:
            # Create identity mapping for indices that exist in both vocabularies
            min_size = min(self.output_size, merged_output_lang.n_tokens)
            output_mapping = {i: i for i in range(min_size)}
        
        # Get current and new sizes
        old_input_size = self.input_size
        old_output_size = self.output_size
        new_input_size = merged_input_lang.n_tokens
        new_output_size = merged_output_lang.n_tokens
        
        # Only resize if new vocabulary is larger
        if new_input_size > old_input_size:
            self.src_tok_embed = self._resize_embedding_layer(
                self.src_tok_embed, new_input_size, input_mapping, init_std
            )
            self.input_size = new_input_size
            self.params["input_size"] = self.input_size
        
        if new_output_size > old_output_size:
            self.tgt_tok_embed = self._resize_embedding_layer(
                self.tgt_tok_embed, new_output_size, output_mapping, init_std
            )
            
            self.fc_out = self._resize_linear_layer(
                self.fc_out, new_output_size, output_mapping, init_std
            )
            self.output_size = new_output_size
            self.params["output_size"] = self.output_size
        
        # Update current language references with merged languages
        self.set_current_languages(merged_input_lang, merged_output_lang)
        
        return self

