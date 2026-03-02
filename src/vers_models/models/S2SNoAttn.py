# SPDX-FileCopyrightText: 2025-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import Union, Optional, List, Iterable

from numpy import ndarray
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import trange

try:
    from .BaseModel import BaseModel
    from ..Language import Language
except ImportError:
    from vers_models.models.BaseModel import BaseModel
    from vers_models.Language import Language

class S2SNoAttn(BaseModel):
    # def __init__(self, input_size, output_size, embed_size, hidden_size, num_layers=1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input_size = self.params["input_size"]
        self.output_size = self.params["output_size"]
        self.embed_size = self.params["embed_size"]
        self.hidden_size = self.params["hidden_size"]
        self.num_layers = self.params["num_layers"]
        self.lr = self.params["lr"]
        self.teacher_forcing_ratio = self.params["teacher_forcing_ratio"]
        self.max_input_length =self.params["max_input_length"]
        self.max_output_length = self.params["max_output_length"]

        # Encoder components
        self.encoder_embedding = nn.Embedding(
            self.input_size,
            self.embed_size,  # Fixed: should be embed_size, not output_size
        )
        self.encoder_lstm = nn.LSTM(
            self.embed_size,  # Fixed: should be embed_size to match embedding output
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Decoder components
        self.decoder_embedding = nn.Embedding(
            self.output_size,
            self.embed_size,
        )
        self.decoder_lstm = nn.LSTM(
            self.embed_size,
            self.hidden_size * 2,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(
            self.hidden_size * 2,
            self.output_size,
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def partial_forward(self, src:Tensor) -> Tensor:
        """
        Return the encoder outputs as latent representation for given input sequence.
        """
        self.eval()
        with torch.inference_mode():
            # src: [batch, seq_len]
            embedded_src = self.encoder_embedding(src)
            encoder_outputs, (_hidden, _cell) = self.encoder_lstm(embedded_src)
        return encoder_outputs

    def forward(self, src:Tensor, trg:Tensor) -> Tensor:
        batch_size, trg_len = trg.size()
        trg_vocab_size = self.fc.out_features

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=src.device)

        # Encode the source sequence
        embedded_src = self.encoder_embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)

        # Concatenate the forward and backward hidden states for each layer
        # Bidirectional encoder returns (num_layers * 2, batch, hidden_size)
        # We need (num_layers, batch, hidden_size * 2) for unidirectional decoder
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

        # First input to the decoder is the <sos> token
        input_ = trg[:, 0]

        for t in range(1, trg_len):
            embedded_trg = self.decoder_embedding(input_).unsqueeze(1)

            # Decoder step
            output, (hidden, cell) = self.decoder_lstm(embedded_trg, (hidden, cell))
            prediction = self.fc(output.squeeze(1))
            outputs[:, t, :] = prediction

            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            input_ = trg[:, t] if teacher_force else prediction.argmax(1)

        return outputs

    def _predict_single(self, src: Tensor, lang_output: Language) -> Iterable[str]:
        self.eval()
        src = self.to_tensor(src)

        # Encode the source sequence
        with torch.inference_mode():
            embedded_src = self.encoder_embedding(src.unsqueeze(0))  # Add batch dimension
            encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)

            if len(hidden.shape) != 3:
                raise ValueError("Hidden shape is not 3D")

            # Concatenate forward and backward states for each layer
            # Bidirectional encoder returns (num_layers * 2, batch, hidden_size)
            # We need (num_layers, batch, hidden_size * 2) for unidirectional decoder
            hidden = hidden.view(self.num_layers, 2, 1, self.hidden_size)
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
            cell = cell.view(self.num_layers, 2, 1, self.hidden_size)
            cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

            # Initialize the decoder input with the <sos> token
            input_ = torch.tensor([lang_output.SOS_ID], device=self.device)

            outputs = [lang_output.SOS_ID]
            for _ in range(self.max_output_length):
                embedded_trg = self.decoder_embedding(input_).unsqueeze(1)
                output, (hidden, cell) = self.decoder_lstm(embedded_trg, (hidden, cell))
                prediction = self.fc(output.squeeze(1))
                predicted_token = prediction.argmax(1).item()

                outputs.append(predicted_token)

                if predicted_token == lang_output.EOS_ID:
                    break

                input_ = torch.tensor([predicted_token], device=self.device)

        return [lang_output.index2token[token] for token in outputs]

    def _predict_batch(self, src: Union[list, ndarray, Tensor], lang_output: Language) -> List[List[str]]:
        src = self._process_batch_input(src)
        batch_size = src.size(0)

        with torch.inference_mode():
            embedded_src = self.encoder_embedding(src)  # [batch_size, seq_len, embed_size]
            encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)

            if len(hidden.shape) != 3:
                raise ValueError("Hidden shape is not 3D")

            # Concatenate forward and backward states for each layer
            # Bidirectional encoder returns (num_layers * 2, batch, hidden_size)
            # We need (num_layers, batch, hidden_size * 2) for unidirectional decoder
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
            cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
            cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

            input_tokens = torch.full((batch_size,), lang_output.SOS_ID, device=self.device, dtype=torch.long)
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            batch_outputs = [[] for _ in range(batch_size)]

            for i in range(batch_size):
                batch_outputs[i].append(lang_output.SOS_ID)

            for step in range(self.max_output_length):
                if not active_mask.any():
                    break

                embedded_input = self.decoder_embedding(input_tokens.unsqueeze(1))  # [batch_size, 1, embed_size]

                output, (hidden, cell) = self.decoder_lstm(embedded_input, (hidden, cell))
                predictions = self.fc(output.squeeze(1))  # [batch_size, vocab_size]
                predicted_tokens = predictions.argmax(1)  # [batch_size]

                for i in range(batch_size):
                    if active_mask[i]:
                        token_id = predicted_tokens[i].item()
                        batch_outputs[i].append(token_id)

                        if token_id == lang_output.EOS_ID:
                            active_mask[i] = False

                input_tokens = predicted_tokens

            all_results = []
            for sequence_tokens in batch_outputs:
                result = [lang_output.index2token[token] for token in sequence_tokens]
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

                self.optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        output = self(src, trg)

                        # Reshape for the loss function
                        output_dim = output.shape[-1]
                        output = output[:, 1:].reshape(-1, output_dim)
                        trg = trg[:, 1:].reshape(-1)

                        loss = self.criterion(output, trg)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    output = self(src, trg)

                    # Reshape for the loss function
                    output_dim = output.shape[-1]
                    output = output[:, 1:].reshape(-1, output_dim)
                    trg = trg[:, 1:].reshape(-1)

                    loss = self.criterion(output, trg)
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()

            # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
            pbar.set_postfix(loss=epoch_loss / len(dataloader))

            if eval_every and eval_fn:
                if epoch % eval_every == 0:
                    losses.append(epoch_loss / len(dataloader))
                    evals.append(eval_fn(**eval_args))
                    self.train()

        if not eval_every:
            losses.append(epoch_loss / len(dataloader))
        elif epoch % eval_every != 0:
            losses.append(epoch_loss / len(dataloader))
            if eval_fn:
                evals.append(eval_fn(**eval_args))

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
            self.encoder_embedding = nn.Embedding(self.input_size, self.embed_size).to(self.device)
            self.decoder_embedding = nn.Embedding(self.output_size, self.embed_size).to(self.device)
            self.fc = nn.Linear(self.hidden_size * 2, self.output_size).to(self.device)
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
            self.encoder_embedding = self._resize_embedding_layer(
                self.encoder_embedding, new_input_size, input_mapping, init_std
            )
            self.input_size = new_input_size
            self.params["input_size"] = self.input_size
        
        if new_output_size > old_output_size:
            self.decoder_embedding = self._resize_embedding_layer(
                self.decoder_embedding, new_output_size, output_mapping, init_std
            )
            
            self.fc = self._resize_linear_layer(
                self.fc, new_output_size, output_mapping, init_std
            )
            self.output_size = new_output_size
            self.params["output_size"] = self.output_size
        
        # Update current language references with merged languages
        self.set_current_languages(merged_input_lang, merged_output_lang)
        
        return self

