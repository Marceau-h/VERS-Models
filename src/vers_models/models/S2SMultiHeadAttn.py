# SPDX-FileCopyrightText: 2025-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import json
from pathlib import Path
from typing import Union, Iterable, Optional

from numpy import ndarray
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

try:
    from .BaseModel import BaseModel
    from ..Language import Language, PAD_ID
except ImportError:
    from vers_models.models.BaseModel import BaseModel
    from vers_models.Language import Language, PAD_ID

class S2SMultiHeadAttn(BaseModel):
    # def __init__(self, input_size, output_size, embed_size, hidden_size, num_layers=1, num_heads=8):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input_size = self.params["input_size"]
        self.output_size = self.params["output_size"]
        self.embed_size = self.params["embed_size"]
        self.hidden_size = self.params["hidden_size"]
        self.num_layers = self.params["num_layers"]
        self.lr = self.params["lr"]
        self.teacher_forcing_ratio = self.params["teacher_forcing_ratio"]
        self.num_heads = self.params["num_heads"]
        self.max_input_length =self.params["max_input_length"]
        self.max_output_length = self.params["max_output_length"]


        # Encoder components
        self.encoder_embedding = nn.Embedding(self.input_size, self.embed_size)
        self.encoder_lstm = nn.LSTM(
            self.embed_size,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True
        )

        # Decoder components
        self.decoder_embedding = nn.Embedding(self.output_size, self.embed_size)
        self.decoder_lstm = nn.LSTM(
            self.embed_size, self.hidden_size * 2,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size * 2, self.num_heads, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size * 4, self.output_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def partial_forward(self, src:Tensor) -> Tensor:
        """
        Return encoder outputs as latent representation for given input sequence.
        """
        self.eval()
        with torch.inference_mode():
            # src: [batch, seq_len]
            embedded_src = self.encoder_embedding(src)
            encoder_outputs, (_hidden, _cell) = self.encoder_lstm(embedded_src)
        return encoder_outputs

    def forward(self, src:Tensor, trg:Tensor) -> Tensor:
        batch_size, trg_len = trg.size()
        trg_vocab_size = self.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        # output = src.zero_like(shape=(batch_size, trg_len, trg_vocab_size))

        # Encode the source sequence
        embedded_src = self.encoder_embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)

        # Concatenate the forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).unsqueeze(0)
        cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1).unsqueeze(0)

        # First input to the decoder is the <sos> token
        input = trg[:, 0]

        for t in range(1, trg_len):
            embedded_trg = self.decoder_embedding(input).unsqueeze(1)

            # Decoder step
            output, (hidden, cell) = self.decoder_lstm(embedded_trg, (hidden, cell))
            attn_output, _ = self.multihead_attn(output, encoder_outputs, encoder_outputs)
            combined = torch.cat((output.squeeze(1), attn_output.squeeze(1)), dim=1)
            prediction = self.fc_out(combined)  # [batch, output_size]
            outputs[:, t, :] = prediction

            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            input = trg[:, t] if teacher_force else prediction.argmax(1)

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

            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).unsqueeze(0)
            cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1).unsqueeze(0)

            # Initialize reusable input tensor with the <sos> token
            input_ = torch.tensor([lang_output.SOS_ID], device=self.device)
            input_.fill_(lang_output.SOS_ID)

            outputs = [lang_output.SOS_ID]
            for _ in range(self.max_output_length):
                embedded_trg = self.decoder_embedding(input_).unsqueeze(1)
                output, (hidden, cell) = self.decoder_lstm(embedded_trg, (hidden, cell))
                dec_state = output.squeeze(1)
                energy = torch.bmm(encoder_outputs, dec_state.unsqueeze(2)).squeeze(2)
                attn_weights = F.softmax(energy, dim=1)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
                combined = torch.cat((dec_state, context), dim=1)
                prediction = self.fc_out(combined)
                predicted_token = prediction.argmax(1).item()

                outputs.append(predicted_token)

                if predicted_token == lang_output.EOS_ID:
                    break

                input_.fill_(predicted_token)

        return [lang_output.index2token[token] for token in outputs]

    def _predict_batch(self, src: Union[list, ndarray, Tensor], lang_output: Language) -> Iterable[Iterable[str]]:
        src = self._process_batch_input(src)
        batch_size = src.size(0)

        with torch.inference_mode():
            embedded_src = self.encoder_embedding(src)  # [batch_size, seq_len, embed_size]
            encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)

            if len(hidden.shape) != 3:
                raise ValueError("Hidden shape is not 3D")

            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).unsqueeze(0)
            cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1).unsqueeze(0)

            input_tokens = torch.full((batch_size,), lang_output.SOS_ID, device=self.device, dtype=torch.long)
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

            batch_outputs = [[] for _ in range(batch_size)]

            for i in range(batch_size):
                batch_outputs[i].append(lang_output.SOS_ID)

            for step in range(self.max_output_length):
                if not active_mask.any():
                    break

                embedded_input = self.decoder_embedding(input_tokens.unsqueeze(1))  # [batch_size, 1, embed_size]
                decoder_output, (hidden, cell) = self.decoder_lstm(embedded_input, (hidden, cell))
                attn_output, _ = self.multihead_attn(decoder_output, encoder_outputs, encoder_outputs)

                combined = torch.cat((decoder_output.squeeze(1), attn_output.squeeze(1)), dim=1)
                predictions = self.fc_out(combined)  # [batch_size, vocab_size]
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

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self(src, trg)

                # Compute the loss
                loss = F.cross_entropy(output[:, 1:].reshape(-1, output.shape[2]), trg[:, 1:].reshape(-1), ignore_index=PAD_ID)
                epoch_loss += loss.item()

                # Backward pass and optimization
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            losses.append(epoch_loss / len(dataloader))

            if eval_every and eval_fn and eval_args:
                if epoch % eval_every == 0:
                    evals.append(eval_fn(**eval_args))
                    pbar.set_postfix({"loss": losses[-1], "eval": evals[-1]})
            else:
                pbar.set_postfix({"loss": losses[-1]})

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
            self.fc_out = nn.Linear(self.hidden_size * 4, self.output_size).to(self.device)
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
            
            self.fc_out = self._resize_linear_layer(
                self.fc_out, new_output_size, output_mapping, init_std
            )
            self.output_size = new_output_size
            self.params["output_size"] = self.output_size
        
        # Update current language references with merged languages
        self.set_current_languages(merged_input_lang, merged_output_lang)
        
        return self




def save_model(model, params, state, model_path, params_path):
    torch.save(model.state_dict(), model_path)

    params["model_path"] = model_path

    with open(params_path, "w") as f:
        json.dump(params, f, ensure_ascii=False, indent=4, default=model.jsonify_types)

    torch.save(state, params_path.with_suffix(".state"))

    print("Model and parameters saved successfully")


def load_model(params_path, model_path, device):
    with open(params_path, "r") as f:
        params = json.load(f)

    print(params)

    model = S2SBiLSTM(
        params["input_size"],
        params["output_size"],
        params["embed_size"],
        params["hidden_size"],
        params["num_layers"]
    ).to(device)

    model.load_state_dict(
        torch.load(
            f=params.get("model_path", model_path),
            weights_only=False,
        )
    )

    state = torch.load(params_path.with_suffix(".state"), weights_only=False)
    # model.load_state_dict(state["model_state_dict"], strict=False,
    #
    # optimizer = optim.Adam(model.parameters(), lr=params["optimizer_parameters"]["lr"])
    # optimizer.load_state_dict(state["optimizer_state_dict"])
    #
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    # criterion.load_state_dict(state["criterion_state_dict"])

    old_vocab_size = model.encoder_embedding.weight.shape[1]

    return model, state, old_vocab_size


def paths(pho: bool = False, suffix: str = "", json_: bool = False) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    assert isinstance(pho, bool), "pho must be a boolean"
    assert isinstance(suffix, str), "suffix must be a string"

    if pho and not suffix:  # if pho is True and suffix is empty
        suffix = "_pho"

    relative_to_root = 0
    cwd = Path.cwd()
    while cwd.name != "S2SBiLSTM":
        relative_to_root += 1
        cwd = cwd.parent

    prepend = Path("../" * relative_to_root)

    params_path = f"params{suffix}.json"
    model_path = prepend / f"model{suffix}.pth"
    data_path = prepend /  f"data{suffix}.{'json' if json_ else 'txt'}"
    x_data = prepend / f"X{suffix}.npy"
    y_data = prepend / f"y{suffix}.npy"
    lang_path = prepend / f"lang{suffix}.json"
    eval_path = prepend / f"results{suffix}.json"

    return params_path, model_path, data_path, x_data, y_data, lang_path, eval_path
