# SPDX-FileCopyrightText: 2028-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import json
from pathlib import Path
from typing import Union, Iterable, Optional

from numpy import ndarray
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
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

    def predict(self, src:Union[ndarray, list, Tensor], lang_output:Language) -> Iterable[str]:
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
            for _ in range(self.output_size):
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
