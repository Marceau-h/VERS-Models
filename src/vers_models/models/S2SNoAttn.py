# SPDX-FileCopyrightText: 2028-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import Union, Optional, List

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

        # Encoder components
        self.encoder_embedding = nn.Embedding(
            self.input_size,
            self.output_size,
        )
        self.encoder_lstm = nn.LSTM(
            self.output_size,
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

    def forward(self, src:Tensor, trg:Tensor) -> Tensor:
        batch_size, trg_len = trg.size()
        trg_vocab_size = self.fc.out_features

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)

        # Encode the source sequence
        embedded_src = self.encoder_embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)

        # Concatenate the forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).unsqueeze(0)
        cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1).unsqueeze(0)

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

    def predict(self, src:Union[ndarray, list, Tensor], lang_output:Language) -> List[str]:
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

            # Initialize the decoder input with the <sos> token
            input_ = torch.tensor([lang_output.SOS_ID], device=self.device)

            outputs = [lang_output.SOS_ID]
            for _ in range(self.output_size):
                embedded_trg = self.decoder_embedding(input_).unsqueeze(1)
                output, (hidden, cell) = self.decoder_lstm(embedded_trg, (hidden, cell))
                prediction = self.fc(output.squeeze(1))
                predicted_token = prediction.argmax(1).item()

                outputs.append(predicted_token)

                if predicted_token == lang_output.EOS_ID:
                    break

                input_ = torch.tensor([predicted_token], device=self.device)

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

                self.optimizer.zero_grad()
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

