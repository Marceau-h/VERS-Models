import json
from pathlib import Path
from random import sample

import numpy as np
import torch
from torch import nn
from tqdm import trange
from tqdm.auto import tqdm

from .BaseModel import BaseModel


class S2SNoAttn(BaseModel):
    # def __init__(self, input_size, output_size, embed_size, hidden_size, num_layers=1):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        # Encoder components
        self.encoder_embedding = nn.Embedding(
            self.params["input_size"],
            self.params["embed_size"],
        )
        self.encoder_lstm = nn.LSTM(
            self.params["embed_size"],
            self.params["hidden_size"],
            num_layers=self.params["num_layers"],
            bidirectional=True,
            batch_first=True,
        )

        # Decoder components
        self.decoder_embedding = nn.Embedding(
            self.params["output_size"],
            self.params["embed_size"],
        )
        self.decoder_lstm = nn.LSTM(
            self.params["embed_size"],
            self.params["hidden_size"] * 2,
            num_layers=self.params["num_layers"],
            batch_first=True,
        )
        self.fc = nn.Linear(
            self.params["hidden_size"] * 2,
            self.params["output_size"],
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.params["lr"])
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, src, trg):
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
        input = trg[:, 0]

        for t in range(1, trg_len):
            embedded_trg = self.decoder_embedding(input).unsqueeze(1)

            # Decoder step
            output, (hidden, cell) = self.decoder_lstm(embedded_trg, (hidden, cell))
            prediction = self.fc(output.squeeze(1))
            outputs[:, t, :] = prediction

            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < self.params["teacher_forcing_ratio"]
            input = trg[:, t] if teacher_force else prediction.argmax(1)

        return outputs

    def predict(self, src, lang_output):
        self.eval()
        if isinstance(src, (np.ndarray, list)):
            src = torch.tensor(src, device=self.device)
        else:
            src = src.to(self.device)

        # Encode the source sequence
        with torch.inference_mode():
            embedded_src = self.encoder_embedding(src.unsqueeze(0))  # Add batch dimension
            encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)

            if len(hidden.shape) != 3:
                raise ValueError("Hidden shape is not 3D")

            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).unsqueeze(0)
            cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1).unsqueeze(0)

            # Initialize the decoder input with the <sos> token
            input = torch.tensor([lang_output.SOS_ID], device=self.device)

            outputs = [lang_output.SOS_ID]
            for _ in range(self.params["max_len"]):
                embedded_trg = self.decoder_embedding(input).unsqueeze(1)
                output, (hidden, cell) = self.decoder_lstm(embedded_trg, (hidden, cell))
                prediction = self.fc(output.squeeze(1))
                predicted_token = prediction.argmax(1).item()

                outputs.append(predicted_token)

                if predicted_token == lang_output.EOS_ID:
                    break

                input = torch.tensor([predicted_token], device=self.device)

        return [lang_output.index2token[token] for token in outputs]

    def do_train(
            self,
            device,
            dataloader,
            num_epochs=10,
            teacher_forcing_ratio=0.5,
            eval_every=None,
            eval_fn=None,
            eval_args=None,
            from_epoch=0,
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

