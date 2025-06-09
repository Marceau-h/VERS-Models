# SPDX-FileCopyrightText: 2025-present Marceau <git@marceau-h.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
from typing import Union, List
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from tqdm.auto import trange
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import polars as pl

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class DFVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_z_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)

        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # x: [batch, seq_len, input_dim]
        _, (h, _) = self.encoder(x)

        h_final = h[-1]  # [batch, hidden_dim]

        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)
        z = self.reparameterize(mu, logvar)

        batch, seq_len, _ = x.size()
        dec_hidden = self.fc_z_to_hidden(z).view(self.num_layers, batch, self.hidden_dim)
        dec_cell = torch.zeros_like(dec_hidden)

        dec_input = torch.zeros(batch, seq_len, self.input_dim, device=x.device)
        dec_out, _ = self.decoder(dec_input, (dec_hidden, dec_cell))
        recon = self.out(dec_out)
        return recon, mu, logvar

    def encode(self, x: Tensor) -> Tensor:
        _, (h, _) = self.encoder(x)
        h_final = h[-1]
        mu = self.fc_mu(h_final)
        return mu


class NpyMemmapDataset(Dataset):
    def __init__(self, paths: list[Union[str, Path]]):
        self.maps = [np.load(str(p), mmap_mode='r') for p in paths]
        lengths = [m.shape[0] for m in self.maps]
        self.cum_lens = np.cumsum(lengths)
        self.total_len = int(self.cum_lens[-1])

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int):
        file_idx = np.searchsorted(self.cum_lens, idx, side='right')
        prev = int(self.cum_lens[file_idx - 1]) if file_idx > 0 else 0
        local_idx = idx - prev
        arr = self.maps[file_idx][local_idx]
        x = torch.tensor(arr, dtype=torch.float32)
        return x, x


class DFVAEPipeline:
    def __init__(
            self,
            hidden_dim: int,
            latent_dim: int,
            num_layers: int = 1,
            lr: float = 1e-3,
            device: torch.device = None,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.lr = lr
        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None

    def visualize(
            self,
            codes: np.ndarray,
            orig_idx: Union[np.ndarray, List[int]] = None,
            labels: Union[np.ndarray, List] = None,
            method: str = 'pca',
            save_path: Union[str, Path] = None,
            n_clusters: int = None,
            y_dataset: Union[Dataset, str, Path, None] = None,
    ) -> None:
        if codes.ndim != 2:
            raise ValueError("codes must be a 2D array of shape (n_samples, latent_dim)")
        if n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            labels = kmeans.fit_predict(codes)

        indices = np.array(orig_idx) if orig_idx is not None else np.arange(codes.shape[0])

        if codes.shape[1] > 2:
            if method.lower() == 'pca':
                reducer = PCA(n_components=2)
            else:
                raise ValueError(f"Unsupported method '{method}'. Only 'pca' is supported.")
            embedded = reducer.fit_transform(codes)
        else:
            embedded = codes

        df = pl.DataFrame({'index': indices, 'dim1': embedded[:, 0], 'dim2': embedded[:, 1]})
        if y_dataset is not None:
            df = df.with_columns(
                y=pl.Series(y_dataset)
            )

        if labels is not None:
            df = df.with_columns(
                label=pl.Series(labels, dtype=pl.Int32)
            )

        fig = px.scatter(
            df,
            x='dim1',
            y='dim2',
            color='label' if 'label' in df.columns else None,
            hover_data=['index', 'dim1', 'dim2', 'label'] if 'label' in df.columns else ['index', 'dim1', 'dim2'],
            title='Latent Code Distribution',
            labels={'dim1': 'Latent Dimension 1', 'dim2': 'Latent Dimension 2',
                    'label': 'Cluster Label' if 'label' in df.columns else None}
        )

        if save_path is not None:
            fig.write_html(save_path.with_suffix('.html'))
            fig.write_image(save_path.with_suffix('.png'))
        fig.show()

    def fit(self,
            latent_data: Union[
                Dataset, str, Path, List[Union[str, Path, np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
            epochs: int = 10,
            batch_size: int = 32,
            ):
        if isinstance(latent_data, Dataset):
            dataset = latent_data
        elif isinstance(latent_data, (str, Path)):
            dataset = NpyMemmapDataset([latent_data])
        elif isinstance(latent_data, list) and all(isinstance(d, (str, Path)) for d in latent_data):
            dataset = NpyMemmapDataset(latent_data)
        elif isinstance(latent_data, (np.ndarray, torch.Tensor)):
            arr = latent_data.cpu().numpy() if isinstance(latent_data, torch.Tensor) else latent_data
            X = torch.tensor(arr, dtype=torch.float32)
            dataset = TensorDataset(X, X)
        elif isinstance(latent_data, list):
            arrs = []
            for d in latent_data:
                if isinstance(d, (str, Path)):
                    arrs.append(np.load(str(d)))
                elif isinstance(d, torch.Tensor):
                    arrs.append(d.cpu().numpy())
                else:
                    arrs.append(d)
            arr = np.concatenate(arrs, axis=0)
            X = torch.tensor(arr, dtype=torch.float32)
            dataset = TensorDataset(X, X)
        else:
            raise TypeError("Unsupported data type for fit")

        origin = dataset
        if isinstance(origin, Subset):
            origin = origin.dataset
        if isinstance(origin, NpyMemmapDataset):
            input_dim = origin.maps[0].shape[-1]
        elif isinstance(origin, TensorDataset):
            input_dim = origin.tensors[0].shape[-1]
        else:
            raise TypeError(f"Unsupported dataset type {type(origin)} for inferring input_dim")

        self.model = DFVAE(input_dim, self.hidden_dim, self.latent_dim, self.num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        print(f"Starting training: {epochs} epochs, batch_size={batch_size}, device={self.device}")
        pbar = trange(1, epochs + 1, desc="Epochs", unit="epoch")
        for _ in pbar:
            epoch_loss = 0.0
            for xb, _ in loader:
                xb = xb.to(self.device)
                self.optimizer.zero_grad()
                recon, mu, logvar = self.model(xb)
                mse = self.criterion(recon, xb)

                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = mse + kld
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            pbar.set_postfix(loss=avg_loss)

        print("Training complete.")
        return self

    def transform(
            self,
            latent_data: Union[Dataset, str, Path, np.ndarray, torch.Tensor],
            output_npy: Union[str, Path, None] = None,
            batch_size: int = 32,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit first.")

        if isinstance(latent_data, Dataset):
            loader = DataLoader(latent_data, batch_size=batch_size)
            codes_list = []
            self.model.eval()
            with torch.inference_mode():
                for xb, _ in loader:
                    xb = xb.to(self.device)
                    codes = self.model.encode(xb)
                    codes_list.append(codes.cpu().numpy())
            codes_np = np.concatenate(codes_list, axis=0)
            if output_npy:
                np.save(str(output_npy), codes_np)
            return codes_np

        if isinstance(latent_data, (str, Path)):
            arr = np.load(str(latent_data))
        elif isinstance(latent_data, torch.Tensor):
            arr = latent_data.cpu().numpy()
        else:
            arr = latent_data
        X = torch.tensor(arr, dtype=torch.float32, device=self.device)

        self.model.eval()

        with torch.inference_mode():
            codes = self.model.encode(X)  # [n, latent_dim]
        codes_np = codes.cpu().numpy()
        if output_npy:
            np.save(str(output_npy), codes_np)
        return codes_np

    def evaluate(self, latent_data: Union[Dataset, str, Path, np.ndarray, torch.Tensor], batch_size: int = 32) -> float:
        if isinstance(latent_data, Dataset):
            loader = DataLoader(latent_data, batch_size=batch_size)
        else:
            if isinstance(latent_data, (str, Path)):
                arr = np.load(str(latent_data))
            elif isinstance(latent_data, torch.Tensor):
                arr = latent_data.cpu().numpy()
            else:
                arr = latent_data

            X = torch.tensor(arr, dtype=torch.float32, device=self.device)
            loader = DataLoader(TensorDataset(X, X), batch_size=batch_size)

        self.model.eval()

        total, count = 0.0, 0
        with torch.inference_mode():
            for xb, _ in loader:
                xb = xb.to(self.device)
                recon, mu, logvar = self.model(xb)
                loss = self.criterion(recon, xb)
                total += loss.item() * xb.size(0)
                count += xb.size(0)
        return total / count


def main(
        paths: List[Path],
        y_paths: List[Path] = None,
        test_size: float = 0.2,
        shuffle: bool = True,
        model_path: Path = Path('dfvae_model.pth'),
        train_codes_path: Path = Path('train_codes.npy'),
        test_codes_path: Path = Path('test_codes.npy'),
        permutation_path: Path = Path('permutation.npy'),
        epochs: int = 10,
        batch_size: int = 32,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 1,
        lr: float = 1e-3,
):
    dataset = NpyMemmapDataset(paths)
    n = len(dataset)
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    if y_paths is not None:
        y_dataset = NpyMemmapDataset(y_paths)
        if len(y_dataset) != n:
            raise ValueError("Length of y_paths dataset does not match the length of paths dataset.")
        # Load full label array for direct indexing
        labels_array = np.concatenate(y_dataset.maps, axis=0)
    else:
        y_dataset = None
        labels_array = None

    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    train_ds = Subset(dataset, train_idx)
    test_ds = Subset(dataset, test_idx)
    np.save(str(permutation_path), indices)

    pipeline = DFVAEPipeline(hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers, lr=lr)
    pipeline.fit(train_ds, epochs=epochs, batch_size=batch_size)

    torch.save(pipeline.model.state_dict(), str(model_path))

    mse = pipeline.evaluate(test_ds, batch_size=batch_size)
    print(f"Test MSE: {mse:.6f}")

    train_codes = pipeline.transform(train_ds, None, batch_size=batch_size)
    test_codes = pipeline.transform(test_ds, None, batch_size=batch_size)

    np.save(str(train_codes_path), train_codes)
    np.save(str(test_codes_path), test_codes)

    pipeline.visualize(train_codes, orig_idx=train_idx, save_path=train_codes_path, n_clusters=8,
                       y_dataset=labels_array[train_idx] if labels_array is not None else None)
    pipeline.visualize(test_codes, orig_idx=test_idx, save_path=test_codes_path, n_clusters=8,
                       y_dataset=labels_array[test_idx] if labels_array is not None else None)


if __name__ == '__main__':
    paths = [
        Path('../../evals/S2SNoAttn_2025-06-08_22-37-40_latents.npy'),
    ]
    y_paths = [
        Path('../../evals/S2SNoAttn_2025-06-08_22-37-40_latents_y.npy'),
    ]
    test_size: float = 0.2
    shuffle: bool = True
    model_path: Path = Path('dfvae_model.pth')
    train_codes_path: Path = Path('train_codes.npy')
    test_codes_path: Path = Path('test_codes.npy')
    permutation_path: Path = Path('permutation.npy')
    epochs: int = 100
    batch_size: int = 3000

    hidden_dim: int = 64
    latent_dim: int = 32
    num_layers: int = 1
    lr: float = 1e-3
    main(
        paths=paths,
        y_paths=y_paths,
        test_size=test_size,
        shuffle=shuffle,
        model_path=model_path,
        train_codes_path=train_codes_path,
        test_codes_path=test_codes_path,
        permutation_path=permutation_path,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        lr=lr,
    )

    exit(0)

    import argparse

    parser = argparse.ArgumentParser(description='Train and eval DFVAE on latent data')
    parser.add_argument('paths', nargs='+', type=Path, help='Paths to latent .npy files')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--model-path', type=Path, default=Path('dfvae_model.pth'))
    parser.add_argument('--train-codes', type=Path, default=Path('train_codes.npy'))
    parser.add_argument('--test-codes', type=Path, default=Path('test_codes.npy'))
    parser.add_argument('--permutation', type=Path, default=Path('permutation.npy'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(
        paths=args.paths,
        test_size=args.test_size,
        shuffle=True,
        model_path=args.model_path,
        train_codes_path=args.train_codes,
        test_codes_path=args.test_codes,
        permutation_path=args.permutation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        lr=args.lr,
    )
