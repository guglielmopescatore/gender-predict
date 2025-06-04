"""
Learning Rate Finder implementation for finding optimal learning rate.
Based on Leslie Smith's paper: https://arxiv.org/abs/1506.01186
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from torch.utils.data import DataLoader
import math


class LRFinder:
    """
    Learning rate finder per identificare il learning rate ottimale.
    """

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Salva stato iniziale
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())

        # History
        self.history = {"lr": [], "loss": []}

    def find(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100,
             smooth_factor=0.98, diverge_threshold=4):
        """
        Esegue la ricerca del learning rate ottimale.

        Args:
            train_loader: DataLoader per il training
            start_lr: Learning rate iniziale
            end_lr: Learning rate finale
            num_iter: Numero di iterazioni
            smooth_factor: Fattore di smoothing per la loss
            diverge_threshold: Soglia per fermarsi quando la loss diverge

        Returns:
            learning_rates, losses: Liste di lr e loss per plotting
        """
        print(f"🔍 Starting LR finder: {start_lr:.2e} -> {end_lr:.2e}")

        # Reset
        self.history = {"lr": [], "loss": []}
        self.model.train()

        # Calcola multiplicatore
        lr_mult = (end_lr / start_lr) ** (1 / (num_iter - 1))
        lr = start_lr

        # Best loss tracking
        best_loss = float('inf')
        smoothed_loss = None

        # Iterator ciclico sul dataloader
        data_iter = iter(train_loader)

        # Progress bar
        pbar = tqdm(range(num_iter), desc="Finding LR")

        for iteration in pbar:
            # Get batch (cycle if needed)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Set LR
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass - gestisci diversi tipi di modello
            loss = self._train_batch(batch)

            # Smooth loss
            if smoothed_loss is None:
                smoothed_loss = loss
            else:
                smoothed_loss = smooth_factor * smoothed_loss + (1 - smooth_factor) * loss

            # Record
            self.history["lr"].append(lr)
            self.history["loss"].append(smoothed_loss)

            # Update progress bar
            pbar.set_postfix({'lr': f'{lr:.2e}', 'loss': f'{smoothed_loss:.4f}'})

            # Check if loss is diverging
            if iteration > 10 and smoothed_loss > diverge_threshold * best_loss:
                print(f"⚠️  Stopping early: loss is diverging at lr={lr:.2e}")
                break

            # Update best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Update LR
            lr *= lr_mult

        # Restore original state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        print("✅ LR finder complete!")

        return self.history["lr"], self.history["loss"]

    def _train_batch(self, batch):
        """Esegue training su un singolo batch."""
        self.optimizer.zero_grad()

        # Rileva tipo di modello dal batch
        if 'first_suffix' in batch:  # V3 model
            outputs = self.model(
                batch['first_name'].to(self.device),
                batch['last_name'].to(self.device),
                batch['first_suffix'].to(self.device),
                batch['last_suffix'].to(self.device),
                batch['phonetic_features'].to(self.device)
            )
        else:  # Standard models
            outputs = self.model(
                batch['first_name'].to(self.device),
                batch['last_name'].to(self.device)
            )

        loss = self.criterion(outputs, batch['gender'].to(self.device))
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def plot(self, skip_start=10, skip_end=5, suggestion=True, save_path=None):
        """
        Plotta il grafico lr vs loss.

        Args:
            skip_start: Salta i primi N punti (spesso rumorosi)
            skip_end: Salta gli ultimi N punti
            suggestion: Se True, suggerisce il LR ottimale
            save_path: Path dove salvare il plot

        Returns:
            suggested_lr: Learning rate suggerito (se suggestion=True)
        """
        lrs = self.history["lr"][skip_start:-skip_end] if skip_end > 0 else self.history["lr"][skip_start:]
        losses = self.history["loss"][skip_start:-skip_end] if skip_end > 0 else self.history["loss"][skip_start:]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lrs, losses)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Loss")
        ax.set_title("Learning Rate Finder")
        ax.grid(True, alpha=0.3)

        suggested_lr = None
        if suggestion:
            # Trova il punto con gradiente più ripido (metodo semplice)
            gradients = []
            for i in range(1, len(losses) - 1):
                grad = (losses[i+1] - losses[i-1]) / (np.log(lrs[i+1]) - np.log(lrs[i-1]))
                gradients.append(grad)

            # Trova il minimo gradiente (discesa più ripida)
            min_grad_idx = np.argmin(gradients) + 1  # +1 per l'offset
            suggested_lr = lrs[min_grad_idx]

            # Mostra sul plot
            ax.axvline(x=suggested_lr, color='red', linestyle='--',
                      label=f'Suggested LR: {suggested_lr:.2e}')
            ax.legend()

            print(f"\n💡 Suggested learning rate: {suggested_lr:.2e}")
            print(f"   (Use a bit lower for safety, e.g., {suggested_lr/10:.2e})")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")
        else:
            plt.show()

        return suggested_lr


def find_optimal_lr(model, train_loader, criterion, device,
                   start_lr=1e-7, end_lr=10, num_iter=100):
    """
    Funzione helper per uso rapido.

    Returns:
        optimal_lr: Learning rate ottimale suggerito
    """
    # Crea optimizer temporaneo
    temp_optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

    # Crea finder
    finder = LRFinder(model, temp_optimizer, criterion, device)

    # Trova LR
    lrs, losses = finder.find(train_loader, start_lr, end_lr, num_iter)

    # Plot e ottieni suggerimento
    optimal_lr = finder.plot(suggestion=True)

    return optimal_lr
