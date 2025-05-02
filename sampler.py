import torch
from torch.utils.data import Sampler
import numpy as np

class BalancedBatchSampler(Sampler):
    """
    Batch sampler that ensures balanced class distribution in each batch.

    Particularly useful for imbalanced datasets where one class is more prevalent.
    Each batch will contain approximately equal numbers of each class.

    Args:
        dataset: Dataset to sample from. Must implement __getitem__ that returns a dict with 'gender' key.
        batch_size: Size of mini-batch. Should be divisible by 2 for binary classification.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        if batch_size % 2 != 0:
            print(f"Warning: batch_size {batch_size} is not divisible by 2. Rounding down samples per class.")

        self.samples_per_class = batch_size // 2
        self.indices_per_class = self._get_indices_per_class()
        self.batch_count = self._get_batch_count()

    def _get_indices_per_class(self):
        """Group dataset indices by class."""
        indices_per_class = {0: [], 1: []}

        for idx in range(len(self.dataset)):
            # Get the gender label (0 for 'M', 1 for 'W')
            item = self.dataset[idx]
            gender = int(item['gender'].item())
            indices_per_class[gender].append(idx)

        return indices_per_class

    def _get_batch_count(self):
        """Calculate how many complete balanced batches we can make."""
        class_counts = [len(indices) for indices in self.indices_per_class.values()]
        min_class_count = min(class_counts)

        # Number of batches is limited by the minority class
        return min_class_count // self.samples_per_class

    def __iter__(self):
        """Generate balanced batches by sampling from each class equally."""
        # Shuffle indices for each class
        indices_per_class = {
            class_idx: np.random.permutation(indices).tolist()
            for class_idx, indices in self.indices_per_class.items()
        }

        for batch_idx in range(self.batch_count):
            batch = []

            # Add samples from each class
            for class_idx in range(2):
                start_idx = batch_idx * self.samples_per_class
                end_idx = start_idx + self.samples_per_class
                batch.extend(indices_per_class[class_idx][start_idx:end_idx])

            # Shuffle the batch to avoid having all samples of one class together
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        """Return the number of batches."""
        return self.batch_count
