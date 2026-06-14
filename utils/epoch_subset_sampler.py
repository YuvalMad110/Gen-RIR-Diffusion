import torch
from torch.utils.data import Sampler


class EpochSubsetSampler(Sampler):
    """
    Randomly selects `n_samples` indices from the dataset each epoch.

    The selection is re-seeded by epoch number, so each epoch sees a different
    random subset of the full dataset. Call set_epoch(epoch) at the start of
    each training epoch to advance the seed.

    Compatible with DistributedSampler conventions — the epoch seed ensures
    all processes in a distributed run select the same indices.
    """

    def __init__(self, dataset_size: int, n_samples: int):
        self.dataset_size = dataset_size
        self.n_samples    = min(n_samples, dataset_size)
        self.epoch        = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.dataset_size, generator=g)[:self.n_samples]
        return iter(indices.tolist())

    def __len__(self):
        return self.n_samples
