import hashlib
import numpy as np
from sklearn.model_selection import train_test_split


def create_data_splits(items, group_keys=None, split_by_group=True,
                       train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split items into train/eval/test lists.

    When split_by_group=True and group_keys is provided, the split is performed
    at the group level so no group appears in more than one split (prevents data
    leakage). Groups are ordered deterministically via MD5 hash of their string
    representation combined with the seed, then sliced by ratio.

    When split_by_group=False, a random item-level split is used.

    Args:
        items:           list of anything to split
        group_keys:      parallel list of group identifiers (one per item).
                         Required when split_by_group=True. Typical values:
                         room IDs (GTU), scene names (SoundSpaces).
        split_by_group:  if True, split at group level (no data leakage);
                         if False, random item-level split (default True)
        train_ratio:     fraction for training (default 0.7)
        eval_ratio:      fraction for evaluation (default 0.15)
        test_ratio:      fraction for test (default 0.15)
        seed:            integer seed for reproducibility

    Returns:
        (train_items, eval_items, test_items) — three lists of items
    """
    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    if split_by_group:
        assert group_keys is not None, \
            "group_keys must be provided when split_by_group=True"

        # Deterministic group ordering via hash
        unique_groups = sorted(set(group_keys))
        hash_values = [
            int(hashlib.md5(f"{g}_{seed}".encode()).hexdigest()[:8], 16)
            for g in unique_groups
        ]
        unique_groups = [unique_groups[i] for i in np.argsort(hash_values)]

        n = len(unique_groups)
        n_train = int(n * train_ratio)
        n_eval  = int(n * eval_ratio)

        train_set = set(unique_groups[:n_train])
        eval_set  = set(unique_groups[n_train:n_train + n_eval])
        test_set  = set(unique_groups[n_train + n_eval:])

        train_items = [item for item, gk in zip(items, group_keys) if gk in train_set]
        eval_items  = [item for item, gk in zip(items, group_keys) if gk in eval_set]
        test_items  = [item for item, gk in zip(items, group_keys) if gk in test_set]
    else:
        indices = np.arange(len(items))
        train_indices, temp_indices = train_test_split(
            indices, test_size=(eval_ratio + test_ratio),
            random_state=seed, shuffle=True
        )
        eval_indices, test_indices = train_test_split(
            temp_indices, test_size=test_ratio / (eval_ratio + test_ratio),
            random_state=seed, shuffle=True
        )
        train_items = [items[i] for i in train_indices]
        eval_items  = [items[i] for i in eval_indices]
        test_items  = [items[i] for i in test_indices]

    return train_items, eval_items, test_items
