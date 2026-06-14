import numpy as np
from sklearn.model_selection import train_test_split


def _greedy_group_split(unique_groups, group_sizes, train_ratio, eval_ratio, test_ratio):
    """
    Assign groups to train/eval/test to approximate the requested sample-count
    ratios using a greedy LPT (Longest Processing Time) strategy.

    Groups are sorted largest-first. Each is assigned to whichever split has
    the largest remaining sample gap. After the greedy pass, any split that
    ended up empty (possible with very few groups) steals the smallest group
    from the split with the most groups.
    """
    total   = sum(group_sizes.values())
    targets = {'train': total * train_ratio, 'eval': total * eval_ratio, 'test': total * test_ratio}
    filled  = {'train': 0.0, 'eval': 0.0, 'test': 0.0}
    assignment = {g: None for g in unique_groups}

    for g in sorted(unique_groups, key=lambda g: group_sizes[g], reverse=True):
        best = max(('train', 'eval', 'test'), key=lambda s: targets[s] - filled[s])
        assignment[g]  = best
        filled[best]  += group_sizes[g]

    # Guarantee at least 1 group per split
    splits = {'train', 'eval', 'test'}
    for empty_split in splits:
        if not any(s == empty_split for s in assignment.values()):
            donor = max(splits - {empty_split},
                        key=lambda s: sum(1 for v in assignment.values() if v == s))
            # Take the smallest group from the donor split
            victim = min((g for g, s in assignment.items() if s == donor),
                         key=lambda g: group_sizes[g])
            assignment[victim] = empty_split

    return (
        {g for g, s in assignment.items() if s == 'train'},
        {g for g, s in assignment.items() if s == 'eval'},
        {g for g, s in assignment.items() if s == 'test'},
    )


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

        unique_groups = sorted(set(group_keys))
        if len(unique_groups) < 3:
            raise ValueError(
                f"Need at least 3 groups for a train/eval/test split, got {len(unique_groups)}."
            )
        group_sizes = {g: sum(1 for gk in group_keys if gk == g) for g in unique_groups}
        train_set, eval_set, test_set = _greedy_group_split(
            unique_groups, group_sizes, train_ratio, eval_ratio, test_ratio
        )

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
