from data.soundspaces_replica import create_soundspaces_datasets


def load_rir_dataset(name, split=True,
                     train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15,
                     random_seed=42, split_by_room=True,
                     sample_max_sec=2, sr_target=None,
                     # SoundSpaces-specific
                     mapping_csv=None, geometry_csv=None,
                     rir_root=None, image_root=None, scenes=None,
                     rir_view_type=None, room_overview_type=None, room_overview_config=None,
                     image_size=None, use_rt60=False):
    """
    Load RIR dataset(s).

    Args:
        name:           Dataset name: 'soundspaces'
        split:          If True, return (train, eval, test); if False, single dataset
        train_ratio:    Proportion for training (default 0.7)
        eval_ratio:     Proportion for evaluation (default 0.15)
        test_ratio:     Proportion for test (default 0.15)
        random_seed:    Seed for reproducible splits
        split_by_room:  If True, split at scene level to avoid data leakage (default True)
        sample_max_sec: Maximum RIR length in seconds
        sr_target:      Target sample rate
        mapping_csv:    Path to soundspaces_replica_mapping.csv
        geometry_csv:   Path to room_geometry.csv
        rir_root:       SoundSpaces RIR root directory
        image_root:     Rendered RGB image root directory
        scenes:         List of scene names to include

    Returns:
        If split=True: (train_dataset, eval_dataset, test_dataset)
        If split=False: single dataset
    """
    if name == 'soundspaces':
        kwargs = {}
        if mapping_csv         is not None: kwargs['mapping_csv']         = mapping_csv
        if geometry_csv        is not None: kwargs['geometry_csv']        = geometry_csv
        if rir_root            is not None: kwargs['rir_root']            = rir_root
        if image_root          is not None: kwargs['image_root']          = image_root
        if sr_target           is not None: kwargs['sr_target']           = sr_target
        if rir_view_type        is not None: kwargs['rir_view_type']        = rir_view_type
        if room_overview_type   is not None: kwargs['room_overview_type']   = room_overview_type
        if room_overview_config is not None: kwargs['room_overview_config'] = room_overview_config
        if image_size           is not None: kwargs['image_size']           = image_size
        return create_soundspaces_datasets(
            scenes=scenes,
            split=split,
            split_by_room=split_by_room,
            train_ratio=train_ratio,
            eval_ratio=eval_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
            sample_max_sec=sample_max_sec,
            use_rt60=use_rt60,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset name: {name!r}. Expected 'soundspaces'.")
