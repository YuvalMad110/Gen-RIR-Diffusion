from data.gtu_rir import create_gtu_datasets
from data.soundspaces_replica import create_soundspaces_datasets


def load_rir_dataset(name, path=None, split=True, nSamples=None,
                     train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15,
                     random_seed=42, split_by_room=True,
                     mode='raw', sample_max_sec=2, hop_length=256, n_fft=512,
                     use_spectrogram=False, sr_target=None,
                     # SoundSpaces-specific
                     mapping_csv=None, geometry_csv=None,
                     rir_root=None, image_root=None, scenes=None,
                     rir_view_type=None, room_overview_type=None, room_overview_config=None,
                     image_size=None, use_rt60=False):
    """
    Load RIR dataset(s).

    Args:
        name:           Dataset name: 'gtu' or 'soundspaces'
        path:           Path to dataset file (GTU pickle) or unused for SoundSpaces
        split:          If True, return (train, eval, test); if False, single dataset
        nSamples:       Number of samples to use (GTU only), None for all
        train_ratio:    Proportion for training (default 0.7)
        eval_ratio:     Proportion for evaluation (default 0.15)
        test_ratio:     Proportion for test (default 0.15)
        random_seed:    Seed for reproducible splits
        split_by_room:  If True, split by room/scene to avoid data leakage
        mode:           Processing mode (GTU only)
        sample_max_sec: Maximum RIR length in seconds
        hop_length:     Hop length for audio processing (GTU only)
        n_fft:          FFT size (GTU only)
        use_spectrogram: Whether to use spectrogram (GTU only)
        sr_target:      Target sample rate
        mapping_csv:    Path to soundspaces_replica_mapping.csv (SoundSpaces only)
        geometry_csv:   Path to room_geometry.csv (SoundSpaces only)
        rir_root:       SoundSpaces RIR root directory (SoundSpaces only)
        image_root:     Rendered RGB image root directory (SoundSpaces only)
        scenes:         List of scene names to include (SoundSpaces only)

    Returns:
        If split=True: (train_dataset, eval_dataset, test_dataset)
        If split=False: single dataset
    """
    if name == 'gtu':
        return create_gtu_datasets(
            tar_path=path,
            split=split,
            nSamples=nSamples,
            train_ratio=train_ratio,
            eval_ratio=eval_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
            split_by_room=split_by_room,
            mode=mode,
            sample_max_sec=sample_max_sec,
            hop_length=hop_length,
            n_fft=n_fft,
            use_spectrogram=use_spectrogram,
            sr_target=sr_target,
        )
    elif name == 'soundspaces':
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
            train_ratio=train_ratio,
            eval_ratio=eval_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
            sample_max_sec=sample_max_sec,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset name: {name!r}. Expected 'gtu' or 'soundspaces'.")
