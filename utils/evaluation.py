"""
Evaluation utilities for RIR comparison.

Contains functions for selecting representative samples based on metrics.
Used by full_model_eval.py and synthetic_eval.py.
"""

import numpy as np


# Metric extractors for selecting samples
METRIC_EXTRACTORS = {
    't60': lambda m: abs(m['t60']['broadband']) if not np.isnan(m['t60']['broadband']) else np.inf,
    't60_perc': lambda m: _t60_perc_extractor(m),
    'lsd': lambda m: m['lsd']['broadband'],
    'edt': lambda m: abs(m['edt']['broadband']) if not np.isnan(m['edt']['broadband']) else np.inf,
    'drr': lambda m: abs(m['drr']['broadband']) if not np.isnan(m['drr']['broadband']) else np.inf,
    'c50': lambda m: abs(m['c50']['broadband']) if not np.isnan(m['c50']['broadband']) else np.inf,
    'c80': lambda m: abs(m['c80']['broadband']) if not np.isnan(m['c80']['broadband']) else np.inf,
    'edc': lambda m: m['edc_distance']['broadband'],
    'cosine': lambda m: m['cosine_similarity'],
}


def _t60_perc_extractor(m):
    """Extract T60 percentage error from metrics dict."""
    t60_gen = m['t60']['broadband_gen']
    t60_ref = m['t60']['broadband_ref']
    if (t60_gen is not None and t60_ref is not None and
        not np.isnan(t60_gen) and not np.isnan(t60_ref) and t60_ref != 0):
        return abs(t60_gen - t60_ref) / t60_ref * 100
    return np.inf


def select_representative_samples(all_samples, metric_names, n_samples_per_category=4):
    """Select best, worst, median, and mean-closest samples for each metric.

    Args:
        all_samples: List of dicts with 'generated', 'reference', 'condition', 'metrics'
        metric_names: List of metric names (e.g., ['t60', 'lsd'])
        n_samples_per_category: Number of samples to select for each category (default: 4)

    Returns:
        dict: {metric_name: {'best': [{'sample': ..., 'idx': ..., 'value': ...}, ...], 'worst': [...], ...}}
              Each category contains a list of n_samples_per_category samples.
    """
    selected = {}

    for metric_name in metric_names:
        if metric_name not in METRIC_EXTRACTORS:
            print(f"Warning: Unknown metric '{metric_name}', skipping")
            continue

        extractor = METRIC_EXTRACTORS[metric_name]

        # Extract metric values from all samples (keep inf for worst, filter only nan for safety)
        values = []
        for sample in all_samples:
            val = extractor(sample['metrics'])
            # Keep inf values (they represent worst cases), but safety check for nan
            if not np.isnan(val):
                values.append(val)
            else:
                values.append(np.inf if metric_name != 'cosine' else -np.inf)

        values = np.array(values)
        n_total = len(values)
        n_select = min(n_samples_per_category, n_total)

        # Sort indices by metric value
        # For most metrics, lower is better (except cosine where higher is better)
        if metric_name == 'cosine':
            sorted_indices = np.argsort(values)[::-1]  # Descending (best=highest first)
        else:
            sorted_indices = np.argsort(values)  # Ascending (best=lowest first)

        # Select top n_select as best (beginning of sorted)
        best_indices = sorted_indices[:n_select]

        # Select bottom n_select as worst (end of sorted, but reversed to have worst first)
        worst_indices = sorted_indices[-n_select:][::-1]

        # For median and mean, exclude inf values
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        finite_indices = np.where(finite_mask)[0]

        if len(finite_values) >= n_select:
            # Sort finite values
            finite_sorted_order = np.argsort(finite_values)
            if metric_name == 'cosine':
                finite_sorted_order = finite_sorted_order[::-1]  # Descending for cosine
            finite_sorted_indices = finite_indices[finite_sorted_order]

            # Median: select n_select samples around the median
            median_center = len(finite_sorted_indices) // 2
            median_start = max(0, median_center - n_select // 2)
            median_end = min(len(finite_sorted_indices), median_start + n_select)
            median_start = max(0, median_end - n_select)  # Adjust if we hit the end
            median_indices = finite_sorted_indices[median_start:median_end]

            # Mean: select n_select samples closest to the mean value
            mean_val = np.mean(finite_values)
            distances_to_mean = np.abs(values - mean_val)
            # Set inf values to have large distance so they're not selected
            distances_to_mean[~finite_mask] = np.inf
            mean_sorted_indices = np.argsort(distances_to_mean)
            mean_indices = mean_sorted_indices[:n_select]
        else:
            # Fallback if not enough finite values
            median_indices = sorted_indices[n_total // 4: n_total // 4 + n_select]
            mean_indices = sorted_indices[n_total // 3: n_total // 3 + n_select]
            mean_val = np.inf

        # Build the result structure with lists of samples
        def build_sample_list(indices):
            return [
                {'sample': all_samples[idx], 'idx': int(idx), 'value': float(values[idx])}
                for idx in indices
            ]

        selected[metric_name] = {
            'best': build_sample_list(best_indices),
            'worst': build_sample_list(worst_indices),
            'median': build_sample_list(median_indices),
            'mean': build_sample_list(mean_indices),
        }

        print(f"\n{metric_name} - Selected samples ({n_select} per category):")
        print(f"  Best:   indices={list(best_indices)}, values={[f'{values[i]:.4f}' for i in best_indices]}")
        print(f"  Worst:  indices={list(worst_indices)}, values={[f'{values[i]:.4f}' for i in worst_indices]}")
        print(f"  Median: indices={list(median_indices)}, values={[f'{values[i]:.4f}' for i in median_indices]}")
        print(f"  Mean:   indices={list(mean_indices)}, values={[f'{values[i]:.4f}' for i in mean_indices]} (target={mean_val:.4f})")

    return selected
