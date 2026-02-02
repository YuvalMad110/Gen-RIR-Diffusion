import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from pathlib import Path

def gen_edc(rir: np.ndarray, sr: int, octave_center: float) -> tuple:
    """
    Generate Energy Decay Curve (EDC) for a specific octave band.
    
    Args:
        rir: Room impulse response waveform
        sr: Sample rate
        octave_center: Center frequency of the octave band in Hz
        
    Returns:
        tuple: (time_axis, edc_db) where edc_db is the energy decay curve in dB
    """
    # Define octave band limits (1/3 octave would be factor of 2^(1/6))
    # For full octave: factor of sqrt(2) = 2^(1/2)
    factor = 2**(1/2)  # Full octave
    f_low = octave_center / factor
    f_high = octave_center * factor
    
    # Design bandpass filter
    nyquist = sr / 2
    low_norm = f_low / nyquist
    high_norm = f_high / nyquist
    
    # Ensure frequencies are within valid range
    low_norm = max(0.001, min(low_norm, 0.999))
    high_norm = max(low_norm + 0.001, min(high_norm, 0.999))
    
    # Create bandpass filter
    try:
        b, a = butter(4, [low_norm, high_norm], btype='band')
        filtered_rir = filtfilt(b, a, rir)
    except Exception as e:
        print(f"Warning: Filter design failed for {octave_center}Hz octave: {e}")
        # Fallback: use original signal
        filtered_rir = rir
    
    # Square the signal to get energy
    energy_signal = filtered_rir**2
    
    # Calculate backwards cumulative sum (Schroeder integration)
    # EDC(t) = integral from t to infinity of p²(τ) dτ
    edc = np.cumsum(energy_signal[::-1])[::-1]
    
    # Normalize to start at 0 dB
    edc = edc / edc[0] if edc[0] > 0 else edc
    
    # Convert to dB, avoiding log(0)
    edc_db = 10 * np.log10(edc + 1e-12)

    # Sanity check the result
    if np.all(np.isnan(edc_db)) or np.all(np.isinf(edc_db)):
        print(f"Warning: Invalid EDC values for {octave_center}Hz octave")
        edc_db = np.zeros_like(edc_db) - 60  # Fallback to -60 dB
    
    # Create time axis
    time_axis = np.arange(len(edc_db)) / sr
    
    return time_axis, edc_db


def format_condition_strings_with_metrics(conditions: np.ndarray, metrics: dict, mode: str = "room") -> list:
    """
    Format condition parameters with evaluation metrics for display.
    
    Args:
        conditions: Array of condition parameters for all RIRs
        metrics: Dictionary containing evaluation metrics (from evaluate_rir_quality)
        mode: Display mode ("room" or "locations")
        
    Returns:
        List of formatted strings, one for each RIR
    """
    formatted_strings = []
    
    for i in range(len(conditions)):
        condition = conditions[i]
        
        # Base condition string
        if mode == "room":
            room_dims = condition[:3]
            rt60 = condition[-1]
            base_str = f"Room: {room_dims[0]:.1f}×{room_dims[1]:.1f}×{room_dims[2]:.1f}m, RT60: {rt60:.2f}s"
        elif mode == "locations":
            mic_loc = condition[3:6]
            speaker_loc = condition[6:9]
            base_str = f"Mic: ({mic_loc[0]:.1f},{mic_loc[1]:.1f},{mic_loc[2]:.1f}), Src: ({speaker_loc[0]:.1f},{speaker_loc[1]:.1f},{speaker_loc[2]:.1f})"
        else:
            base_str = f"Condition {i+1}"
        
        # Add metrics
        metrics_parts = []
        for metric_name, metric_data in metrics.items():
            if 'individual' in metric_data and i < len(metric_data['individual']):
                individual_value = metric_data['individual'][i]
                formatted_value = f"{individual_value:.4f}" if not np.isinf(individual_value) else "inf"
                metric_abbrev = metric_name.upper() if len(metric_name) <= 4 else metric_name[:4].upper()
                metrics_parts.append(f"{metric_abbrev}: {formatted_value}")
                # # Format the metric value based on its type and range
                # if metric_name.lower() == 'nmse':
                #     if np.isinf(individual_value):
                #         formatted_value = "inf"
                #     else:
                #         formatted_value = f"{individual_value:.4f}"
                #     metrics_parts.append(f"NMSE: {formatted_value}")
                # elif metric_name.lower() == 'cosine_distance':
                #     formatted_value = f"{individual_value:.4f}"
                #     metrics_parts.append(f"CD: {formatted_value}")
                # else:
                #     # Generic formatting for other metrics
                #     if np.isinf(individual_value):
                #         formatted_value = "inf"
                #     elif abs(individual_value) < 0.01:
                #         formatted_value = f"{individual_value:.6f}"
                #     else:
                #         formatted_value = f"{individual_value:.4f}"
                #     # Use abbreviated metric name if it's long
                #     metric_abbrev = metric_name.upper() if len(metric_name) <= 4 else metric_name[:4].upper()
                #     metrics_parts.append(f"{metric_abbrev}: {formatted_value}")
        
        # Combine base string with metrics
        if metrics_parts:
            metrics_str = ", ".join(metrics_parts)
            full_str = f"{base_str}, {metrics_str}"
        else:
            full_str = base_str
        
        formatted_strings.append(full_str)
    
    return formatted_strings


def create_edc_plots_mode2(real_rirs_wave: list, generated_rirs_wave: list, 
                          conditions: np.ndarray, rir_indices: list, sr: int, 
                          save_path: str, metrics: dict = None, octaves: list = None, 
                          title: str = "EDC Comparison - Mode 2") -> None:
    """
    Create EDC (Energy Decay Curve) plots per octave for Mode 2 comparison.
    
    Args:
        real_rirs_wave: List of real RIR waveforms from dataset
        generated_rirs_wave: List of generated RIR waveforms
        conditions: Conditioning parameters used
        rir_indices: Dataset indices used
        sr: Sample rate
        save_path: Directory to save plot
        metrics: Dictionary containing evaluation metrics (from evaluate_rir_quality)
        octaves: List of octave center frequencies in Hz (default: [125, 250, 500, 1000, 2000, 4000])
        title: Title for the plot
    """
    if octaves is None:
        octaves = [125, 250, 500, 1000, 2000, 4000]
    
    n_rirs = len(real_rirs_wave)
    n_octaves = len(octaves)
    
    # Generate formatted condition strings with metrics
    if metrics is not None:
        condition_strings = format_condition_strings_with_metrics(conditions, metrics, "room")
    else:
        # Fallback to original formatting if no metrics provided
        condition_strings = [format_condition_string(conditions[i], "room") for i in range(n_rirs)]
    
    # Create subplots: one row per RIR pair, one column per octave
    subplot_ratio = [3.5, 2.3]
    fig, axes = plt.subplots(n_rirs, n_octaves, figsize=(subplot_ratio[0]*n_octaves, subplot_ratio[1]*n_rirs))
    fig.suptitle(title, fontsize=20, y=0.98)
    
    # Handle single RIR case
    if n_rirs == 1:
        axes = axes.reshape(1, -1)
    # Handle single octave case
    if n_octaves == 1:
        axes = axes.reshape(-1, 1)
    # Handle single RIR and single octave case
    if n_rirs == 1 and n_octaves == 1:
        axes = np.array([[axes]])
    
    for i in range(n_rirs):
        real_rir = real_rirs_wave[i]
        gen_rir = generated_rirs_wave[i]
        idx = rir_indices[i]
        
        # Use the pre-formatted condition string with metrics
        condition_str = condition_strings[i]
        
        for j, octave_freq in enumerate(octaves):
            ax = axes[i, j] if n_octaves > 1 else axes[i]
            
            try:
                # Generate EDC for real RIR
                time_real, edc_real = gen_edc(real_rir, sr, octave_freq)
                
                # Generate EDC for generated RIR
                time_gen, edc_gen = gen_edc(gen_rir, sr, octave_freq)
                
                # Plot EDCs
                ax.plot(time_real, edc_real, '--', linewidth=2, color='green',
                       label=f'Real RIR #{idx}', alpha=0.8)
                ax.plot(time_gen, edc_gen, '-', linewidth=2, color='blue',
                       label=f'Generated RIR #{idx}', alpha=0.8)

                # Add -40 dB reference line
                ax.axhline(y=-40, color='red', linestyle='--', linewidth=1, alpha=0.6)

                # Formatting
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Energy Decay (dB)')
                ax.set_title(f'{octave_freq} Hz Octave')
                ax.grid(True, alpha=0.3)
                # ax.legend()

                # Dynamic y-axis limits
                y_min = min(edc_real.min(), edc_gen.min()) - 5
                y_max = max(edc_real.max(), edc_gen.max()) + 5
                y_min = max(y_min, -80)  # Don't go below -80 dB
                y_max = min(y_max, 10)   # Don't go above 10 dB
                ax.set_ylim([y_min, y_max])
                
                # Set reasonable y-axis limits (typically -60 to 0 dB for EDC)
                ax.set_ylim([-60, 5])
                
                # Limit x-axis to reasonable time range (e.g., first 2 seconds or RIR length)
                max_time = min(2.0, max(time_real[-1], time_gen[-1]))
                ax.set_xlim([0, max_time])
                
            except Exception as e:
                print(f"Warning: Failed to generate EDC for RIR {idx}, octave {octave_freq}Hz: {e}")
                ax.text(0.5, 0.5, f'EDC generation failed\n{octave_freq}Hz', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{octave_freq} Hz Octave (Error)')
        
        # Add condition info as a text annotation on the leftmost plot of each row
        if n_octaves > 0:
            left_ax = axes[i, 0] if n_octaves > 1 else axes[i]
            # Create a row title that spans across the row
            row_title = f'RIR #{idx} - {condition_str}'
            left_ax.text(-0.05, 1.25, row_title, transform=left_ax.transAxes, 
                        fontsize=11, fontweight='bold', ha='left', va='bottom')
            
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.3, hspace=0.6)

    # Save plot
    if save_path.endswith(".png"):
        plot_path = save_path
    else:
        plot_path = Path(save_path) / "edc_comparison_mode2.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"EDC comparison plot saved to: {plot_path}")


def format_condition_string(condition: np.ndarray, mode: str = "room") -> str:
    """Format condition parameters for display (legacy function for backward compatibility)."""
    if mode == "room":
        room_dims = condition[:3]
        rt60 = condition[-1]
        return f"Room: {room_dims[0]:.1f}×{room_dims[1]:.1f}×{room_dims[2]:.1f}m, RT60: {rt60:.2f}s"
    elif mode == "locations":
        mic_loc = condition[3:6]
        speaker_loc = condition[6:9]
        return f"Mic: ({mic_loc[0]:.1f},{mic_loc[1]:.1f},{mic_loc[2]:.1f}), Src: ({speaker_loc[0]:.1f},{speaker_loc[1]:.1f},{speaker_loc[2]:.1f})"