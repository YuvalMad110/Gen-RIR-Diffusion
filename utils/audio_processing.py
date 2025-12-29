"""
Audio processing utilities for RIR inference.
Handles speech loading, convolution, and audio file operations.

Functions list:
- find_speech_files: Locate speech audio files in a directory.
- load_speech: Load and preprocess speech audio files.
- convolve_with_rir: Convolve speech signals with RIRs.
- process_speech_convolution: Process speech files and convolve with RIRs.
"""

import os
import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.signal import convolve
from tqdm import tqdm


def find_speech_files(speech_path: str, speech_ids: Optional[List[str]] = None,
                      n_files: int = 3, extensions: Tuple[str, ...] = ('.wav', '.flac')) -> List[str]:
    """Find speech audio files in a directory."""
    speech_path = Path(speech_path)
    
    if speech_ids is not None:
        found_files = []
        for sid in speech_ids:
            for ext in extensions:
                target = speech_path / f"{sid}{ext}"
                if target.exists():
                    found_files.append(str(target))
                    print(f"Found: {target}")
                    break
            else:
                print(f"Not found: {sid}")
        return found_files
    
    # Random selection
    print(f"Searching for {n_files} random audio files...")
    candidates = []
    
    for root, _, files in os.walk(speech_path):
        for file in files:
            if file.lower().endswith(extensions):
                candidates.append(os.path.join(root, file))
                if len(candidates) >= n_files * 10:
                    break
        if len(candidates) >= n_files * 10:
            break
    
    if not candidates:
        print(f"No audio files found in {speech_path}")
        return []
    
    selected = random.sample(candidates, min(n_files, len(candidates)))
    print(f"Found {len(candidates)} files, selected {len(selected)}")
    return selected


def load_speech(file_path: str, target_sr: int, max_duration: float = 10.0,
                normalize: bool = True) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Load and preprocess a speech file."""
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        max_samples = int(max_duration * target_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        if normalize and np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio, target_sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def convolve_with_rir(speech: np.ndarray, rir: np.ndarray,
                      normalize_rir: bool = True, normalize_output: bool = True) -> np.ndarray:
    """Convolve speech signal with RIR."""
    speech = speech.squeeze()
    rir = rir.squeeze()
    
    if normalize_rir and np.max(np.abs(rir)) > 0:
        rir = rir / np.max(np.abs(rir))
    
    reverb = convolve(speech, rir, mode='full')
    
    if normalize_output and np.max(np.abs(reverb)) > 0:
        reverb = reverb / np.max(np.abs(reverb)) * 0.95
    
    return reverb


def process_speech_convolution(speech_path: str, generated_rirs: List[np.ndarray],
                               real_rirs: Optional[List[np.ndarray]], save_path: str, sr: int,
                               speech_ids: Optional[List[str]] = None, n_speech_files: int = 3,
                               normalize_rirs: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Process speech files and convolve with RIRs."""
    print("\n=== Processing Speech Convolution ===")
    
    speech_files = find_speech_files(speech_path, speech_ids, n_speech_files, ['.wav'])
    if not speech_files:
        print("No speech files found!")
        return [], []
    
    print(f"Processing {len(speech_files)} speech files with {len(generated_rirs)} RIRs")
    
    # Create output directories
    output_dir = Path(save_path) / "convolved_speech"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clean_dir = output_dir / "clean"
    gen_dir = output_dir / "generated_rir"
    real_dir = output_dir / "real_rir" if real_rirs else None
    
    clean_dir.mkdir(exist_ok=True)
    gen_dir.mkdir(exist_ok=True)
    if real_dir:
        real_dir.mkdir(exist_ok=True)
    
    gen_reverb_list = []
    real_reverb_list = []
    
    # Optionally normalize RIRs
    if normalize_rirs:
        generated_rirs = [_normalize(r) for r in generated_rirs]
        if real_rirs:
            real_rirs = [_normalize(r) for r in real_rirs]
    
    for speech_idx, speech_file in enumerate(tqdm(speech_files, desc="Processing speech")):
        speech, _ = load_speech(speech_file, target_sr=sr)
        if speech is None:
            continue
        
        file_name = Path(speech_file).stem
        sf.write(clean_dir / f"clean_{speech_idx:02d}_{file_name}.wav", speech, sr)
        
        # Convolve with generated RIRs
        for rir_idx, gen_rir in enumerate(generated_rirs):
            try:
                reverb = convolve_with_rir(speech, gen_rir, normalize_rir=False)
                sf.write(gen_dir / f"gen_rir_{rir_idx:02d}_speech_{speech_idx:02d}_{file_name}.wav", reverb, sr)
                gen_reverb_list.append(reverb)
            except Exception as e:
                print(f"Error with generated RIR {rir_idx}: {e}")
        
        # Convolve with real RIRs
        if real_rirs:
            for rir_idx, real_rir in enumerate(real_rirs):
                try:
                    reverb = convolve_with_rir(speech, real_rir, normalize_rir=False)
                    sf.write(real_dir / f"real_rir_{rir_idx:02d}_speech_{speech_idx:02d}_{file_name}.wav", reverb, sr)
                    real_reverb_list.append(reverb)
                except Exception as e:
                    print(f"Error with real RIR {rir_idx}: {e}")
    
    print(f"Speech convolution completed!")
    print(f"  Clean: {clean_dir}")
    print(f"  Generated: {gen_dir}")
    if real_dir:
        print(f"  Real: {real_dir}")
    
    return gen_reverb_list, real_reverb_list


def _normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize signal to unit peak amplitude."""
    peak = np.max(np.abs(signal))
    return signal / peak if peak > 0 else signal


def save_audio_files(rirs: List[np.ndarray], save_path: str, sr: int, prefix: str = "generated"):
    """Save RIRs as audio files."""
    audio_dir = Path(save_path) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    for i, rir in enumerate(rirs):
        filename = audio_dir / f"{prefix}_rir_{i+1:03d}.wav"
        sf.write(filename, rir, sr)
    
    print(f"{prefix.capitalize()} audio files saved to: {audio_dir}")
