import torchaudio
import soundfile as sf
import numpy as np
import torch
import logging


def _safe_sr_frames(path: str):
    try:
        info = sf.info(path)
        return info.samplerate, info.frames
    except sf.LibsndfileError:
        # soundfile can't read some formats (e.g. mp3); fall back to torchaudio,
        # which uses the same ffmpeg/soundfile backends without needing torchcodec.
        info = torchaudio.info(path)
        return info.sample_rate, info.num_frames


def load_audio_chunk(path, target_n_samples, target_sr, start=None, verbose=False):
    """
    Load a chunk of audio from a file using torchaudio.
    
    Args:
        path: Path to audio file
        target_n_samples: Number of samples to load (at target_sr)
        target_sr: Target sample rate
        start: Starting frame (at original sr). If None, random start.
        verbose: Print debug info
    
    Returns:
        audio: Tensor of shape [n_samples, n_channels]
    """
    # Get audio metadata using soundfile
    sr, frames = _safe_sr_frames(path)

    print(f"length of audio in seconds: {frames/sr}") if verbose else None
    print(f"Original sample rate: {sr}") if verbose else None

    # Adjust for MP3 padding
    if path.split(".")[-1].lower() == "mp3":
        frames = frames - 8192

    # Calculate how many frames to load at original sr
    new_target_n_samples = int(target_n_samples * sr / target_sr)

    print(f"New target n samples: {new_target_n_samples}") if verbose else None

    # Random start if not specified
    if start is None:
        max_start = max(1, frames - new_target_n_samples)
        start = np.random.randint(0, max_start)

    # Load audio chunk with torchaudio
    # torchaudio.load returns (waveform, sample_rate) where waveform is [channels, samples]
    audio, loaded_sr = torchaudio.load(
        path,
        frame_offset=start,
        num_frames=new_target_n_samples,
        normalize=True,
    )

    # Resample if needed
    if loaded_sr != target_sr:
        audio = torchaudio.functional.resample(audio, loaded_sr, target_sr)
        print(f"Resampled to {target_sr}, shape of audio: {audio.shape}") if verbose else None

    # pad if needed
    if audio.shape[1] < target_n_samples:
        audio = torch.nn.functional.pad(audio, (0, target_n_samples - audio.shape[1]))

    # Convert from [channels, samples] to [samples, channels]
    audio = audio.T

    return audio


def load_full_audio(path, target_sr, verbose=False):
    """
    Load full audio file using torchaudio.
    
    Args:
        path: Path to audio file
        target_sr: Target sample rate
        verbose: Print debug info
    
    Returns:
        audio: Tensor of shape [n_channels, n_samples]
    """
    # Load with torchaudio - returns [channels, samples]
    audio, sr = torchaudio.load(path, normalize=True)

    # If stereo, average to mono
    if audio.shape[0] == 2:
        audio = audio.mean(dim=0, keepdim=True)


    # Resample if needed
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)


    return audio


def load_full_and_split(path, target_sr, target_n_samples, hop=None, verbose=False):
    """
    Load full audio and split into overlapping chunks.
    
    Args:
        path: Path to audio file
        target_sr: Target sample rate
        target_n_samples: Samples per chunk
        hop: Hop size between chunks (default: target_n_samples)
        verbose: Print debug info
    
    Returns:
        audio: Tensor of shape [n_chunks, 1, target_n_samples]
    """
    hop = target_n_samples if hop is None else hop
    audio = load_full_audio(path, target_sr, verbose=verbose)
    audio = audio.squeeze()


    # If audio is shorter than target, repeat
    if audio.shape[0] < target_n_samples:
        n_repeats = int(np.ceil(target_n_samples / audio.shape[0]))
        audio = audio.repeat(n_repeats)

    audio = audio.unfold(0, int(target_n_samples), int(hop)).unsqueeze(1)

    return audio
