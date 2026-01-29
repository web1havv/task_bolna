"""
GPU-accelerated TTS module using transformers-based TTS for fast audio generation.
This replaces gTTS for much faster generation on GPU.
Uses SpeechT5 from `transformers` and avoids any deprecated dataset scripts.
"""
import torch
from typing import Optional
from pydub import AudioSegment
import numpy as np
from rich import print


class GPUTTS:
    """GPU-accelerated TTS using transformers-based models"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize GPU TTS model.
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        try:
            from transformers import (
                SpeechT5Processor,
                SpeechT5ForTextToSpeech,
                SpeechT5HifiGan,
            )
        except ImportError:
            raise ImportError(
                "transformers required. Install with: pip install transformers"
            )
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"[bold]Initializing GPU TTS on device: {device}[/bold]")
        
        try:
            # Load SpeechT5 model (fast and good quality)
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained(
                "microsoft/speecht5_tts"
            ).to(device)
            self.vocoder = SpeechT5HifiGan.from_pretrained(
                "microsoft/speecht5_hifigan"
            ).to(device)
            self.model.eval()
            self.vocoder.eval()

            # Create a fixed dummy speaker embedding instead of loading a dataset
            speaker_dim = getattr(self.model.config, "speaker_embedding_dim", 512)
            self.speaker_embeddings = torch.zeros(1, speaker_dim, device=device)
            
            print("[green]Loaded TTS model: SpeechT5 (GPU-accelerated)[/green]")
        except Exception as e:
            print(f"[red]Failed to load SpeechT5: {e}[/red]")
            raise
    
    def synthesize(self, text: str, speaker_wav: Optional[str] = None) -> Optional[AudioSegment]:
        """
        Synthesize speech from text using GPU.
        
        Args:
            text: Text to synthesize
            speaker_wav: Not used (kept for compatibility)
        
        Returns:
            AudioSegment or None if synthesis fails
        """
        try:
            # Process text
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            
            # Generate speech
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    self.speaker_embeddings,
                    vocoder=self.vocoder
                )
            
            # Convert to numpy and normalize
            audio_np = speech.cpu().numpy()
            
            # Normalize to int16 range
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            # Convert to AudioSegment (16kHz, mono)
            audio = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=16000,
                sample_width=2,
                channels=1
            )
            
            return audio
            
        except Exception as e:
            print(f"[red]GPU TTS synthesis failed: {e}[/red]")
            return None


# Global TTS instance (lazy initialization)
_tts_instance: Optional[GPUTTS] = None


def get_tts(device: Optional[str] = None) -> GPUTTS:
    """Get or create global TTS instance"""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = GPUTTS(device=device)
    return _tts_instance


def synthesize_gpu_tts(text: str, device: Optional[str] = None) -> Optional[AudioSegment]:
    """
    Synthesize speech using GPU TTS (drop-in replacement for synthesize_gtts).
    
    Args:
        text: Text to synthesize
        device: Optional device ('cuda', 'cpu', or None for auto)
    
    Returns:
        AudioSegment or None if synthesis fails
    """
    tts = get_tts(device=device)
    return tts.synthesize(text)
