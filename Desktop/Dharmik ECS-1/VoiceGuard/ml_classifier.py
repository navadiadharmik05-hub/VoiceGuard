import os
import numpy as np
import torch
from typing import Dict, Any, Optional, Union

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Model candidate fallbacks
PRETRAINED_MODEL_CANDIDATES = [
    "mo-thecreator/Deepfake-audio-detection",
    "Siham/wav2vec2-base-deepfake-audio-detection",
    "Hemant-A/deepfake-audio-detection"
]

class MLVoiceClassifier:
    """
    Singleton Pre-trained Deepfake Audio Detection Classifier.
    Integrates Hugging Face transformers audio-classification pipeline for Wav2Vec2/WavLM models.
    Instantiated ONCE on server startup.
    """
    _instance: Optional["MLVoiceClassifier"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MLVoiceClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "mo-thecreator/Deepfake-audio-detection", device: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return
            
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.feature_extractor = None
        self.model = None
        self.is_loaded = False
        self._initialized = True
        
        self._load_pretrained_model()

    def _load_pretrained_model(self):
        if not HAS_TRANSFORMERS:
            print("[MLVoiceClassifier] HuggingFace transformers not installed. Operating in Hybrid DSP mode.")
            return
            
        candidates = [self.model_name] + [m for m in PRETRAINED_MODEL_CANDIDATES if m != self.model_name]
        for model_id in candidates:
            try:
                print(f"[MLVoiceClassifier] Instantiating Hugging Face model '{model_id}' on {self.device}...")
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
                self.model = AutoModelForAudioClassification.from_pretrained(model_id).to(self.device)
                self.model.eval()
                self.is_loaded = True
                self.model_name = model_id
                print(f"[MLVoiceClassifier] HuggingFace model '{model_id}' loaded successfully on {self.device}!")
                break
            except Exception as e:
                print(f"[MLVoiceClassifier] Could not load model '{model_id}': {e}")
                
        if not self.is_loaded:
            print("[MLVoiceClassifier] Operating in robust hybrid feature fallback mode.")

    def preprocess_audio(self, audio_input: Union[np.ndarray, str, bytes], sample_rate: int = 16000) -> np.ndarray:
        """
        Fast audio array/file preprocessing. Converts bytes, filepaths, or numpy arrays to 16kHz float32.
        """
        if isinstance(audio_input, str):
            if HAS_LIBROSA:
                y, _ = librosa.load(audio_input, sr=sample_rate, mono=True)
                return y.astype(np.float32)
            raise ValueError("librosa required to load audio filepath")
            
        if isinstance(audio_input, bytes):
            int16_data = np.frombuffer(audio_input, dtype=np.int16)
            return (int16_data.astype(np.float32) / 32768.0)
            
        arr = np.asarray(audio_input, dtype=np.float32)
        return arr

    def predict_frame(self, audio_input: Union[np.ndarray, str, bytes], sample_rate: int = 16000) -> float:
        """
        Evaluates raw in-memory audio buffers or file paths.
        Returns normalized synthetic deepfake probability score in [0.0, 100.0]%.
        Features robust fallback returning 50.0% neutral score on error.
        """
        try:
            audio = self.preprocess_audio(audio_input, sample_rate)
            if audio is None or len(audio) < 256:
                return 0.0
                
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms < 0.012:
                return 0.0

            # 1. HuggingFace Model Inference if available
            if self.is_loaded and self.model is not None and self.feature_extractor is not None:
                try:
                    inputs = self.feature_extractor(
                        audio, 
                        sampling_rate=sample_rate, 
                        return_tensors="pt", 
                        padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        logits = self.model(**inputs).logits
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                        # Class index mapping (fake probability)
                        fake_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                        return round(fake_prob * 100.0, 2)
                except Exception as e:
                    print(f"[MLVoiceClassifier] Inference warning: {e}")
                    # Fallback to feature engine below

            # 2. Fast Acoustic Feature Heuristic Classifier
            return self._feature_based_ml_score(audio, sample_rate)
        except Exception as err:
            print(f"[MLVoiceClassifier] Error encountered: {err}. Returning neutral fallback 50.0%")
            return 50.0

    def _feature_based_ml_score(self, audio: np.ndarray, sample_rate: int) -> float:
        import scipy.signal as signal
        
        rms = float(np.sqrt(np.mean(audio**2)))
        n_fft = 512
        hop = 128
        f, t_spec, Zxx = signal.stft(audio, fs=sample_rate, nperseg=n_fft, noverlap=n_fft-hop)
        mag = np.abs(Zxx) + 1e-9
        
        hf_bins = f >= 6000
        hf_energy = float(np.mean(mag[hf_bins, :])) if np.any(hf_bins) else 0.0
        lf_energy = float(np.mean(mag[f < 4000, :])) + 1e-9
        hf_ratio = hf_energy / lf_energy
        
        angles = np.angle(Zxx)
        phase_diff = np.diff(angles, axis=1)
        phase_var = float(np.var(np.angle(np.exp(1j * phase_diff)))) if phase_diff.size > 0 else 0.0
        
        geom_mean = np.exp(np.mean(np.log(mag), axis=0))
        arith_mean = np.mean(mag, axis=0)
        flatness = float(np.mean(geom_mean / arith_mean))
        
        flat_score = min(1.0, max(0.0, (flatness - 0.02) / 0.06))
        phase_score = min(1.0, max(0.0, (phase_var - 1.8) / 3.0))
        hf_score = min(1.0, max(0.0, (hf_ratio - 0.05) / 0.15))
        
        combined_score = (0.45 * flat_score + 0.35 * phase_score + 0.20 * hf_score) * 100.0
        return round(float(np.clip(combined_score, 0.0, 100.0)), 2)
