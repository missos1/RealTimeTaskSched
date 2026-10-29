import os
import sys
from pathlib import Path
from typing import Union
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

#  Register CUDA/cuDNN DLL directories for Windows
# Fixes Error 126 (cublasLt64_12.dll missing) by ensuring the C++ dynamic loader
# finds the conda nvidia packages before importing sherpa_onnx.
if sys.platform == "win32":
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    for pkg in ["cublas", "cudnn", "cuda_nvrtc"]:
        dll_dir = site_packages / "nvidia" / pkg / "bin"
        if dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ["PATH"]

    torch_lib = site_packages / "torch" / "lib"
    if torch_lib.is_dir():
        os.add_dll_directory(str(torch_lib))
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ["PATH"]

import sherpa_onnx

# ── 2. ASR Class Definition ─────────────────────────────────────────────────
class GipformerASRGPU:
    def __init__(
        self,
        hf_token,
        version: str = "1.5",
        quantize: str = "fp32",  # Standard FP32 is recommended for CUDA
        num_threads: int = 2,
        decoding_method: str = "modified_beam_search", 
    ):

        repo_id = f"g-group-ai-lab/gipformer{'' if version == '1' else '1.5'}-68M-rnnt"
        suffix = ".int8.onnx" if quantize == "int8" else ".onnx"

        # Resolve or download model files via huggingface_hub
        encoder_path = hf_hub_download(repo_id=repo_id, filename=f"encoder{suffix}", token=hf_token)
        decoder_path = hf_hub_download(repo_id=repo_id, filename=f"decoder{suffix}", token=hf_token)
        joiner_path = hf_hub_download(repo_id=repo_id, filename=f"joiner{suffix}", token=hf_token)
        tokens_path = hf_hub_download(repo_id=repo_id, filename="tokens.txt", token=hf_token)
        
        # Initialize recognizer directly using the transducer factory
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            tokens=tokens_path,
            sample_rate=16000,
            feature_dim=80,
            num_threads=num_threads,
            decoding_method=decoding_method,
            provider="cuda",
        )

        active_provider = getattr(
            self.recognizer.config.model_config, "provider", "unknown"
        )
        print(f"ASR Initialized successfully. Active Provider: {active_provider}")

    def _prepare_audio(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        target_sr: int = 16000,
    ) -> tuple[np.ndarray, int]:
        """Convert arbitrary audio to 16kHz float32 mono array."""
        if isinstance(audio_input, (str, Path)):
            samples, sr = sf.read(str(audio_input), dtype="float32")
        elif isinstance(audio_input, bytes):
            import io
            samples, sr = sf.read(io.BytesIO(audio_input), dtype="float32")
        elif isinstance(audio_input, np.ndarray):
            samples = audio_input.astype(np.float32)
            sr = target_sr
        else:
            raise TypeError(f"Unsupported audio type: {type(audio_input)}")

        # Convert stereo to mono
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        return samples, sr

    def transcribe(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        sample_rate: int = 16000,
    ) -> str:
        samples, sr = self._prepare_audio(audio_input, sample_rate)

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sr, samples)
        self.recognizer.decode_streams([stream])

        return stream.result.text.strip()

    def transcribe_timed(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        sample_rate: int = 16000,
        offset: float = 0.0,
        gap: float = 0.8,
    ) -> list[dict]:
        """Return [{'start': s, 'end': s, 'text': str}] segments.

        Uses sherpa-onnx per-token timestamps; a new segment starts after a
        pause longer than `gap` seconds. `offset` shifts times so chunks of a
        live stream get meeting-absolute timestamps.
        """
        samples, sr = self._prepare_audio(audio_input, sample_rate)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sr, samples)
        self.recognizer.decode_streams([stream])
        res = stream.result

        segments: list[dict] = []
        for tok, t in zip(res.tokens, res.timestamps):
            t += offset
            if not segments or t - segments[-1]["end"] > gap:
                segments.append({"start": t, "end": t, "text": ""})
            segments[-1]["text"] += tok.replace("\u2581", " ")
            segments[-1]["end"] = t
        for s in segments:
            s["text"] = s["text"].strip()
        return segments

    @staticmethod
    def format_timed(segments: list[dict]) -> str:
        """Render segments as '[mm:ss] text' lines for the LLM."""
        return "\n".join(
            f"[{int(s['start']) // 60:02d}:{int(s['start']) % 60:02d}] {s['text']}"
            for s in segments
        )


# ── 3. Execution Verification ───────────────────────────────────────────────
if __name__ == "__main__":
    load_dotenv()
    
    hf_token = os.getenv("HF_TOKEN")
    # print(f"Hf token: {hf_token}\n")
    
    asr = GipformerASRGPU(hf_token, version="1.5", quantize="fp32")
    result = asr.transcribe_timed("test.mp3")
    transcript_timed = GipformerASRGPU.format_timed(result) 
    print(f"Transcription: {transcript_timed}")
    
    