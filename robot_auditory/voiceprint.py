"""基于 PaddleSpeech ECAPA-TDNN 的本地声纹录入与比对。"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol

import numpy as np

from . import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceprintProfile:
    version: int
    sample_rate: int
    backend: str
    embedding_dim: int
    centroid: list[float]
    spread: list[float]
    samples: list[list[float]]
    centroid_cosine_floor: float
    nearest_cosine_floor: float
    distance_cap: float


class EmbeddingBackend(Protocol):
    name: str

    def embed_raw(self, raw: bytes, sample_rate: int) -> Optional[np.ndarray]:
        ...

    def warmup(self) -> None:
        ...


def _waveform_from_pcm(raw: bytes) -> np.ndarray:
    if not raw:
        return np.array([], dtype=np.float32)
    wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if wav.size == 0:
        return np.array([], dtype=np.float32)
    return wav / 32768.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


def _standard_distance(vec: np.ndarray, centroid: np.ndarray, spread: np.ndarray) -> float:
    z = (vec - centroid) / np.maximum(spread, 1e-3)
    return float(np.sqrt(np.mean(z * z)))


class PaddleSpeechEmbeddingBackend:
    """使用官方 PaddleSpeech ECAPA-TDNN 提取说话人嵌入。"""

    name = "paddlespeech-ecapatdnn"

    def __init__(self) -> None:
        self._executor = None

    def _load(self):
        if self._executor is not None:
            return self._executor

        cache_dir = config.VOICEPRINT_MODEL_CACHE_DIR.resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PPSPEECH_HOME", str(cache_dir))

        try:
            from paddlespeech.cli.vector.infer import VectorExecutor
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "PaddleSpeech 未安装。请在项目虚拟环境中安装 paddlespeech / paddlepaddle。"
            ) from exc

        logger.info(
            "正在加载 PaddleSpeech 声纹模型：%s（首次运行会下载到 %s）",
            config.VOICEPRINT_MODEL_NAME,
            cache_dir,
        )
        self._executor = VectorExecutor()
        return self._executor

    def _write_temp_wav(self, raw: bytes, sample_rate: int) -> Optional[str]:
        waveform = _waveform_from_pcm(raw)
        if waveform.size < sample_rate:
            return None

        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak <= 1e-6:
            return None

        pcm = np.clip(waveform, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = handle.name

        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return temp_path

    def embed_waveform(self, waveform: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        if waveform.size < sample_rate:
            return None
        raw = np.clip(waveform, -1.0, 1.0)
        pcm = (raw * 32767.0).astype(np.int16).tobytes()
        return self.embed_raw(pcm, sample_rate=sample_rate)

    def embed_raw(self, raw: bytes, sample_rate: int) -> Optional[np.ndarray]:
        audio_path = self._write_temp_wav(raw, sample_rate)
        if not audio_path:
            return None

        try:
            executor = self._load()
            vec = executor(
                audio_file=audio_path,
                model=config.VOICEPRINT_MODEL_NAME,
                sample_rate=sample_rate,
                force_yes=True,
                device="cpu",
            )
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-6:
            return None
        return arr / norm

    def warmup(self) -> None:
        executor = self._load()
        executor._init_from_path(
            model_type=config.VOICEPRINT_MODEL_NAME,
            sample_rate=config.VOICEPRINT_SAMPLE_RATE,
            cfg_path=None,
            ckpt_path=None,
        )


class VoiceprintVerifier:
    def __init__(
        self,
        path: Path | str = config.VOICEPRINT_PATH,
        backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        self._path = Path(path)
        self._backend = backend or PaddleSpeechEmbeddingBackend()
        self._profile: Optional[VoiceprintProfile] = None
        self._needs_reenroll = False
        self._warmed_up = False
        self._last_verify_score: Optional[float] = None
        self._last_verify_passed: Optional[bool] = None
        self.load()

    @property
    def enrolled(self) -> bool:
        return self._profile is not None

    @property
    def needs_reenroll(self) -> bool:
        return self._needs_reenroll or self._profile is None

    @property
    def warmed_up(self) -> bool:
        return self._warmed_up

    @property
    def last_verify_score(self) -> Optional[float]:
        return self._last_verify_score

    @property
    def last_verify_passed(self) -> Optional[bool]:
        return self._last_verify_passed

    def load(self) -> None:
        self._needs_reenroll = False
        if not self._path.exists():
            self._profile = None
            return

        payload = json.loads(self._path.read_text(encoding="utf-8"))
        version = int(payload.get("version", 1))
        backend_name = payload.get("backend", "")
        if version != config.VOICEPRINT_MODEL_VERSION or backend_name != self._backend.name:
            logger.info(
                "检测到旧版或不同后端的声纹模型(version=%s, backend=%s)，需要重新录入。",
                version,
                backend_name,
            )
            self._profile = None
            self._needs_reenroll = True
            return

        self._profile = VoiceprintProfile(
            version=version,
            sample_rate=int(payload["sample_rate"]),
            backend=str(payload["backend"]),
            embedding_dim=int(payload["embedding_dim"]),
            centroid=[float(v) for v in payload["centroid"]],
            spread=[float(v) for v in payload["spread"]],
            samples=[[float(v) for v in sample] for sample in payload["samples"]],
            centroid_cosine_floor=float(payload["centroid_cosine_floor"]),
            nearest_cosine_floor=float(payload["nearest_cosine_floor"]),
            distance_cap=float(payload["distance_cap"]),
        )

    def warmup(self) -> None:
        logger.info("开始预热 PaddleSpeech 声纹模型。")
        self._backend.warmup()
        self._warmed_up = True
        logger.info("PaddleSpeech 声纹模型预热完成。")

    def reset_profile(self, delete_file: bool = True) -> None:
        if delete_file and self._path.exists():
            self._path.unlink()
        self._profile = None
        self._needs_reenroll = True
        self._last_verify_score = None
        self._last_verify_passed = None

    def save_profile(self, profile: VoiceprintProfile) -> None:
        self._path.write_text(
            json.dumps(
                {
                    "version": profile.version,
                    "sample_rate": profile.sample_rate,
                    "backend": profile.backend,
                    "embedding_dim": profile.embedding_dim,
                    "centroid": profile.centroid,
                    "spread": profile.spread,
                    "samples": profile.samples,
                    "centroid_cosine_floor": profile.centroid_cosine_floor,
                    "nearest_cosine_floor": profile.nearest_cosine_floor,
                    "distance_cap": profile.distance_cap,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._profile = profile
        self._needs_reenroll = False

    def enroll_from_raw_samples(
        self,
        raw_samples: Iterable[bytes],
        sample_rate: int = config.VOICEPRINT_SAMPLE_RATE,
    ) -> VoiceprintProfile:
        vectors = []
        for raw in raw_samples:
            vec = self._backend.embed_raw(raw, sample_rate=sample_rate)
            if vec is not None:
                vectors.append(vec.astype(np.float32))
        if len(vectors) < config.VOICEPRINT_MIN_VALID_SAMPLES:
            raise ValueError("有效声纹样本不足，请重新录入。")

        stack = np.stack(vectors)
        centroid = np.mean(stack, axis=0)
        centroid /= float(np.linalg.norm(centroid))
        spread = np.std(stack, axis=0) + 1e-3

        centroid_cos = [cosine_similarity(v, centroid) for v in stack]
        nearest_cos = []
        dist_vals = []
        for i, vec in enumerate(stack):
            others = [cosine_similarity(vec, other) for j, other in enumerate(stack) if j != i]
            nearest_cos.append(max(others) if others else 1.0)
            dist_vals.append(_standard_distance(vec, centroid, spread))

        profile = VoiceprintProfile(
            version=config.VOICEPRINT_MODEL_VERSION,
            sample_rate=sample_rate,
            backend=self._backend.name,
            embedding_dim=int(stack.shape[1]),
            centroid=centroid.astype(float).tolist(),
            spread=spread.astype(float).tolist(),
            samples=stack.astype(float).tolist(),
            centroid_cosine_floor=max(
                config.VOICEPRINT_MIN_CENTROID_COSINE,
                float(min(centroid_cos) - 0.01),
            ),
            nearest_cosine_floor=max(
                config.VOICEPRINT_MIN_NEAREST_COSINE,
                float(min(nearest_cos) - 0.01),
            ),
            distance_cap=min(
                config.VOICEPRINT_MAX_STANDARD_DISTANCE,
                float(max(dist_vals) + 0.12),
            ),
        )
        self.save_profile(profile)
        logger.info(
            "PaddleSpeech 声纹模型训练完成并保存到 %s；样本数=%d, embedding_dim=%d",
            self._path,
            len(vectors),
            profile.embedding_dim,
        )
        return profile

    def verify_raw(
        self,
        raw: bytes,
        sample_rate: int = config.VOICEPRINT_SAMPLE_RATE,
    ) -> tuple[bool, float]:
        if self._profile is None:
            return True, 1.0

        candidate = self._backend.embed_raw(raw, sample_rate=sample_rate)
        if candidate is None:
            return False, 0.0

        centroid = np.asarray(self._profile.centroid, dtype=np.float32)
        spread = np.asarray(self._profile.spread, dtype=np.float32)
        sample_bank = np.asarray(self._profile.samples, dtype=np.float32)

        centroid_cos = cosine_similarity(centroid, candidate)
        nearest_cos = max(cosine_similarity(sample, candidate) for sample in sample_bank)
        distance = _standard_distance(candidate, centroid, spread)

        score = 0.45 * centroid_cos + 0.45 * nearest_cos + 0.10 * max(0.0, 1.0 - distance / 4.0)
        matched = score >= config.VOICEPRINT_MATCH_THRESHOLD
        self._last_verify_score = score
        self._last_verify_passed = matched
        logger.info(
            "PaddleSpeech 声纹打分：matched=%s, centroid_cos=%.3f, nearest_cos=%.3f, distance=%.3f, score=%.3f, threshold=%.2f",
            matched,
            centroid_cos,
            nearest_cos,
            distance,
            score,
            config.VOICEPRINT_MATCH_THRESHOLD,
        )
        return matched, score
