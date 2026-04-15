"""PaddleSpeech 声纹包装层测试。"""

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from robot_auditory import config
from robot_auditory.voiceprint import VoiceprintVerifier, cosine_similarity


class FakeBackend:
    name = "paddlespeech-ecapatdnn"

    def embed_raw(self, raw: bytes, sample_rate: int):  # noqa: ARG002
        if not raw:
            return None
        vec = np.frombuffer(raw, dtype=np.float32)
        if vec.size == 0:
            return None
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            return None
        return vec / norm


def _raw(*values: float) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


class TestVoiceprint(unittest.TestCase):
    def test_similarity(self) -> None:
        a = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.asarray([0.99, 0.01, 0.0], dtype=np.float32)
        c = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        self.assertGreater(cosine_similarity(a, b), 0.99)
        self.assertLess(cosine_similarity(a, c), 0.1)

    def test_verifier_enroll_and_verify(self) -> None:
        enrolled = [
            _raw(1.0, 0.00, 0.00, 0.00),
            _raw(0.99, 0.02, 0.00, 0.00),
            _raw(0.98, 0.01, 0.01, 0.00),
            _raw(1.01, -0.01, 0.00, 0.00),
            _raw(1.00, 0.00, 0.02, 0.00),
        ]
        intruder = _raw(0.0, 1.0, 0.0, 0.0)

        with tempfile.TemporaryDirectory() as td:
            verifier = VoiceprintVerifier(Path(td) / "voiceprint.json", backend=FakeBackend())
            verifier.enroll_from_raw_samples(enrolled)
            matched_good, _ = verifier.verify_raw(enrolled[0])
            matched_bad, _ = verifier.verify_raw(intruder)

        self.assertTrue(matched_good)
        self.assertFalse(matched_bad)

    def test_old_profile_should_require_reenroll(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "voiceprint.json"
            path.write_text(
                json.dumps({"version": 1, "sample_rate": 16000, "vector": [0.1, 0.2]}),
                encoding="utf-8",
            )
            verifier = VoiceprintVerifier(path, backend=FakeBackend())
            self.assertFalse(verifier.enrolled)
            self.assertTrue(verifier.needs_reenroll)

    def test_profile_uses_new_model_version(self) -> None:
        enrolled = [
            _raw(1.0, 0.00, 0.00, 0.00),
            _raw(0.99, 0.02, 0.00, 0.00),
            _raw(0.98, 0.01, 0.01, 0.00),
            _raw(1.01, -0.01, 0.00, 0.00),
            _raw(1.00, 0.00, 0.02, 0.00),
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "voiceprint.json"
            verifier = VoiceprintVerifier(path, backend=FakeBackend())
            profile = verifier.enroll_from_raw_samples(enrolled)
            self.assertEqual(profile.version, config.VOICEPRINT_MODEL_VERSION)
            self.assertEqual(profile.backend, "paddlespeech-ecapatdnn")
            self.assertEqual(profile.embedding_dim, 4)


if __name__ == "__main__":
    unittest.main()
