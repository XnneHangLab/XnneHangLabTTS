import io

import numpy as np
import soundfile as sf

from xnnehanglab_tts.webui.genie_tts import _wav_bytes_to_audio


def test_wav_bytes_to_audio_preserves_pcm16_dtype():
    samples = np.array([0, 1024, -1024, 32767, -32768], dtype=np.int16)
    buffer = io.BytesIO()
    sf.write(buffer, samples, 32000, format="WAV", subtype="PCM_16")

    sample_rate, data = _wav_bytes_to_audio(buffer.getvalue())

    assert sample_rate == 32000
    assert data.dtype == np.int16
    np.testing.assert_array_equal(data, samples)
