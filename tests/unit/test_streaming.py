# -*- coding: utf-8 -*-

"""
Tests that the TSM procedures produce the same output regardless of the
chunking of the input stream.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from audiotsm import ola, wsola, phasevocoder
from audiotsm.io.array import ArrayReader, ArrayWriter


class ChunkedArrayReader(ArrayReader):
    """An :class:`ArrayReader` that returns at most ``chunk`` samples per
    :func:`read` call, mimicking a real-time stream from which only a limited
    amount of data is available at a time."""
    def __init__(self, data, chunk):
        super().__init__(data)
        self._chunk = chunk

    def read(self, buffer):
        n = min(buffer.shape[1], self._chunk)
        return super().read(buffer[:, :n])


def generate_signal(channels=2, length=8192, samplerate=44100):
    """Generate a fixed test signal."""
    t = np.arange(length) / samplerate
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) +
              0.3 * np.sin(2 * np.pi * 881 * t + 0.5) +
              0.2 * np.sin(2 * np.pi * 1237 * t + 1.0))
    signal *= 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    return np.tile(signal, (channels, 1))


def run_tsm(tsm, reader, channels):
    """Run a TSM procedure on the content of ``reader`` and return the
    output."""
    writer = ArrayWriter(channels)
    tsm.run(reader, writer)
    return writer.data


@pytest.mark.parametrize("tsm", [
    ola(2, frame_length=256),
    wsola(2, frame_length=256),
    phasevocoder(2, frame_length=256),
], ids=["ola", "wsola", "phasevocoder"])
@pytest.mark.parametrize("speed", [0.75, 1.5])
@pytest.mark.parametrize("chunk", [1, 7, 100, 256, 1000])
def test_chunk_size_invariance(tsm, speed, chunk):
    """The output of the TSM procedures should not depend on the chunking of
    the input stream: feeding the signal through a reader that only returns
    ``chunk`` samples at a time (as real-time stream readers do) should give
    the same result as reading the whole signal at once."""
    channels = 2
    signal = generate_signal(channels)

    tsm.set_speed(speed)
    expected = run_tsm(tsm, ArrayReader(signal.copy()), channels)

    tsm.set_speed(speed)
    actual = run_tsm(tsm, ChunkedArrayReader(signal.copy(), chunk), channels)

    assert actual.shape == expected.shape
    assert_allclose(actual, expected, atol=1e-6)
