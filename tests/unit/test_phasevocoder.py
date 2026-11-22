# -*- coding: utf-8 -*-

"""
Tests for the audiotsm.phasevocoder package.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from audiotsm import phasevocoder, PhaseLocking
from audiotsm.io.array import ArrayReader, ArrayWriter
from audiotsm.phasevocoder import find_peaks, get_closest_peaks


@pytest.mark.parametrize("amplitude, peaks", [
    ([], []),
    ([0], [True]),
    ([1], [True]),
    ([1, 0, 0, 1, 0, 0, 0, 1, 0],
     [True, False, False, True, False, False, False, True, False]),
    ([0, 1, 2, 3, 1, 2, 4, 5, -1, 6, 0],
     [False, False, False, True, False, False, False, False, False, True,
      False]),
])
def test_find_peaks(amplitude, peaks):
    """Run tests for the find_peaks function."""
    actual_peaks = find_peaks(np.array(amplitude))
    assert_array_equal(actual_peaks, np.array(peaks, dtype=bool))


@pytest.mark.parametrize("peaks, closest_peak", [
    ([], []),
    ([True], [0]),
    ([True], [0]),
    ([True, False, False, True, False, False, False, True, False],
     [0, 0, 3, 3, 3, 3, 7, 7, 7]),
    ([True, True, True, True, True, True, True, True, True],
     [0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ([False, False, False, True, False, False, False, False, False, True,
      False],
     [3, 3, 3, 3, 3, 3, 3, 9, 9, 9, 9]),
])
def test_get_closest_peaks(peaks, closest_peak):
    """Run tests for the get_closest_peaks."""
    actual_closest_peak = get_closest_peaks(np.array(peaks))
    assert_array_equal(actual_closest_peak, np.array(closest_peak, dtype=int))


# Fixed stereo sample used by the multi-channel tests below: the right
# channel is a delayed and attenuated copy of the left channel, so the
# inter-channel delay and the stereo image are known exactly.
SAMPLE_LENGTH = 44100
INTER_CHANNEL_DELAY = 13


def make_stereo_sample():
    """Generate a fixed stereo sample with a known inter-channel delay."""
    random = np.random.RandomState(34585)
    left = random.uniform(-0.5, 0.5, SAMPLE_LENGTH)
    right = 0.8 * np.concatenate(
        (np.zeros(INTER_CHANNEL_DELAY), left[:-INTER_CHANNEL_DELAY]))
    return np.vstack((left, right))


def run_phasevocoder(data, speed, phase_locking=PhaseLocking.IDENTITY):
    """Run the phase vocoder on a numpy array and return the output."""
    tsm = phasevocoder(data.shape[0], speed=speed, phase_locking=phase_locking)
    reader = ArrayReader(data)
    writer = ArrayWriter(data.shape[0])
    tsm.run(reader, writer)
    return writer.data


def estimate_delay_and_correlation(reference, signal, max_lag=100):
    """Estimate the delay of ``signal`` relatively to ``reference`` (in
    samples) by cross-correlation, and return it along with the normalized
    correlation between the two signals at that delay."""
    nfft = 1 << (2 * max(reference.shape[0], signal.shape[0]) - 1).bit_length()
    corr = np.fft.irfft(
        np.fft.rfft(signal, nfft) * np.conj(np.fft.rfft(reference, nfft)),
        nfft)
    corr = np.concatenate((corr[-max_lag:], corr[:max_lag + 1]))
    delay = int(np.argmax(corr) - max_lag)
    correlation = corr[delay + max_lag] / (
        np.linalg.norm(reference) * np.linalg.norm(signal))
    return delay, correlation


@pytest.mark.parametrize("phase_locking",
                         [PhaseLocking.NONE, PhaseLocking.IDENTITY])
@pytest.mark.parametrize("speed", [0.75, 1.5])
def test_multichannel_preserves_inter_channel_delay(speed, phase_locking):
    """The phase vocoder should preserve the phase relations and the delay
    between the channels of a stereo signal, and keep the stereo image
    stable over time."""
    sample = make_stereo_sample()
    output = run_phasevocoder(sample, speed, phase_locking)

    # Check the inter-channel delay and correlation on successive windows
    # of the output (skipping the edges, where the signal fades in and
    # out), to make sure the stereo image is stable over time.
    window = 8192
    for start in range(4096, output.shape[1] - 4096 - window, 4096):
        left = output[0, start:start + window]
        right = output[1, start:start + window]

        delay, correlation = estimate_delay_and_correlation(left, right)

        # The inter-channel delay should be preserved
        assert abs(delay - INTER_CHANNEL_DELAY) <= 1

        # The channels should stay in sync: with independent per-channel
        # analysis, the phase drift between the channels decorrelates them
        # (the correlation drops around 0.6 with identity phase locking)
        assert correlation >= 0.9


@pytest.mark.parametrize("phase_locking",
                         [PhaseLocking.NONE, PhaseLocking.IDENTITY])
@pytest.mark.parametrize("speed", [0.75, 1.5])
def test_multichannel_identical_channels_stay_identical(speed, phase_locking):
    """Two identical channels should remain identical after the phase
    vocoder (the channels should stay in sync)."""
    sample = make_stereo_sample()
    mono = sample[0]
    output = run_phasevocoder(np.vstack((mono, mono)), speed, phase_locking)

    assert_allclose(output[0], output[1], atol=1e-6)


@pytest.mark.parametrize("phase_locking",
                         [PhaseLocking.NONE, PhaseLocking.IDENTITY])
@pytest.mark.parametrize("speed", [0.75, 1.5])
def test_mono_behavior_unchanged(speed, phase_locking):
    """Running the phase vocoder on a mono signal should give the same
    result as before: a mono signal processed alone should give the same
    output as one channel of a stereo signal made of two copies of that
    mono signal."""
    sample = make_stereo_sample()
    mono = sample[0]

    output_mono = run_phasevocoder(mono.reshape(1, -1), speed, phase_locking)
    output_stereo = run_phasevocoder(
        np.vstack((mono, mono)), speed, phase_locking)

    assert output_mono.shape == output_stereo[0].reshape(1, -1).shape
    assert_allclose(output_mono[0], output_stereo[0], atol=1e-6)
