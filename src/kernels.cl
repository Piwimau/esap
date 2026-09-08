/**
 * @brief Applies a Hann window to a specified sequence of samples.
 *
 * @param[out] windowedSamples The output samples after applying the window.
 * @param[in]  samples         The input samples to apply the window to.
 * @param[in]  window          The precomputed coefficients of a Hann window.
 * @param[in]  numFrames       The number of STFT frames (i.e., the number of
 *                             windowed FFTs computed per audio channel).
 * @param[in]  numWavFrames    The number of audio frames (i.e., the number of
 *                             samples per audio channel) in the original
 *                             waveform audio file.
 */
__kernel void apply_window(
    __global float* windowedSamples,
    const __global float* samples,
    __constant float* window,
    const ulong numFrames,
    const ulong numWavFrames
) {
    const size_t batch = get_global_id(0);
    const size_t k = get_global_id(1);
    if (k >= WINDOW_SIZE) {
        return;
    }
    const size_t c = batch / numFrames;
    const size_t f = batch % numFrames;
    const size_t idx = f * HOP_SIZE + k;
    windowedSamples[batch * WINDOW_SIZE + k] = (idx < numWavFrames)
        ? samples[c * numWavFrames + idx] * window[k]
        : 0.0F;
}

/**
 * @brief Applies a specified sequence of gains to the bins of a Short-Time
 * Fourier Transform (STFT).
 *
 * @param[in, out] bins  The bins of the STFT to apply the gains to.
 * @param[in]      gains The sequence of gains to apply to the bins.
 */
__kernel void apply_gains(__global float2* bins, const __global float* gains) {
    const size_t batch = get_global_id(0);
    const size_t k = get_global_id(1);
    if (k >= NUM_BINS) {
        return;
    }
    bins[batch * NUM_BINS + k] *= gains[k];
}

/**
 * @brief Applies the overlap-add method to reconstruct the original waveform
 * audio samples from the output of an inverse Short-Time Fourier Transform
 * (STFT).
 *
 * @param[out] samples      The output samples after applying the overlap-add
 *                          method.
 * @param[in]  ifftOut      The output of the inverse STFT, which consists of
 *                          the time-domain samples for each windowed FFT.
 * @param[in]  window       The precomputed coefficients of a Hann window.
 * @param[in]  numFrames    The number of STFT frames (i.e., the number of
 *                          windowed FFTs computed per audio channel).
 * @param[in]  numWavFrames The number of audio frames (i.e., the number of
 *                          samples per audio channel) in the original waveform
 *                          audio file.
 */
__kernel void overlap_add(
    __global float* samples,
    const __global float* ifftOut,
    __constant float* window,
    const ulong numFrames,
    const ulong numWavFrames
) {
    const size_t c = get_global_id(0);
    const size_t idx = get_global_id(1);
    if (idx >= numWavFrames) {
        return;
    }
    float sum = 0.0F;
    float norm = 0.0F;
    const size_t first = (idx >= WINDOW_SIZE)
        ? (idx - WINDOW_SIZE) / HOP_SIZE + 1
        : 0;
    for (size_t f = first; f < numFrames; f++) {
        const size_t k = idx - f * HOP_SIZE;
        if (k >= WINDOW_SIZE) {
            break;
        }
        sum += ifftOut[(c * numFrames + f) * WINDOW_SIZE + k] / WINDOW_SIZE
            * window[k];
        norm += window[k] * window[k];
    }
    samples[c * numWavFrames + idx] = (norm > 1.0E-8F) ? sum / norm : 0.0F;
}