/**
 * @brief Applies a window function to a buffer of samples.
 *
 * @param[out] windowedSamples The output samples after applying the window
 *                             function.
 * @param[in]  samples         The input samples to apply the window function
 *                             to.
 * @param[in]  window          The precomputed coefficients of a window
 *                             function.
 * @param[in]  numFrames       The number of FFTs computed per audio channel.
 * @param[in]  numWavFrames    The number of samples per audio channel in the
 *                             original waveform audio file.
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
 * @brief Applies a filter (represented by gain coefficients) to a buffer of
 * complex frequency bins.
 *
 * @param[in, out] bins  The complex frequency bins.
 * @param[in]      gains The gains to apply to the bins.
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
 * @brief Reconstructs the time-domain signal from the output of an inverse
 * Short-Time Fourier Transform (ISTFT) by means of the overlap-add method.
 *
 * @param[out] samples      The reconstructed time-domain samples.
 * @param[in]  frames       The output of the ISTFT, which consists of the
 *                          time-domain samples for each individual FFT.
 * @param[in]  window       The precomputed coefficients of a window function.
 * @param[in]  numFrames    The number of FFTs computed per audio channel.
 * @param[in]  numWavFrames The number of samples per audio channel in the
 *                          original waveform audio file.
 */
__kernel void overlap_add(
    __global float* samples,
    const __global float* frames,
    __constant float* window,
    const ulong numFrames,
    const ulong numWavFrames
) {
    const size_t c = get_global_id(0);
    const size_t f = get_global_id(1);
    if (f >= numWavFrames) {
        return;
    }
    float sum = 0.0F;
    float norm = 0.0F;
    const size_t start = (f >= WINDOW_SIZE)
        ? (f - WINDOW_SIZE) / HOP_SIZE + 1
        : 0;
    for (size_t i = start; i < numFrames; i++) {
        const size_t j = f - i * HOP_SIZE;
        if (j >= WINDOW_SIZE) {
            break;
        }
        sum += frames[(c * numFrames + i) * WINDOW_SIZE + j] / WINDOW_SIZE
            * window[j];
        norm += window[j] * window[j];
    }
    samples[c * numWavFrames + f] = (norm > 1.0E-8F) ? sum / norm : 0.0F;
}