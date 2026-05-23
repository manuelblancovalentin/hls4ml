#ifndef NNET_DENSE_BACKPROP_H_
#define NNET_DENSE_BACKPROP_H_

#include "ap_int.h"
#include "../common/trainable_trace.h"

namespace nnet {

    // Dense backward pass for the first trainable path.
    //
    // This kernel is intentionally split from weight application. It computes:
    //   - grad_out: dL/dx for the previous layer,
    //   - weight_grad_accum and bias_grad_accum: running batch sums,
    //   - weight_grad and bias_grad: averaged gradients at batch_end.
    //
    // The update itself is handled later by the optimizer/controller path so a
    // global alpha can throttle every parameter update without changing the
    // layer-local gradient direction.
    template<typename CONFIG_T>
    void dense_backpass(
        const typename CONFIG_T::data_in_t data_in[CONFIG_T::n_in],
        const typename CONFIG_T::grad_in_t grad_in[CONFIG_T::n_out],
        const typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],

        typename CONFIG_T::grad_out_t grad_out[CONFIG_T::n_in],

        typename CONFIG_T::gradient_accum_t weight_grad_accum[CONFIG_T::n_in * CONFIG_T::n_out],
        typename CONFIG_T::gradient_accum_t bias_grad_accum[CONFIG_T::n_out],

        typename CONFIG_T::weight_grad_t weight_grad[CONFIG_T::n_in * CONFIG_T::n_out],
        typename CONFIG_T::bias_grad_t bias_grad[CONFIG_T::n_out],

        bool reset_accumulators,
        bool batch_end
    ) {

        constexpr unsigned n_in = CONFIG_T::n_in;
        constexpr unsigned n_out = CONFIG_T::n_out;
        constexpr unsigned n_weights = CONFIG_T::n_in * CONFIG_T::n_out;

        using accum_t = typename CONFIG_T::gradient_accum_t;
        using weight_grad_t = typename CONFIG_T::weight_grad_t;
        using bias_grad_t = typename CONFIG_T::bias_grad_t;
        using grad_out_t = typename CONFIG_T::grad_out_t;

        // reset_accumulators is asserted at the first sample of a batch. This
        // lets one static accumulator buffer collect gradients across the batch.
        if (reset_accumulators) {
            ResetWeightAccumulators:
            for (unsigned i = 0; i < n_weights; i++) {
                #pragma HLS PIPELINE II=1
                weight_grad_accum[i] = 0;
            }

            ResetBiasAccumulators:
            for (unsigned j = 0; j < n_out; j++) {
                #pragma HLS PIPELINE II=1
                bias_grad_accum[j] = 0;
            }
        }

        // dL/dx_i = sum_j dL/dy_j * W_ij.
        BackwardInputGradient:
        for (unsigned i = 0; i < n_in; i++) {
            #pragma HLS PIPELINE II=1
            accum_t acc = 0;

            BackwardInputGradientOut:
            for (unsigned j = 0; j < n_out; j++) {
                const unsigned idx = i * n_out + j;
                acc += accum_t(weights[idx]) * accum_t(grad_in[j]);
            }

            grad_out[i] = grad_out_t(acc);
        }

        // dL/dW_ij = x_i * dL/dy_j.
        AccumulateWeightGradient:
        for (unsigned i = 0; i < n_in; i++) {
            AccumulateWeightGradientOut:
            for (unsigned j = 0; j < n_out; j++) {
                #pragma HLS PIPELINE II=1
                const unsigned idx = i * n_out + j;
                weight_grad_accum[idx] += accum_t(data_in[i]) * accum_t(grad_in[j]);
            }
        }

        // dL/db_j = dL/dy_j.
        AccumulateBiasGradient:
        for (unsigned j = 0; j < n_out; j++) {
            #pragma HLS PIPELINE II=1
            bias_grad_accum[j] += accum_t(grad_in[j]);
        }

        // batch_size_log2 comes from ENABOL/hls4ml config. We require power-of-2
        // batches so averaging can be emitted as a shift instead of a divider.
        if (batch_end) {
            const ap_uint<16> batch_shift = CONFIG_T::batch_size_log2;

            EmitWeightGradient:
            for (unsigned i = 0; i < n_weights; i++) {
                #pragma HLS PIPELINE II=1
                accum_t avg = weight_grad_accum[i];
                avg >>= batch_shift;
                weight_grad[i] = weight_grad_t(avg);
            }

            EmitBiasGradient:
            for (unsigned j = 0; j < n_out; j++) {
                #pragma HLS PIPELINE II=1
                accum_t avg = bias_grad_accum[j];
                avg >>= batch_shift;
                bias_grad[j] = bias_grad_t(avg);
            }
        }

        // These trace calls are no-ops unless the generated firmware is compiled
        // with HLS4ML_TRAINABLE_TRACE, which should only happen when hls4ml trace
        // collection is enabled. The CONFIG_T trace names will be emitted by the
        // writer when we wire trainable tracing into parameters.h/bridge storage.
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_data_in_name, data_in, n_in);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_grad_in_name, grad_in, n_out);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_grad_out_name, grad_out, n_in);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_weight_grad_accum_name, weight_grad_accum, n_weights);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_bias_grad_accum_name, bias_grad_accum, n_out);

        if (batch_end) {
            HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_weight_grad_name, weight_grad, n_weights);
            HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_bias_grad_name, bias_grad, n_out);
        }

    } // dense_backpass

} // namespace nnet

#endif
