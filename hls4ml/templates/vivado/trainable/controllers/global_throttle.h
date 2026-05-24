#ifndef HLS4ML_TRAINABLE_GLOBAL_THROTTLE_H_
#define HLS4ML_TRAINABLE_GLOBAL_THROTTLE_H_

#include "../common/trainable_trace.h"

#include <cmath>

namespace nnet {

    // CTRL-NONE baseline.
    //
    // This preserves the trainable data path but disables throttling by emitting
    // alpha = 1. It is the first controller target because it lets us verify
    // loss, backprop, SGD, and weight application before adding curvature logic.
    template<typename CONFIG_T>
    void global_throttle_none(typename CONFIG_T::alpha_t alpha[1]) {
        alpha[0] = typename CONFIG_T::alpha_t(1);

        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_alpha_name, alpha, 1);

    } // global_throttle_none

    // CTRL-GT-ORDER-0: algebraic safe-gain global throttle.
    //
    // Computes the curvature proxy C = ||ΔG|| / (||Δθ|| + ε) from persistent
    // previous-θ and previous-G storage, then derives the safe throttle
    // α = clip(χ / (η·C + ε), α_min, α_max).
    //
    // Persistent state (prev_theta, prev_grad, has_prev) is managed internally via
    // static arrays. The `reset_numerator` signal clears the has_prev flag so that
    // the next call stores fresh prev values and emits α = 1 (no curvature yet).
    template<typename CONFIG_T>
    void global_throttle_order0(
        typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
        typename CONFIG_T::bias_t biases[CONFIG_T::n_out],
        const typename CONFIG_T::weight_grad_t weight_grad[CONFIG_T::n_in * CONFIG_T::n_out],
        const typename CONFIG_T::bias_grad_t bias_grad[CONFIG_T::n_out],
        typename CONFIG_T::alpha_t alpha[1],
        bool reset_numerator
    ) {

        constexpr unsigned n_weights = CONFIG_T::n_in * CONFIG_T::n_out;
        constexpr unsigned n_out = CONFIG_T::n_out;

        using metric_t = typename CONFIG_T::controller_metric_t;
        using weight_t = typename CONFIG_T::weight_t;
        using bias_t = typename CONFIG_T::bias_t;
        using grad_t = typename CONFIG_T::weight_grad_t;
        using bias_grad_t = typename CONFIG_T::bias_grad_t;

        // Persistent storage for previous parameters and gradients.
        static weight_t prev_weights[CONFIG_T::n_in * CONFIG_T::n_out];
        static bias_t prev_biases[CONFIG_T::n_out];
        static grad_t prev_weight_grad[CONFIG_T::n_in * CONFIG_T::n_out];
        static bias_grad_t prev_bias_grad[CONFIG_T::n_out];
        static bool has_prev = false;

        if (reset_numerator) {
            has_prev = false;
        }

        if (!has_prev) {
            // First call or after reset: store current values, emit α = 1.
            alpha[0] = typename CONFIG_T::alpha_t(1);

            StorePrevWeights:
            for (unsigned i = 0; i < n_weights; i++) {
                #pragma HLS PIPELINE II=1
                prev_weights[i] = weights[i];
                prev_weight_grad[i] = weight_grad[i];
            }

            StorePrevBiases:
            for (unsigned i = 0; i < n_out; i++) {
                #pragma HLS PIPELINE II=1
                prev_biases[i] = biases[i];
                prev_bias_grad[i] = bias_grad[i];
            }

            has_prev = true;

        } else {
            // Accumulate squared L2 differences across all parameters.
            metric_t dtheta_sq = 0;
            metric_t dgrad_sq = 0;

            NormWeightDiff:
            for (unsigned i = 0; i < n_weights; i++) {
                #pragma HLS PIPELINE II=1
                metric_t dw = metric_t(weights[i]) - metric_t(prev_weights[i]);
                metric_t dg = metric_t(weight_grad[i]) - metric_t(prev_weight_grad[i]);
                dtheta_sq += dw * dw;
                dgrad_sq += dg * dg;
            }

            NormBiasDiff:
            for (unsigned i = 0; i < n_out; i++) {
                #pragma HLS PIPELINE II=1
                metric_t dw = metric_t(biases[i]) - metric_t(prev_biases[i]);
                metric_t dg = metric_t(bias_grad[i]) - metric_t(prev_bias_grad[i]);
                dtheta_sq += dw * dw;
                dgrad_sq += dg * dg;
            }

            // C = ||ΔG|| / (||Δθ|| + ε)
            // sqrt computed through double; for CSIM this casts to double and
            // back. For synthesis this should be replaced with hls::sqrt or an
            // iterative fixed-point sqrt (future work, the controller runs at
            // batch-end rate so latency is not on the critical path).
            metric_t dtheta_norm = metric_t(std::sqrt(double(dtheta_sq)));
            metric_t dgrad_norm = metric_t(std::sqrt(double(dgrad_sq)));

            metric_t eps = metric_t(CONFIG_T::controller_epsilon);
            metric_t curvature = dgrad_norm / (dtheta_norm + eps);

            // α = clip(χ / (η·C + ε), α_min, α_max)
            metric_t chi = metric_t(CONFIG_T::controller_chi);
            metric_t lr = metric_t(CONFIG_T::learning_rate);
            metric_t alpha_raw = chi / (lr * curvature + eps);

            metric_t alpha_min = metric_t(CONFIG_T::controller_alpha_min);
            metric_t alpha_max = metric_t(CONFIG_T::controller_alpha_max);

            if (alpha_raw < alpha_min) {
                alpha[0] = typename CONFIG_T::alpha_t(alpha_min);
            } else if (alpha_raw > alpha_max) {
                alpha[0] = typename CONFIG_T::alpha_t(alpha_max);
            } else {
                alpha[0] = typename CONFIG_T::alpha_t(alpha_raw);
            }

            // Store current values as prev for the next batch.
            UpdatePrevWeights:
            for (unsigned i = 0; i < n_weights; i++) {
                #pragma HLS PIPELINE II=1
                prev_weights[i] = weights[i];
                prev_weight_grad[i] = weight_grad[i];
            }

            UpdatePrevBiases:
            for (unsigned i = 0; i < n_out; i++) {
                #pragma HLS PIPELINE II=1
                prev_biases[i] = biases[i];
                prev_bias_grad[i] = bias_grad[i];
            }
        }

        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_alpha_name, alpha, 1);

    } // global_throttle_order0

    // Apply one alpha-scaled update to a Dense layer's weights and biases.
    //
    // This helper is shared by CTRL-NONE and future global-throttle controllers.
    // The optimizer decides the raw direction; the controller decides alpha; this
    // function is only the final state mutation.
    template<typename CONFIG_T>
    void apply_dense_update(
        typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
        typename CONFIG_T::bias_t biases[CONFIG_T::n_out],
        const typename CONFIG_T::raw_update_t weight_update[CONFIG_T::n_in * CONFIG_T::n_out],
        const typename CONFIG_T::raw_update_t bias_update[CONFIG_T::n_out],
        const typename CONFIG_T::alpha_t alpha[1]
    ) {

        constexpr unsigned n_out = CONFIG_T::n_out;
        constexpr unsigned n_weights = CONFIG_T::n_in * CONFIG_T::n_out;

        using weight_t = typename CONFIG_T::weight_t;
        using bias_t = typename CONFIG_T::bias_t;
        using update_t = typename CONFIG_T::update_t;

        // Convert through update_t before assigning back to the stored parameter
        // type. That keeps update precision separate from parameter precision.
        WeightApplyUpdate:
        for (unsigned i = 0; i < n_weights; i++) {
            #pragma HLS PIPELINE II=1
            update_t throttled_update = update_t(alpha[0] * weight_update[i]);
            weights[i] = weight_t(weights[i] + throttled_update);
        }

        BiasApplyUpdate:
        for (unsigned i = 0; i < n_out; i++) {
            #pragma HLS PIPELINE II=1
            update_t throttled_update = update_t(alpha[0] * bias_update[i]);
            biases[i] = bias_t(biases[i] + throttled_update);
        }

        // Inactive unless HLS4ML_TRAINABLE_TRACE is defined by generated code.
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_weights_after_update_name, weights, n_weights);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_biases_after_update_name, biases, n_out);

    } // apply_dense_update

} // namespace nnet

#endif
