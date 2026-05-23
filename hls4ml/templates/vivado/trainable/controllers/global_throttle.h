#ifndef HLS4ML_TRAINABLE_GLOBAL_THROTTLE_H_
#define HLS4ML_TRAINABLE_GLOBAL_THROTTLE_H_

#include "../common/trainable_trace.h"

namespace nnet {

    template<typename CONFIG_T>
    void global_throttle_none(typename CONFIG_T::alpha_t alpha[1]) {
        alpha[0] = typename CONFIG_T::alpha_t(1);

        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_alpha_name, alpha, 1);

    } // global_throttle_none

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

        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_weights_after_update_name, weights, n_weights);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_biases_after_update_name, biases, n_out);

    } // apply_dense_update

} // namespace nnet

#endif
