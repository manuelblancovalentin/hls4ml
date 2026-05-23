#ifndef HLS4ML_TRAINABLE_SGD_H_
#define HLS4ML_TRAINABLE_SGD_H_

#include "../common/trainable_trace.h"

namespace nnet {

    template<typename CONFIG_T>
    void sgd(
        const typename CONFIG_T::weight_grad_t weight_grad[CONFIG_T::n_in * CONFIG_T::n_out],
        const typename CONFIG_T::bias_grad_t bias_grad[CONFIG_T::n_out],
        typename CONFIG_T::raw_update_t weight_update[CONFIG_T::n_in * CONFIG_T::n_out],
        typename CONFIG_T::raw_update_t bias_update[CONFIG_T::n_out],
        typename CONFIG_T::learning_rate_t learning_rate
    ) {

        constexpr unsigned n_out = CONFIG_T::n_out;
        constexpr unsigned n_weights = CONFIG_T::n_in * CONFIG_T::n_out;

        using raw_update_t = typename CONFIG_T::raw_update_t;

        WeightUpdateProposal:
        for (unsigned i = 0; i < n_weights; i++) {
            #pragma HLS PIPELINE II=1
            weight_update[i] = raw_update_t(-learning_rate * weight_grad[i]);
        }

        BiasUpdateProposal:
        for (unsigned i = 0; i < n_out; i++) {
            #pragma HLS PIPELINE II=1
            bias_update[i] = raw_update_t(-learning_rate * bias_grad[i]);
        }

        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_weight_update_name, weight_update, n_weights);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_bias_update_name, bias_update, n_out);

    } // sgd

} // namespace nnet

#endif
