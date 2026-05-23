#ifndef HLS4ML_TRAINABLE_MSE_H_
#define HLS4ML_TRAINABLE_MSE_H_

#include "../common/trainable_trace.h"

namespace nnet {

    // Shared squared-error endpoint.
    //
    // LOSS_NUM / LOSS_DEN controls the scalar loss prefactor, while
    // GRAD_NUM / GRAD_DEN controls dL/dy. This lets `mse` and `half_mse` use the
    // same implementation without runtime branches:
    //   mse:      L = sum(diff^2),       dL/dy = 2 * diff
    //   half_mse: L = 0.5 * sum(diff^2), dL/dy = diff
    template<typename CONFIG_T, int LOSS_NUM, int LOSS_DEN, int GRAD_NUM, int GRAD_DEN>
    void mse_core(
        const typename CONFIG_T::data_in_t prediction[CONFIG_T::n_out],
        const typename CONFIG_T::ground_truth_t ground_truth[CONFIG_T::n_out],
        typename CONFIG_T::loss_t loss[1],
        typename CONFIG_T::grad_out_t loss_grad[CONFIG_T::n_out]
    ) {

        constexpr unsigned n_out = CONFIG_T::n_out;

        using loss_t = typename CONFIG_T::loss_t;
        using grad_out_t = typename CONFIG_T::grad_out_t;

        loss_t loss_accum = 0;

        // diff is prediction - ground_truth so the emitted gradient points in the
        // standard descent direction when SGD later proposes -lr * gradient.
        LossGradient:
        for (unsigned i = 0; i < n_out; i++) {
            #pragma HLS PIPELINE II=1
            loss_t diff = loss_t(prediction[i]) - loss_t(ground_truth[i]);
            loss_accum += (loss_t(LOSS_NUM) * diff * diff) / loss_t(LOSS_DEN);
            loss_grad[i] = grad_out_t((loss_t(GRAD_NUM) * diff) / loss_t(GRAD_DEN));
        }

        loss[0] = loss_accum;

        // These are inactive unless trainable trace support is enabled by the
        // writer. See trainable/common/trainable_trace.h.
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_prediction_name, prediction, n_out);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_ground_truth_name, ground_truth, n_out);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_loss_name, loss, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_loss_grad_name, loss_grad, n_out);

    } // mse_core

    template<typename CONFIG_T>
    void mse(
        const typename CONFIG_T::data_in_t prediction[CONFIG_T::n_out],
        const typename CONFIG_T::ground_truth_t ground_truth[CONFIG_T::n_out],
        typename CONFIG_T::loss_t loss[1],
        typename CONFIG_T::grad_out_t loss_grad[CONFIG_T::n_out]
    ) {
        mse_core<CONFIG_T, 1, 1, 2, 1>(prediction, ground_truth, loss, loss_grad);
    } // mse

    template<typename CONFIG_T>
    void half_mse(
        const typename CONFIG_T::data_in_t prediction[CONFIG_T::n_out],
        const typename CONFIG_T::ground_truth_t ground_truth[CONFIG_T::n_out],
        typename CONFIG_T::loss_t loss[1],
        typename CONFIG_T::grad_out_t loss_grad[CONFIG_T::n_out]
    ) {
        mse_core<CONFIG_T, 1, 2, 1, 1>(prediction, ground_truth, loss, loss_grad);
    } // half_mse

} // namespace nnet

#endif
