#ifndef HLS4ML_TRAINABLE_GLOBAL_THROTTLE_H_
#define HLS4ML_TRAINABLE_GLOBAL_THROTTLE_H_

#include "../common/trainable_trace.h"

namespace nnet {

    // -------------------------------------------------------------------------
    //  Phase 1  —  Curvature sensor (per-layer squared-norm contribution)
    //
    //  Called once per trainable layer inside the batch_end block.  Maintains
    //  persistent prev-θ / prev-G storage and emits two scalar outputs:
    //
    //      dtheta_sq_contrib = ||θ_t - θ_{t-1}||²   (summed over this layer)
    //      dgrad_sq_contrib  = ||G_t - G_{t-1}||²
    //
    //  The caller sums these across layers, then feeds the global totals into
    //  the controller law (Phase 2).
    //
    //  reset_numerator  – when true, clears the has_prev flag so the next call
    //  stores fresh prev values and emits zero squared-norm contributions.
    //
    //  This sensor is shared by all global-throttle orders (GT-0, GT-1, …).
    //  Higher orders may add an EMA layer on top of the raw curvature.
    // -------------------------------------------------------------------------
    template<typename CONFIG_T>
    void curvature_sensor_order0(
        const typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
        const typename CONFIG_T::bias_t biases[CONFIG_T::n_out],
        const typename CONFIG_T::weight_grad_t weight_grad[CONFIG_T::n_in * CONFIG_T::n_out],
        const typename CONFIG_T::bias_grad_t bias_grad[CONFIG_T::n_out],
        typename CONFIG_T::controller_metric_t &dtheta_sq_contrib,
        typename CONFIG_T::controller_metric_t &dgrad_sq_contrib,
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
            dtheta_sq_contrib = metric_t(0);
            dgrad_sq_contrib = metric_t(0);

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
            dtheta_sq_contrib = metric_t(0);
            dgrad_sq_contrib = metric_t(0);

            NormWeightDiff:
            for (unsigned i = 0; i < n_weights; i++) {
                #pragma HLS PIPELINE II=1
                metric_t dw = metric_t(weights[i]) - metric_t(prev_weights[i]);
                metric_t dg = metric_t(weight_grad[i]) - metric_t(prev_weight_grad[i]);
                dtheta_sq_contrib += dw * dw;
                dgrad_sq_contrib += dg * dg;
            }

            NormBiasDiff:
            for (unsigned i = 0; i < n_out; i++) {
                #pragma HLS PIPELINE II=1
                metric_t dw = metric_t(biases[i]) - metric_t(prev_biases[i]);
                metric_t dg = metric_t(bias_grad[i]) - metric_t(prev_bias_grad[i]);
                dtheta_sq_contrib += dw * dw;
                dgrad_sq_contrib += dg * dg;
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

    } // curvature_sensor_order0


    // -------------------------------------------------------------------------
    //  Phase 2  —  Controller laws (global, called once per batch_end)
    //
    //  Each takes the global (cross-layer) squared norms and emits alpha.
    // -------------------------------------------------------------------------


    // CTRL-NONE baseline.
    // Identity: alpha = 1, no curvature computation needed.
    template<typename CONFIG_T>
    void global_throttle_none(typename CONFIG_T::alpha_t alpha[1]) {
        alpha[0] = typename CONFIG_T::alpha_t(1);

        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_alpha_name, alpha, 1);

    } // global_throttle_none


    // CTRL-GT-ORDER-0: division-free binary-search global throttle.
    //
    //  Replaces the algebraic safe-gain law α = χ / (η·C + ε) with an
    //  inequality comparison search over a table of binary-fraction alpha
    //  candidates.  No division, no sqrt — CSIM-safe with ap_fixed types.
    //
    //  Constraint (from the stability inequality η·α·C ≤ χ):
    //
    //      α² · η² · ||ΔG||²  ≤  χ² · (||Δθ||² + ε²)
    //
    //  Candidates are evaluated in descending order; the first (largest) alpha
    //  satisfying the inequality is selected.  If none satisfy, α = α_min.
    //
    //  State: none.  Curvature is recomputed fresh each batch_end from the
    //  global squared norms passed in.
    template<typename CONFIG_T>
    void global_throttle_order0_law(
        const typename CONFIG_T::controller_metric_t dtheta_sq,
        const typename CONFIG_T::controller_metric_t dgrad_sq,
        typename CONFIG_T::alpha_t alpha[1],
        bool reset_numerator
    ) {

        using metric_t = typename CONFIG_T::controller_metric_t;

        if (reset_numerator) {
            alpha[0] = typename CONFIG_T::alpha_t(1);
            HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_alpha_name, alpha, 1);
            return;
        }

        metric_t lr(CONFIG_T::learning_rate);
        metric_t chi(CONFIG_T::controller_chi);
        metric_t eps(CONFIG_T::controller_epsilon);
        metric_t lr_sq = lr * lr;
        metric_t chi_sq = chi * chi;
        metric_t eps_sq = eps * eps;

        metric_t rhs = chi_sq * (dtheta_sq + eps_sq);
        metric_t lhs_base = lr_sq * dgrad_sq;

        static const metric_t cand_val[11] = {
            metric_t(1.000000),  metric_t(0.875000),  metric_t(0.750000),
            metric_t(0.625000),  metric_t(0.500000),  metric_t(0.375000),
            metric_t(0.250000),  metric_t(0.187500),  metric_t(0.125000),
            metric_t(0.062500),  metric_t(0.031250),
        };
        static const metric_t cand_sq[11] = {
            metric_t(1.000000),  metric_t(0.765625),  metric_t(0.562500),
            metric_t(0.390625),  metric_t(0.250000),  metric_t(0.140625),
            metric_t(0.062500),  metric_t(0.035156),  metric_t(0.015625),
            metric_t(0.003906),  metric_t(0.000977),
        };
        static const unsigned n_candidates = 11;

        alpha[0] = typename CONFIG_T::alpha_t(CONFIG_T::controller_alpha_min);

        for (unsigned i = 0; i < n_candidates; i++) {
            #pragma HLS UNROLL
            metric_t lhs = cand_sq[i] * lhs_base;
            if (lhs <= rhs) {
                alpha[0] = typename CONFIG_T::alpha_t(cand_val[i]);
                break;
            }
        }

        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_dgrad_sq_name, &dgrad_sq, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_dtheta_sq_name, &dtheta_sq, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_lhs_sq_name, &lhs_base, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_rhs_sq_name, &rhs, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_alpha_name, alpha, 1);

    } // global_throttle_order0_law


    // CTRL-GT-ORDER-1: division-free first-order alpha-state global throttle.
    //
    //  Same inequality search as GT-0, but alpha_state evolves smoothly toward
    //  the feasible candidate via a first-order attractor:
    //
    //      alpha_feasible = largest candidate satisfying  α²·η²·||ΔG||² ≤ χ²·(||Δθ||² + ε²)
    //      α ← clip(α + k_α·(alpha_feasible - α),  α_min,  α_max)
    //
    //  No division, no sqrt — CSIM-safe with ap_fixed types.
    //
    //  State: α_state (persistent scalar).
    template<typename CONFIG_T>
    void global_throttle_order1_law(
        const typename CONFIG_T::controller_metric_t dtheta_sq,
        const typename CONFIG_T::controller_metric_t dgrad_sq,
        typename CONFIG_T::alpha_t alpha[1],
        bool reset_numerator
    ) {

        using metric_t = typename CONFIG_T::controller_metric_t;
        static metric_t alpha_state = metric_t(1);

        if (reset_numerator) {
            alpha_state = metric_t(1);
            alpha[0] = typename CONFIG_T::alpha_t(1);
            HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_alpha_name, alpha, 1);
            return;
        }

        metric_t lr(CONFIG_T::learning_rate);
        metric_t chi(CONFIG_T::controller_chi);
        metric_t eps(CONFIG_T::controller_epsilon);
        metric_t k_alpha(CONFIG_T::controller_k_alpha);
        metric_t lr_sq = lr * lr;
        metric_t chi_sq = chi * chi;
        metric_t eps_sq = eps * eps;

        metric_t rhs = chi_sq * (dtheta_sq + eps_sq);
        metric_t lhs_base = lr_sq * dgrad_sq;

        static const metric_t cand_val[11] = {
            metric_t(1.000000),  metric_t(0.875000),  metric_t(0.750000),
            metric_t(0.625000),  metric_t(0.500000),  metric_t(0.375000),
            metric_t(0.250000),  metric_t(0.187500),  metric_t(0.125000),
            metric_t(0.062500),  metric_t(0.031250),
        };
        static const metric_t cand_sq[11] = {
            metric_t(1.000000),  metric_t(0.765625),  metric_t(0.562500),
            metric_t(0.390625),  metric_t(0.250000),  metric_t(0.140625),
            metric_t(0.062500),  metric_t(0.035156),  metric_t(0.015625),
            metric_t(0.003906),  metric_t(0.000977),
        };
        static const unsigned n_candidates = 11;

        metric_t alpha_feasible = metric_t(CONFIG_T::controller_alpha_min);
        for (unsigned i = 0; i < n_candidates; i++) {
            #pragma HLS UNROLL
            metric_t lhs = cand_sq[i] * lhs_base;
            if (lhs <= rhs) {
                alpha_feasible = cand_val[i];
                break;
            }
        }

        metric_t alpha_next = alpha_state + k_alpha * (alpha_feasible - alpha_state);

        metric_t alpha_min = metric_t(CONFIG_T::controller_alpha_min);
        metric_t alpha_max = metric_t(CONFIG_T::controller_alpha_max);

        if (alpha_next < alpha_min) {
            alpha_state = alpha_min;
        } else if (alpha_next > alpha_max) {
            alpha_state = alpha_max;
        } else {
            alpha_state = alpha_next;
        }

        alpha[0] = typename CONFIG_T::alpha_t(alpha_state);

        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_dgrad_sq_name, &dgrad_sq, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_dtheta_sq_name, &dtheta_sq, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_alpha_feasible_name, &alpha_feasible, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_alpha_state_name, &alpha_state, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_alpha_name, alpha, 1);

    } // global_throttle_order1_law


    // -------------------------------------------------------------------------
    //  Phase 3  —  Apply one alpha-scaled update to a Dense layer
    //
    //  Shared by all controllers.  The optimizer decides the raw direction; the
    //  controller law decides alpha; this function is only the final mutation.
    // -------------------------------------------------------------------------
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
