#ifndef HLS4ML_TRAINABLE_GLOBAL_THROTTLE_H_
#define HLS4ML_TRAINABLE_GLOBAL_THROTTLE_H_

#include "../common/trainable_trace.h"

namespace nnet {

    // -------------------------------------------------------------------------
    //  Phase 1  —  Raw-update controller sensor
    //
    //  Called once per trainable layer inside the batch_end block, after the
    //  optimizer has proposed raw updates and before alpha is applied. Maintains
    //  persistent prev-G storage and emits two scalar outputs:
    //
    //      raw_update_norm_sq_contrib = ||Δθ_raw||²
    //      dgrad_norm_sq_contrib      = ||G_t - G_{t-1}||²
    //
    //  Δθ_raw is the optimizer-proposed update before alpha. For SGD this
    //  already includes the learning rate because sgd.h computes:
    //
    //      Δθ_raw = -learning_rate * gradient
    //
    //  The controller must use this raw geometry, not the already-throttled or
    //  quantized actual parameter movement. Otherwise alpha can suppress
    //  movement, which makes ||Δθ|| tiny, which makes curvature explode, which
    //  suppresses alpha further.
    //
    //  reset_numerator  – when true, clears the has_prev flag so the next call
    //  stores fresh prev gradients and emits zero ΔG contribution.
    // -------------------------------------------------------------------------
    template<typename CONFIG_T>
    void raw_update_sensor_order0(
        const typename CONFIG_T::raw_update_t weight_update[CONFIG_T::n_in * CONFIG_T::n_out],
        const typename CONFIG_T::raw_update_t bias_update[CONFIG_T::n_out],
        const typename CONFIG_T::weight_grad_t weight_grad[CONFIG_T::n_in * CONFIG_T::n_out],
        const typename CONFIG_T::bias_grad_t bias_grad[CONFIG_T::n_out],
        typename CONFIG_T::controller_metric_t &raw_update_norm_sq_contrib,
        typename CONFIG_T::controller_metric_t &dgrad_norm_sq_contrib,
        bool reset_numerator
    ) {

        constexpr unsigned n_weights = CONFIG_T::n_in * CONFIG_T::n_out;
        constexpr unsigned n_out = CONFIG_T::n_out;

        using metric_t = typename CONFIG_T::controller_metric_t;
        using grad_t = typename CONFIG_T::weight_grad_t;
        using bias_grad_t = typename CONFIG_T::bias_grad_t;

        // Persistent storage for previous gradients. Parameter movement is not
        // used for the control geometry because it is downstream of alpha.
        static grad_t prev_weight_grad[CONFIG_T::n_in * CONFIG_T::n_out];
        static bias_grad_t prev_bias_grad[CONFIG_T::n_out];
        static bool has_prev = false;

        if (reset_numerator) {
            has_prev = false;
        }

        raw_update_norm_sq_contrib = metric_t(0);
        RawWeightUpdateNorm:
        for (unsigned i = 0; i < n_weights; i++) {
            #pragma HLS PIPELINE II=1
            metric_t du = metric_t(weight_update[i]);
            raw_update_norm_sq_contrib += du * du;
        }

        RawBiasUpdateNorm:
        for (unsigned i = 0; i < n_out; i++) {
            #pragma HLS PIPELINE II=1
            metric_t du = metric_t(bias_update[i]);
            raw_update_norm_sq_contrib += du * du;
        }

        if (!has_prev) {
            dgrad_norm_sq_contrib = metric_t(0);

            StorePrevWeightGrad:
            for (unsigned i = 0; i < n_weights; i++) {
                #pragma HLS PIPELINE II=1
                prev_weight_grad[i] = weight_grad[i];
            }

            StorePrevBiasGrad:
            for (unsigned i = 0; i < n_out; i++) {
                #pragma HLS PIPELINE II=1
                prev_bias_grad[i] = bias_grad[i];
            }

            has_prev = true;

        } else {
            dgrad_norm_sq_contrib = metric_t(0);

            NormWeightDiff:
            for (unsigned i = 0; i < n_weights; i++) {
                #pragma HLS PIPELINE II=1
                metric_t dg = metric_t(weight_grad[i]) - metric_t(prev_weight_grad[i]);
                dgrad_norm_sq_contrib += dg * dg;
            }

            NormBiasDiff:
            for (unsigned i = 0; i < n_out; i++) {
                #pragma HLS PIPELINE II=1
                metric_t dg = metric_t(bias_grad[i]) - metric_t(prev_bias_grad[i]);
                dgrad_norm_sq_contrib += dg * dg;
            }

            // Store current gradients as prev for the next batch.
            UpdatePrevWeightGrad:
            for (unsigned i = 0; i < n_weights; i++) {
                #pragma HLS PIPELINE II=1
                prev_weight_grad[i] = weight_grad[i];
            }

            UpdatePrevBiasGrad:
            for (unsigned i = 0; i < n_out; i++) {
                #pragma HLS PIPELINE II=1
                prev_bias_grad[i] = bias_grad[i];
            }
        }

    } // raw_update_sensor_order0


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


    template<typename CONFIG_T>
    void reset_controller_metrics(
        typename CONFIG_T::controller_metric_t raw_update_norm_sq[1],
        typename CONFIG_T::controller_metric_t controlled_update_norm_sq[1],
        typename CONFIG_T::controller_metric_t actual_update_norm_sq[1],
        typename CONFIG_T::controller_metric_t dgrad_norm_sq[1],
        typename CONFIG_T::controller_metric_t dtheta_for_control_sq[1],
        typename CONFIG_T::controller_metric_t stability_lhs_raw[1],
        typename CONFIG_T::controller_metric_t stability_lhs_ctrl[1],
        typename CONFIG_T::controller_metric_t stability_rhs[1],
        typename CONFIG_T::controller_metric_t alpha_feasible[1],
        typename CONFIG_T::controller_metric_t alpha_state[1],
        typename CONFIG_T::controller_metric_t alpha_code[1],
        typename CONFIG_T::controller_metric_t alpha_min[1],
        typename CONFIG_T::controller_metric_t feasible[1]
    ) {
        raw_update_norm_sq[0] = typename CONFIG_T::controller_metric_t(0);
        controlled_update_norm_sq[0] = typename CONFIG_T::controller_metric_t(0);
        actual_update_norm_sq[0] = typename CONFIG_T::controller_metric_t(0);
        dgrad_norm_sq[0] = typename CONFIG_T::controller_metric_t(0);
        dtheta_for_control_sq[0] = typename CONFIG_T::controller_metric_t(0);
        stability_lhs_raw[0] = typename CONFIG_T::controller_metric_t(0);
        stability_lhs_ctrl[0] = typename CONFIG_T::controller_metric_t(0);
        stability_rhs[0] = typename CONFIG_T::controller_metric_t(0);
        alpha_feasible[0] = typename CONFIG_T::controller_metric_t(1);
        alpha_state[0] = typename CONFIG_T::controller_metric_t(1);
        alpha_code[0] = typename CONFIG_T::controller_metric_t(32);
        alpha_min[0] = typename CONFIG_T::controller_metric_t(0.031250);
        feasible[0] = typename CONFIG_T::controller_metric_t(1);
    }


    // CTRL-GT-ORDER-0: division-free binary-search global throttle.
    //
    //  Replaces the algebraic safe-gain law α = χ / (η·C + ε) with an
    //  inequality comparison search over a table of binary-fraction alpha
    //  candidates.  No division, no sqrt — CSIM-safe with ap_fixed types.
    //
    //  Constraint (from the stability inequality η·α·C ≤ χ), using raw
    //  proposed update geometry:
    //
    //      α² · η² · ||ΔG||²  ≤  χ² · (||Δθ_raw||² + ε²)
    //
    //  Δθ_raw already includes the learning rate for SGD. We do not multiply
    //  Δθ_raw by η again.
    //
    //  Candidates are evaluated in descending order; the first (largest) alpha
    //  satisfying the inequality is selected.  If none satisfy, α = α_min.
    //
    //  State: none.  Curvature is recomputed fresh each batch_end from the
    //  global squared norms passed in.
    template<typename CONFIG_T>
    void global_throttle_order0_law(
        const typename CONFIG_T::controller_metric_t raw_update_norm_sq,
        const typename CONFIG_T::controller_metric_t dgrad_norm_sq,
        typename CONFIG_T::alpha_t alpha[1],
        typename CONFIG_T::controller_metric_t controller_raw_update_norm_sq[1],
        typename CONFIG_T::controller_metric_t controller_controlled_update_norm_sq[1],
        typename CONFIG_T::controller_metric_t controller_actual_update_norm_sq[1],
        typename CONFIG_T::controller_metric_t controller_dgrad_norm_sq[1],
        typename CONFIG_T::controller_metric_t controller_dtheta_for_control_sq[1],
        typename CONFIG_T::controller_metric_t controller_stability_lhs_raw[1],
        typename CONFIG_T::controller_metric_t controller_stability_lhs_ctrl[1],
        typename CONFIG_T::controller_metric_t controller_stability_rhs[1],
        typename CONFIG_T::controller_metric_t controller_alpha_feasible[1],
        typename CONFIG_T::controller_metric_t controller_alpha_state[1],
        typename CONFIG_T::controller_metric_t controller_alpha_code[1],
        typename CONFIG_T::controller_metric_t controller_alpha_min[1],
        typename CONFIG_T::controller_metric_t controller_feasible[1],
        bool reset_numerator
    ) {

        using metric_t = typename CONFIG_T::controller_metric_t;

        if (reset_numerator) {
            alpha[0] = typename CONFIG_T::alpha_t(1);
            reset_controller_metrics<CONFIG_T>(
                controller_raw_update_norm_sq, controller_controlled_update_norm_sq,
                controller_actual_update_norm_sq, controller_dgrad_norm_sq,
                controller_dtheta_for_control_sq, controller_stability_lhs_raw,
                controller_stability_lhs_ctrl, controller_stability_rhs,
                controller_alpha_feasible, controller_alpha_state,
                controller_alpha_code, controller_alpha_min, controller_feasible
            );
            HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_alpha_name, alpha, 1);
            return;
        }

        metric_t lr(CONFIG_T::learning_rate);
        metric_t chi(CONFIG_T::controller_chi);
        metric_t eps(CONFIG_T::controller_epsilon);
        metric_t lr_sq = lr * lr;
        metric_t chi_sq = chi * chi;
        metric_t eps_sq = eps * eps;

        metric_t rhs = chi_sq * (raw_update_norm_sq + eps_sq);
        metric_t lhs_base = lr_sq * dgrad_norm_sq;

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
        static const metric_t cand_code[11] = {
            metric_t(32), metric_t(28), metric_t(24), metric_t(20),
            metric_t(16), metric_t(12), metric_t(8),  metric_t(6),
            metric_t(4),  metric_t(2),  metric_t(1),
        };
        static const unsigned n_candidates = 11;

        metric_t alpha_min = cand_val[n_candidates - 1];
        if (metric_t(CONFIG_T::controller_alpha_min) > alpha_min) {
            alpha_min = metric_t(CONFIG_T::controller_alpha_min);
        }

        metric_t alpha_feasible = alpha_min;
        metric_t alpha_feasible_sq = alpha_min * alpha_min;
        metric_t alpha_code = cand_code[n_candidates - 1];
        metric_t feasible = metric_t(0);

        for (unsigned i = 0; i < n_candidates; i++) {
            #pragma HLS UNROLL
            metric_t lhs = cand_sq[i] * lhs_base;
            if (lhs <= rhs) {
                alpha_feasible = cand_val[i];
                alpha_feasible_sq = cand_sq[i];
                alpha_code = cand_code[i];
                feasible = metric_t(1);
                break;
            }
        }

        alpha[0] = typename CONFIG_T::alpha_t(alpha_feasible);
        controller_raw_update_norm_sq[0] = raw_update_norm_sq;
        controller_controlled_update_norm_sq[0] = alpha_feasible_sq * raw_update_norm_sq;
        controller_actual_update_norm_sq[0] = metric_t(0);
        controller_dgrad_norm_sq[0] = dgrad_norm_sq;
        controller_dtheta_for_control_sq[0] = raw_update_norm_sq;
        controller_stability_lhs_raw[0] = lhs_base;
        controller_stability_lhs_ctrl[0] = alpha_feasible_sq * lhs_base;
        controller_stability_rhs[0] = rhs;
        controller_alpha_feasible[0] = alpha_feasible;
        controller_alpha_state[0] = alpha_feasible;
        controller_alpha_code[0] = alpha_code;
        controller_alpha_min[0] = alpha_min;
        controller_feasible[0] = feasible;

        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_dgrad_norm_sq_name, &dgrad_norm_sq, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_raw_update_norm_sq_name, &raw_update_norm_sq, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_stability_lhs_raw_name, &lhs_base, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_stability_rhs_name, &rhs, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_alpha_feasible_name, &alpha_feasible, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_alpha_name, alpha, 1);

    } // global_throttle_order0_law


    // CTRL-GT-ORDER-1: division-free first-order alpha-state global throttle.
    //
    //  Same inequality search as GT-0, but alpha_state evolves smoothly toward
    //  the feasible candidate via a first-order attractor:
    //
    //      alpha_feasible = largest candidate satisfying  α²·η²·||ΔG||² ≤ χ²·(||Δθ_raw||² + ε²)
    //      α ← clip(α + k_α·(alpha_feasible - α),  α_min,  α_max)
    //
    //  No division, no sqrt — CSIM-safe with ap_fixed types.
    //
    //  State: α_state (persistent scalar).
    template<typename CONFIG_T>
    void global_throttle_order1_law(
        const typename CONFIG_T::controller_metric_t raw_update_norm_sq,
        const typename CONFIG_T::controller_metric_t dgrad_norm_sq,
        typename CONFIG_T::alpha_t alpha[1],
        typename CONFIG_T::controller_metric_t controller_raw_update_norm_sq[1],
        typename CONFIG_T::controller_metric_t controller_controlled_update_norm_sq[1],
        typename CONFIG_T::controller_metric_t controller_actual_update_norm_sq[1],
        typename CONFIG_T::controller_metric_t controller_dgrad_norm_sq[1],
        typename CONFIG_T::controller_metric_t controller_dtheta_for_control_sq[1],
        typename CONFIG_T::controller_metric_t controller_stability_lhs_raw[1],
        typename CONFIG_T::controller_metric_t controller_stability_lhs_ctrl[1],
        typename CONFIG_T::controller_metric_t controller_stability_rhs[1],
        typename CONFIG_T::controller_metric_t controller_alpha_feasible[1],
        typename CONFIG_T::controller_metric_t controller_alpha_state[1],
        typename CONFIG_T::controller_metric_t controller_alpha_code[1],
        typename CONFIG_T::controller_metric_t controller_alpha_min[1],
        typename CONFIG_T::controller_metric_t controller_feasible[1],
        bool reset_numerator
    ) {

        using metric_t = typename CONFIG_T::controller_metric_t;
        static metric_t alpha_state = metric_t(1);

        if (reset_numerator) {
            alpha_state = metric_t(1);
            alpha[0] = typename CONFIG_T::alpha_t(1);
            reset_controller_metrics<CONFIG_T>(
                controller_raw_update_norm_sq, controller_controlled_update_norm_sq,
                controller_actual_update_norm_sq, controller_dgrad_norm_sq,
                controller_dtheta_for_control_sq, controller_stability_lhs_raw,
                controller_stability_lhs_ctrl, controller_stability_rhs,
                controller_alpha_feasible, controller_alpha_state,
                controller_alpha_code, controller_alpha_min, controller_feasible
            );
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

        metric_t rhs = chi_sq * (raw_update_norm_sq + eps_sq);
        metric_t lhs_base = lr_sq * dgrad_norm_sq;

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
        static const metric_t cand_code[11] = {
            metric_t(32), metric_t(28), metric_t(24), metric_t(20),
            metric_t(16), metric_t(12), metric_t(8),  metric_t(6),
            metric_t(4),  metric_t(2),  metric_t(1),
        };
        static const unsigned n_candidates = 11;

        metric_t alpha_min = cand_val[n_candidates - 1];
        if (metric_t(CONFIG_T::controller_alpha_min) > alpha_min) {
            alpha_min = metric_t(CONFIG_T::controller_alpha_min);
        }

        metric_t alpha_feasible = alpha_min;
        metric_t alpha_feasible_sq = alpha_min * alpha_min;
        metric_t alpha_code = cand_code[n_candidates - 1];
        metric_t feasible = metric_t(0);
        for (unsigned i = 0; i < n_candidates; i++) {
            #pragma HLS UNROLL
            metric_t lhs = cand_sq[i] * lhs_base;
            if (lhs <= rhs) {
                alpha_feasible = cand_val[i];
                alpha_feasible_sq = cand_sq[i];
                alpha_code = cand_code[i];
                feasible = metric_t(1);
                break;
            }
        }

        metric_t alpha_next = alpha_state + k_alpha * (alpha_feasible - alpha_state);

        metric_t alpha_max = metric_t(CONFIG_T::controller_alpha_max);

        if (alpha_next < alpha_min) {
            alpha_state = alpha_min;
        } else if (alpha_next > alpha_max) {
            alpha_state = alpha_max;
        } else {
            alpha_state = alpha_next;
        }

        alpha[0] = typename CONFIG_T::alpha_t(alpha_state);
        metric_t alpha_state_sq = alpha_state * alpha_state;
        controller_raw_update_norm_sq[0] = raw_update_norm_sq;
        controller_controlled_update_norm_sq[0] = alpha_state_sq * raw_update_norm_sq;
        controller_actual_update_norm_sq[0] = metric_t(0);
        controller_dgrad_norm_sq[0] = dgrad_norm_sq;
        controller_dtheta_for_control_sq[0] = raw_update_norm_sq;
        controller_stability_lhs_raw[0] = lhs_base;
        controller_stability_lhs_ctrl[0] = alpha_state_sq * lhs_base;
        controller_stability_rhs[0] = rhs;
        controller_alpha_feasible[0] = alpha_feasible;
        controller_alpha_state[0] = alpha_state;
        controller_alpha_code[0] = alpha_code;
        controller_alpha_min[0] = alpha_min;
        controller_feasible[0] = feasible;

        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_dgrad_norm_sq_name, &dgrad_norm_sq, 1);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_raw_update_norm_sq_name, &raw_update_norm_sq, 1);
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
        const typename CONFIG_T::alpha_t alpha[1],
        typename CONFIG_T::controller_metric_t &actual_update_norm_sq_contrib
    ) {

        constexpr unsigned n_out = CONFIG_T::n_out;
        constexpr unsigned n_weights = CONFIG_T::n_in * CONFIG_T::n_out;

        using weight_t = typename CONFIG_T::weight_t;
        using bias_t = typename CONFIG_T::bias_t;
        using update_t = typename CONFIG_T::update_t;
        using metric_t = typename CONFIG_T::controller_metric_t;

        actual_update_norm_sq_contrib = metric_t(0);

        // Convert through update_t before assigning back to the stored parameter
        // type. That keeps update precision separate from parameter precision.
        WeightApplyUpdate:
        for (unsigned i = 0; i < n_weights; i++) {
            #pragma HLS PIPELINE II=1
            weight_t old_weight = weights[i];
            update_t throttled_update = update_t(alpha[0] * weight_update[i]);
            weight_t new_weight = weight_t(old_weight + throttled_update);
            weights[i] = new_weight;
            metric_t actual_update = metric_t(new_weight) - metric_t(old_weight);
            actual_update_norm_sq_contrib += actual_update * actual_update;
        }

        BiasApplyUpdate:
        for (unsigned i = 0; i < n_out; i++) {
            #pragma HLS PIPELINE II=1
            bias_t old_bias = biases[i];
            update_t throttled_update = update_t(alpha[0] * bias_update[i]);
            bias_t new_bias = bias_t(old_bias + throttled_update);
            biases[i] = new_bias;
            metric_t actual_update = metric_t(new_bias) - metric_t(old_bias);
            actual_update_norm_sq_contrib += actual_update * actual_update;
        }

        // Inactive unless HLS4ML_TRAINABLE_TRACE is defined by generated code.
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_weights_after_update_name, weights, n_weights);
        HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_biases_after_update_name, biases, n_out);

    } // apply_dense_update

} // namespace nnet

#endif
