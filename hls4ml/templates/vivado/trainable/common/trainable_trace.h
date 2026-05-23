#ifndef HLS4ML_TRAINABLE_TRACE_H_
#define HLS4ML_TRAINABLE_TRACE_H_

// Trainable kernels use this macro for optional debug traces during C simulation.
//
// This does nothing in synthesis, and it also does nothing in normal C simulation
// unless the generated project defines HLS4ML_TRAINABLE_TRACE. When enabled, the
// macro reuses hls4ml's native nnet::save_layer_output path, so trainable traces
// are collected through the same bridge/testbench storage as forward activations.
//
// The `name` argument is intentionally expected to come from CONFIG_T. That keeps
// string naming in the generated config/writer layer instead of hard-coding names
// inside static kernels.
#if !defined(__SYNTHESIS__) && defined(HLS4ML_TRAINABLE_TRACE)
#include "../../nnet_utils/nnet_helpers.h"
#define HLS4ML_TRAINABLE_TRACE_ARRAY(name, data, size) nnet::save_layer_output(data, name, size)
#else
#define HLS4ML_TRAINABLE_TRACE_ARRAY(name, data, size) \
    do {                                               \
    } while (0)
#endif

#endif
