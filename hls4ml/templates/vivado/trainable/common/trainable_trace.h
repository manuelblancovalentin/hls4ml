#ifndef HLS4ML_TRAINABLE_TRACE_H_
#define HLS4ML_TRAINABLE_TRACE_H_

#if !defined(__SYNTHESIS__) && defined(HLS4ML_TRAINABLE_TRACE)
#include "../../nnet_utils/nnet_helpers.h"
#define HLS4ML_TRAINABLE_TRACE_ARRAY(name, data, size) nnet::save_layer_output(data, name, size)
#else
#define HLS4ML_TRAINABLE_TRACE_ARRAY(name, data, size) \
    do {                                               \
    } while (0)
#endif

#endif
