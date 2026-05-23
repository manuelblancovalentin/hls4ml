import glob
import os
import stat
import tarfile
from collections import OrderedDict
from pathlib import Path
from shutil import copyfile, copytree, rmtree

import numpy as np
import yaml

from hls4ml.writer.writers import Writer

config_filename = 'hls4ml_config.yml'


class VivadoWriter(Writer):
    @staticmethod
    def _is_trainable_model(model):
        return hasattr(model.config, 'is_trainable') and model.config.is_trainable()

    @staticmethod
    def _trainable_type_name(layer, attr_name, default_type_name=None):
        type_attr = layer.get_attr(attr_name, None)
        if type_attr is not None:
            return type_attr.name
        if default_type_name is not None:
            return default_type_name
        raise Exception(f'Trainable layer {layer.name} is missing required type attribute {attr_name}.')

    @staticmethod
    def _trainable_trace_name(owner_name, signal_name):
        return f'{owner_name}_{signal_name}'

    @staticmethod
    def _trainable_dense_config_name(layer):
        return f'trainable_config{layer.index}'

    @staticmethod
    def _trainable_layer_signal(layer, signal):
        return f'{layer.name}_{signal}'

    @staticmethod
    def _batch_size_log2(training_config):
        if training_config.get('BatchSizeLog2') is not None:
            return int(training_config['BatchSizeLog2'])

        batch_size = int(training_config.get('BatchSize', 1))
        if batch_size < 1 or batch_size & (batch_size - 1):
            raise Exception(
                'Trainable hls4ml requires a power-of-two BatchSize or an explicit BatchSizeLog2 '
                f'for firmware generation, got BatchSize={batch_size}.'
            )

        return batch_size.bit_length() - 1

    @staticmethod
    def _normalize_trainable_choice(value):
        if value is None:
            return None
        return str(value).strip().lower().replace('-', '_')

    def _make_trainable_loss_config(self, model, endpoint):
        output_layer = model.graph[endpoint['loss_input_layer']]
        output_variable = model.get_layer_output_variable(endpoint['loss_input_name'])
        data_in_t = output_variable.type.name
        ground_truth_t = output_layer.get_attr('ground_truth_t', None)
        ground_truth_t = ground_truth_t.name if ground_truth_t is not None else data_in_t
        loss_t = self._trainable_type_name(output_layer, 'loss_t', data_in_t)
        loss_grad_t = self._trainable_type_name(
            output_layer, 'loss_grad_t', self._trainable_type_name(output_layer, 'grad_in_t', data_in_t)
        )
        config_name = self._trainable_loss_config_name(endpoint)

        return f"""// Trainable loss endpoint {endpoint['index']}
struct {config_name} {{
    static const unsigned n_out = {endpoint['loss_input_size']};
    typedef {data_in_t} data_in_t;
    typedef {ground_truth_t} ground_truth_t;
    typedef {loss_t} loss_t;
    typedef {loss_grad_t} grad_out_t;
#if !defined(__SYNTHESIS__) && defined(HLS4ML_TRAINABLE_TRACE)
    static constexpr const char *trace_prediction_name = "{self._trainable_trace_name(endpoint['loss_input_name'], 'prediction')}";
    static constexpr const char *trace_ground_truth_name = "{self._trainable_trace_name(endpoint['ground_truth_name'], 'ground_truth')}";
    static constexpr const char *trace_loss_name = "{endpoint['loss_scalar_name']}";
    static constexpr const char *trace_loss_grad_name = "{endpoint['loss_gradient_name']}";
#endif
}};\n"""

    def _make_trainable_dense_config(self, model, layer):
        training_config = model.config.get_training_config()
        optimizer_config = model.config.get_optimizer_config()
        input_type = layer.get_input_variable().type.name
        output_type = layer.get_output_variable().type.name
        weight_type = layer.get_weights('weight').type.name
        bias_type = layer.get_weights('bias').type.name
        config_name = self._trainable_dense_config_name(layer)
        batch_size_log2 = self._batch_size_log2(training_config)
        learning_rate = optimizer_config.get('LearningRate')
        learning_rate_decl = ''
        if self._trainable_learning_rate_input_name(model) is None and learning_rate is not None:
            learning_rate_decl = f'    static constexpr double learning_rate = {learning_rate};\n'

        return f"""// Trainable Dense state for {layer.name}
struct {config_name} {{
    static const unsigned n_in = {layer.get_attr('n_in')};
    static const unsigned n_out = {layer.get_attr('n_out')};
    static const unsigned batch_size_log2 = {batch_size_log2};
{learning_rate_decl}    typedef {input_type} data_in_t;
    typedef {output_type} data_out_t;
    typedef {weight_type} weight_t;
    typedef {bias_type} bias_t;
    typedef {self._trainable_type_name(layer, 'grad_in_t', output_type)} grad_in_t;
    typedef {self._trainable_type_name(layer, 'grad_out_t', input_type)} grad_out_t;
    typedef {self._trainable_type_name(layer, 'weight_grad_t')} weight_grad_t;
    typedef {self._trainable_type_name(layer, 'bias_grad_t')} bias_grad_t;
    typedef {self._trainable_type_name(layer, 'gradient_accum_t')} gradient_accum_t;
    typedef {self._trainable_type_name(layer, 'raw_update_t')} raw_update_t;
    typedef {self._trainable_type_name(layer, 'update_t')} update_t;
    typedef {self._trainable_type_name(layer, 'optimizer_state_t')} optimizer_state_t;
    typedef {self._trainable_type_name(layer, 'controller_metric_t')} controller_metric_t;
    typedef {self._trainable_type_name(layer, 'alpha_t')} alpha_t;
    typedef {self._trainable_type_name(layer, 'learning_rate_t', self._trainable_type_name(layer, 'alpha_t'))} learning_rate_t;
#if !defined(__SYNTHESIS__) && defined(HLS4ML_TRAINABLE_TRACE)
    static constexpr const char *trace_data_in_name = "{self._trainable_trace_name(layer.name, 'backprop_data_in')}";
    static constexpr const char *trace_grad_in_name = "{self._trainable_trace_name(layer.name, 'backprop_grad_in')}";
    static constexpr const char *trace_grad_out_name = "{self._trainable_trace_name(layer.name, 'backprop_grad_out')}";
    static constexpr const char *trace_weight_grad_accum_name = "{self._trainable_trace_name(layer.name, 'weight_grad_accum')}";
    static constexpr const char *trace_bias_grad_accum_name = "{self._trainable_trace_name(layer.name, 'bias_grad_accum')}";
    static constexpr const char *trace_weight_grad_name = "{self._trainable_trace_name(layer.name, 'weight_grad')}";
    static constexpr const char *trace_bias_grad_name = "{self._trainable_trace_name(layer.name, 'bias_grad')}";
    static constexpr const char *trace_weight_update_name = "{self._trainable_trace_name(layer.name, 'weight_update')}";
    static constexpr const char *trace_bias_update_name = "{self._trainable_trace_name(layer.name, 'bias_update')}";
    static constexpr const char *trace_alpha_name = "{self._trainable_trace_name(layer.name, 'alpha')}";
    static constexpr const char *trace_weights_after_update_name = "{self._trainable_trace_name(layer.name, 'weights_after_update')}";
    static constexpr const char *trace_biases_after_update_name = "{self._trainable_trace_name(layer.name, 'biases_after_update')}";
#endif
}};\n"""

    def _make_trainable_configs(self, model):
        if not self._is_trainable_model(model):
            return ''

        configs = '\n// Trainable firmware configuration\n'

        for endpoint in getattr(model, 'trainable_loss_endpoints', ()):
            configs += self._make_trainable_loss_config(model, endpoint) + '\n'

        for layer_name in getattr(model, 'trainable_backward_order', ()):
            layer = model.graph[layer_name]
            if layer.class_name == 'Dense':
                configs += self._make_trainable_dense_config(model, layer) + '\n'

        return configs

    @staticmethod
    def _trainable_loss_config_name(endpoint):
        return f'trainable_loss_config{endpoint["index"]}'

    def _trainable_loss_type_name(self, model, endpoint):
        output_layer = model.graph[endpoint['loss_input_layer']]
        return self._trainable_type_name(output_layer, 'loss_t', endpoint['loss_input_type'])

    @staticmethod
    def _trainable_loss_prediction_name(model, endpoint):
        return model.get_layer_output_variable(endpoint['loss_input_name']).name

    def _make_trainable_internal_buffers(self, model):
        if not self._is_trainable_model(model):
            return ''

        lines = ['    // Trainable internal state and temporaries\n']

        for endpoint in getattr(model, 'trainable_loss_endpoints', ()):
            lines.append(
                '    {type} {name}[{size}];\n'.format(
                    type=endpoint['loss_gradient_type'],
                    name=endpoint['loss_gradient_name'],
                    size=f'{self._trainable_loss_config_name(endpoint)}::n_out',
                )
            )

        for layer_name in getattr(model, 'trainable_backward_order', ()):
            layer = model.graph[layer_name]
            if layer.class_name != 'Dense':
                continue

            prefix = layer.name
            config = self._trainable_dense_config_name(layer)
            n_in = f'{config}::n_in'
            n_out = f'{config}::n_out'
            n_weights = f'{config}::n_in * {config}::n_out'

            lines.append(f'    {self._trainable_type_name(layer, "grad_out_t")} {prefix}_grad_out[{n_in}];\n')
            lines.append(
                f'    static {self._trainable_type_name(layer, "gradient_accum_t")} '
                f'{prefix}_weight_grad_accum[{n_weights}];\n'
            )
            lines.append(
                f'    static {self._trainable_type_name(layer, "gradient_accum_t")} '
                f'{prefix}_bias_grad_accum[{n_out}];\n'
            )
            lines.append(f'    {self._trainable_type_name(layer, "weight_grad_t")} {prefix}_weight_grad[{n_weights}];\n')
            lines.append(f'    {self._trainable_type_name(layer, "bias_grad_t")} {prefix}_bias_grad[{n_out}];\n')
            lines.append(f'    {self._trainable_type_name(layer, "raw_update_t")} {prefix}_weight_update[{n_weights}];\n')
            lines.append(f'    {self._trainable_type_name(layer, "raw_update_t")} {prefix}_bias_update[{n_out}];\n')

        lines.append('\n')
        return ''.join(lines)

    def _trainable_learning_rate_input_name(self, model):
        optimizer_config = model.config.get_optimizer_config()
        learning_rate_input = optimizer_config.get('LearningRateInput')
        if not learning_rate_input:
            return None
        if isinstance(learning_rate_input, str):
            return learning_rate_input
        return 'learning_rate'

    def _trainable_learning_rate_expr(self, model, layer):
        learning_rate_input = self._trainable_learning_rate_input_name(model)
        if learning_rate_input is not None:
            return learning_rate_input

        optimizer_config = model.config.get_optimizer_config()
        learning_rate = optimizer_config.get('LearningRate')
        if learning_rate is None:
            raise Exception('Trainable hls4ml requires a static or input learning rate for SGD emission.')

        config_name = self._trainable_dense_config_name(layer)
        return f'{config_name}::learning_rate_t({config_name}::learning_rate)'

    def _make_trainable_top_level_ports(self, model):
        if not self._is_trainable_model(model):
            return []

        ports = []
        for endpoint in getattr(model, 'trainable_loss_endpoints', ()):
            ports.append(
                {
                    'type': endpoint['loss_input_type'],
                    'name': endpoint['ground_truth_name'],
                    'size': endpoint['loss_input_size'],
                    'direction': 'input',
                }
            )
            ports.append(
                {
                    'type': self._trainable_loss_type_name(model, endpoint),
                    'name': endpoint['loss_scalar_name'],
                    'size': 1,
                    'direction': 'output',
                }
            )

        if getattr(model, 'trainable_backward_order', ()):
            first_layer = model.graph[model.trainable_backward_order[0]]
            ports.append(
                {
                    'type': self._trainable_type_name(first_layer, 'alpha_t'),
                    'name': 'trainable_alpha',
                    'size': 1,
                    'direction': 'output',
                }
            )

            learning_rate_input = self._trainable_learning_rate_input_name(model)
            if learning_rate_input is not None:
                ports.append(
                    {
                        'type': self._trainable_type_name(first_layer, 'learning_rate_t', self._trainable_type_name(first_layer, 'alpha_t')),
                        'name': learning_rate_input,
                        'size': None,
                        'direction': 'input',
                    }
                )

        ports.extend(
            [
                {'type': 'bool', 'name': 'train_enable', 'size': None, 'direction': 'input'},
                {'type': 'bool', 'name': 'reset_accumulators', 'size': None, 'direction': 'input'},
                {'type': 'bool', 'name': 'batch_end', 'size': None, 'direction': 'input'},
            ]
        )

        return ports

    @staticmethod
    def _port_definition(port):
        if port['size'] is None:
            return f"{port['type']} {port['name']}"
        return f"{port['type']} {port['name']}[{port['size']}]"

    @staticmethod
    def _port_definition_bridge(port, dtype):
        return f'{dtype} *{port["name"]}'

    @staticmethod
    def _port_call_name(port):
        return port['name']

    def _make_trainable_top_level_call_args(self, model):
        return [self._port_call_name(port) for port in self._make_trainable_top_level_ports(model)]

    def _make_top_level_header(self, model, model_inputs, model_outputs, model_brams, indent):
        port_defs = [i.definition_cpp(as_reference=True) for i in model_inputs]
        port_defs.extend(o.definition_cpp(as_reference=True) for o in model_outputs)
        port_defs.extend(self._port_definition(port) for port in self._make_trainable_top_level_ports(model))
        port_defs.extend(indent + b.definition_cpp(as_reference=False) for b in model_brams)

        return (',\n' + indent).join(port_defs) + '\n'

    def _make_trainable_testbench_data(self, model, indent, index_name):
        if not self._is_trainable_model(model):
            return ''

        lines = []
        for endpoint in getattr(model, 'trainable_loss_endpoints', ()):
            lines.append(
                indent
                + '{type} {name}[{size}];\n'.format(
                    type=endpoint['loss_input_type'],
                    name=endpoint['ground_truth_name'],
                    size=endpoint['loss_input_size'],
                )
            )
            lines.append(
                indent
                + 'nnet::copy_data<float, {type}, 0, {size}>(pr, {name});\n'.format(
                    type=endpoint['loss_input_type'],
                    size=endpoint['loss_input_size'],
                    name=endpoint['ground_truth_name'],
                )
            )
            lines.append(
                indent
                + '{type} {name}[1];\n'.format(
                    type=self._trainable_loss_type_name(model, endpoint),
                    name=endpoint['loss_scalar_name'],
                )
            )

        if getattr(model, 'trainable_backward_order', ()):
            first_layer = model.graph[model.trainable_backward_order[0]]
            lines.append(indent + f'{self._trainable_type_name(first_layer, "alpha_t")} trainable_alpha[1];\n')

            learning_rate_input = self._trainable_learning_rate_input_name(model)
            if learning_rate_input is not None:
                learning_rate = model.config.get_optimizer_config().get('LearningRate', 0)
                lines.append(
                    indent
                    + f'{self._trainable_type_name(first_layer, "learning_rate_t", self._trainable_type_name(first_layer, "alpha_t"))} '
                    + f'{learning_rate_input} = {learning_rate};\n'
                )

        batch_size = int(model.config.get_training_config().get('BatchSize', 1))
        lines.append(indent + 'bool train_enable = true;\n')
        lines.append(indent + f'bool reset_accumulators = (({index_name} % {batch_size}) == 0);\n')
        lines.append(indent + f'bool batch_end = ((({index_name} + 1) % {batch_size}) == 0);\n')

        return ''.join(lines)

    def _make_trainable_zero_data(self, model, indent, index_name):
        if not self._is_trainable_model(model):
            return ''

        lines = []
        for endpoint in getattr(model, 'trainable_loss_endpoints', ()):
            lines.append(
                indent
                + '{type} {name}[{size}];\n'.format(
                    type=endpoint['loss_input_type'],
                    name=endpoint['ground_truth_name'],
                    size=endpoint['loss_input_size'],
                )
            )
            lines.append(
                indent
                + 'nnet::fill_zero<{type}, {size}>({name});\n'.format(
                    type=endpoint['loss_input_type'],
                    size=endpoint['loss_input_size'],
                    name=endpoint['ground_truth_name'],
                )
            )
            lines.append(
                indent
                + '{type} {name}[1];\n'.format(
                    type=self._trainable_loss_type_name(model, endpoint),
                    name=endpoint['loss_scalar_name'],
                )
            )

        if getattr(model, 'trainable_backward_order', ()):
            first_layer = model.graph[model.trainable_backward_order[0]]
            lines.append(indent + f'{self._trainable_type_name(first_layer, "alpha_t")} trainable_alpha[1];\n')

            learning_rate_input = self._trainable_learning_rate_input_name(model)
            if learning_rate_input is not None:
                learning_rate = model.config.get_optimizer_config().get('LearningRate', 0)
                lines.append(
                    indent
                    + f'{self._trainable_type_name(first_layer, "learning_rate_t", self._trainable_type_name(first_layer, "alpha_t"))} '
                    + f'{learning_rate_input} = {learning_rate};\n'
                )

        batch_size = int(model.config.get_training_config().get('BatchSize', 1))
        lines.append(indent + 'bool train_enable = true;\n')
        lines.append(indent + f'bool reset_accumulators = (({index_name} % {batch_size}) == 0);\n')
        lines.append(indent + f'bool batch_end = ((({index_name} + 1) % {batch_size}) == 0);\n')

        return ''.join(lines)

    def _make_trainable_loss_log_expr(self, model):
        loss_names = [endpoint['loss_scalar_name'] for endpoint in getattr(model, 'trainable_loss_endpoints', ())]
        if not loss_names:
            return '0.0'

        return ' + '.join(f'(double){name}[0]' for name in loss_names)

    def _make_trainable_weight_trace_header(self, model):
        columns = ['epoch', 'sample', 'global_step', 'sample_index']
        for layer_name in getattr(model, 'trainable_forward_order', ()):
            layer = model.graph[layer_name]
            if layer.class_name != 'Dense':
                continue

            n_weights = layer.get_attr('n_in') * layer.get_attr('n_out')
            n_biases = layer.get_attr('n_out')
            columns.extend(f'{layer.name}_weight_{index}' for index in range(n_weights))
            columns.extend(f'{layer.name}_bias_{index}' for index in range(n_biases))

        return ','.join(columns)

    def _make_trainable_weight_trace_declarations(self, model):
        lines = []
        for layer_name in getattr(model, 'trainable_forward_order', ()):
            layer = model.graph[layer_name]
            if layer.class_name != 'Dense':
                continue

            weights = layer.get_weights('weight')
            biases = layer.get_weights('bias')
            n_weights = layer.get_attr('n_in') * layer.get_attr('n_out')
            n_biases = layer.get_attr('n_out')
            lines.append(f'extern {weights.type.name} {weights.name}[{n_weights}];\n')
            lines.append(f'extern {biases.type.name} {biases.name}[{n_biases}];\n')

        return ''.join(lines)

    def _make_trainable_weight_trace_values(self, model, indent):
        lines = []
        for layer_name in getattr(model, 'trainable_forward_order', ()):
            layer = model.graph[layer_name]
            if layer.class_name != 'Dense':
                continue

            weights = layer.get_weights('weight').name
            biases = layer.get_weights('bias').name
            n_weights = layer.get_attr('n_in') * layer.get_attr('n_out')
            n_biases = layer.get_attr('n_out')
            lines.append(indent + f'for (unsigned i = 0; i < {n_weights}; i++) {{\n')
            lines.append(indent + f'    fweights << "," << (double){weights}[i];\n')
            lines.append(indent + '}\n')
            lines.append(indent + f'for (unsigned i = 0; i < {n_biases}; i++) {{\n')
            lines.append(indent + f'    fweights << "," << (double){biases}[i];\n')
            lines.append(indent + '}\n')

        return ''.join(lines)

    def _make_trainable_result_logging(self, model, indent):
        lines = []
        for out in model.get_output_variables():
            lines.append(indent + f'nnet::print_result<{out.type.name}, {out.size_cpp()}>({out.name}, fout);\n')
        return ''.join(lines)

    def _write_trainable_test_bench(self, model, filedir):
        """Write an epoch-based C simulation testbench for trainable models."""

        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_test.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if getattr(var, 'storage', '').lower() == 'bram']
        training_config = model.config.get_training_config()
        epochs = int(training_config.get('Epochs', 1))
        shuffle = str(bool(training_config.get('Shuffle', True))).lower()
        shuffle_seed = int(training_config.get('ShuffleSeed', 13))
        log_every = int(training_config.get('LogEvery', 1))
        optimizer_config = model.config.get_optimizer_config()
        controller_config = model.config.get_controller_config()
        loss_config = model.config.get_loss_config()
        metadata = training_config.get('Metadata', {})
        generated_by = metadata.get('GeneratedBy', 'hls4ml-trainable')
        enabol_version = metadata.get('EnabolVersion', 'unknown')
        hls4ml_trainable_version = metadata.get('Hls4mlTrainableVersion', '0.0.0a')
        learning_rate = optimizer_config.get('LearningRate', 'dynamic')
        controller_kind = controller_config.get('Kind', 'none')
        optimizer_kind = optimizer_config.get('Kind', 'sgd')
        loss_kind = loss_config.get('Kind', 'unknown')

        namespace = model.config.get_writer_config().get('Namespace', None)
        namespace_line = f'    using namespace {namespace};\n' if namespace is not None else ''
        bram_includes = ''.join(f'#include "firmware/weights/{bram.name}.h"\n' for bram in model_brams)

        data_lines = []
        offset = 0
        for inp in model_inputs:
            data_lines.append('            ' + inp.definition_cpp() + ';\n')
            data_lines.append(
                '            nnet::copy_data<float, {}, {}, {}>(in, {});\n'.format(
                    inp.type.name, offset, inp.size_cpp(), inp.name
                )
            )
            offset += inp.size()
        for out in model_outputs:
            data_lines.append('            ' + out.definition_cpp() + ';\n')
        data_lines.append(self._make_trainable_testbench_data(model, '            ', 'sample_step'))

        zero_lines = []
        for inp in model_inputs:
            zero_lines.append('            ' + inp.definition_cpp() + ';\n')
            zero_lines.append('            ' + f'nnet::fill_zero<{inp.type.name}, {inp.size_cpp()}>({inp.name});\n')
        for out in model_outputs:
            zero_lines.append('            ' + out.definition_cpp() + ';\n')
        zero_lines.append(self._make_trainable_zero_data(model, '            ', 'sample_step'))

        input_vars = ','.join([i.name for i in model_inputs])
        output_vars = ','.join([o.name for o in model_outputs])
        bram_vars = ','.join([b.name for b in model_brams])
        trainable_vars = ','.join(self._make_trainable_top_level_call_args(model))
        all_vars = ','.join(filter(None, [input_vars, output_vars, trainable_vars, bram_vars]))
        top_call = f'{model.config.get_project_name()}({all_vars});'

        loss_expr = self._make_trainable_loss_log_expr(model)
        result_logging = self._make_trainable_result_logging(model, '            ')
        weight_trace_header = self._make_trainable_weight_trace_header(model)
        weight_trace_declarations = self._make_trainable_weight_trace_declarations(model)
        weight_trace_values = self._make_trainable_weight_trace_values(model, '            ')

        fout.write(
            f"""#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <numeric>
#include <random>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "firmware/{model.config.get_project_name()}.h"
#include "firmware/nnet_utils/nnet_helpers.h"

{bram_includes}
{weight_trace_declarations}
#define CHECKPOINT 5000
#define TRAINABLE_BATCH_SIZE {int(training_config.get('BatchSize', 1))}
#define TRAINABLE_NUM_EPOCHS {epochs}
#define TRAINABLE_SHUFFLE {shuffle}
#define TRAINABLE_SHUFFLE_SEED {shuffle_seed}
#define TRAINABLE_LOG_EVERY {log_every}

namespace nnet {{
bool trace_enabled = true;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
}} // namespace nnet

struct TrainableSample {{
    std::string input;
    std::string target;
}};

static std::vector<float> parse_floats(const std::string &line) {{
    std::vector<float> values;
    std::istringstream stream(line);
    float value;
    while (stream >> value) {{
        values.push_back(value);
    }}
    return values;
}}

static const char *env_or_unknown(const char *name) {{
    const char *value = std::getenv(name);
    return value == NULL ? "unknown" : value;
}}

static std::string current_datetime() {{
    std::time_t now = std::time(NULL);
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buffer);
}}

static void write_trainable_metadata(std::ofstream &stream, const char *trace_name, const std::string &datetime) {{
    stream << "# ----" << std::endl;
    stream << "# Trace: " << trace_name << std::endl;
    stream << "# Generated by {generated_by}" << std::endl;
    stream << "# ENABOL version: {enabol_version}" << std::endl;
    stream << "# hls4ml-trainable version: {hls4ml_trainable_version}" << std::endl;
    stream << "# User: " << env_or_unknown("USER") << std::endl;
    stream << "# Host: " << env_or_unknown("HOSTNAME") << std::endl;
    stream << "# Date: " << datetime << std::endl;
    stream << "# Project: {model.config.get_project_name()}" << std::endl;
    stream << "# Backend: {model.config.get_config_value('Backend', 'unknown')}" << std::endl;
    stream << "# Controller: {controller_kind}" << std::endl;
    stream << "# Optimizer: {optimizer_kind}" << std::endl;
    stream << "# LearningRate: {learning_rate}" << std::endl;
    stream << "# Loss: {loss_kind}" << std::endl;
    stream << "# BatchSize: " << TRAINABLE_BATCH_SIZE << std::endl;
    stream << "# Epochs: " << TRAINABLE_NUM_EPOCHS << std::endl;
    stream << "# Shuffle: " << (TRAINABLE_SHUFFLE ? "true" : "false") << std::endl;
    stream << "# ShuffleSeed: " << TRAINABLE_SHUFFLE_SEED << std::endl;
    stream << "# LogEvery: " << TRAINABLE_LOG_EVERY << std::endl;
    stream << "# ---" << std::endl;
}}

int main(int argc, char **argv) {{
{namespace_line}
    const std::string run_datetime = current_datetime();

    std::ifstream fin("tb_data/tb_input_features.dat");
    std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
    std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
    std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
    std::ofstream fout(RESULTS_LOG);
    std::ofstream floss("tb_data/training/loss.dat");
    std::ofstream falpha("tb_data/training/alpha.dat");
    std::ofstream fweights("tb_data/training/weights.dat");
    write_trainable_metadata(floss, "loss", run_datetime);
    floss << "epoch,sample,global_step,sample_index,loss" << std::endl;
    write_trainable_metadata(falpha, "alpha", run_datetime);
    falpha << "epoch,sample,global_step,sample_index,alpha" << std::endl;
    write_trainable_metadata(fweights, "weights", run_datetime);
    fweights << "{weight_trace_header}" << std::endl;

    std::cout << "============================================================" << std::endl;
    std::cout << "ENABOL + hls4ml-trainable CSIM training run" << std::endl;
    std::cout << "Generated by {generated_by}" << std::endl;
    std::cout << "ENABOL version: {enabol_version}, hls4ml-trainable version: {hls4ml_trainable_version}" << std::endl;
    std::cout << "Project: {model.config.get_project_name()} | Backend: {model.config.get_config_value('Backend', 'unknown')}" << std::endl;
    std::cout << "Controller: {controller_kind} | Optimizer: {optimizer_kind} | Loss: {loss_kind}" << std::endl;
    std::cout << "Learning rate: {learning_rate} | Batch size: " << TRAINABLE_BATCH_SIZE
              << " | Epochs: " << TRAINABLE_NUM_EPOCHS << std::endl;
    std::cout << "Shuffle: " << (TRAINABLE_SHUFFLE ? "true" : "false") << " | Shuffle seed: " << TRAINABLE_SHUFFLE_SEED
              << " | Log every: " << TRAINABLE_LOG_EVERY << std::endl;
    std::cout << "User: " << env_or_unknown("USER") << " | Host: " << env_or_unknown("HOSTNAME")
              << " | Date: " << run_datetime << std::endl;
    std::cout << "============================================================" << std::endl;

    std::vector<TrainableSample> samples;
    std::string iline;
    std::string pline;
    while (std::getline(fin, iline) && std::getline(fpr, pline)) {{
        samples.push_back({{iline, pline}});
    }}

    if (!samples.empty()) {{
        std::vector<unsigned> order(samples.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(TRAINABLE_SHUFFLE_SEED);
        unsigned long global_step = 0;

        for (unsigned epoch = 0; epoch < TRAINABLE_NUM_EPOCHS; epoch++) {{
            if (TRAINABLE_SHUFFLE) {{
                std::shuffle(order.begin(), order.end(), rng);
            }}

            double epoch_loss = 0.0;
            for (unsigned sample_step = 0; sample_step < order.size(); sample_step++) {{
                const unsigned sample_index = order[sample_step];
                std::vector<float> in = parse_floats(samples[sample_index].input);
                std::vector<float> pr = parse_floats(samples[sample_index].target);

{''.join(data_lines)}

            {top_call}

{result_logging}
                const double step_loss = {loss_expr};
                epoch_loss += step_loss;
                floss << (epoch + 1) << "," << (sample_step + 1) << "," << global_step << ","
                      << sample_index << "," << step_loss << std::endl;
                falpha << (epoch + 1) << "," << (sample_step + 1) << "," << global_step << ","
                       << sample_index << "," << (double)trainable_alpha[0] << std::endl;

                if (global_step % TRAINABLE_LOG_EVERY == 0) {{
                    std::cout << "Epoch [" << (epoch + 1) << "/" << TRAINABLE_NUM_EPOCHS
                              << "] - sample " << (sample_step + 1) << "/" << samples.size()
                              << ": loss " << step_loss << ", alpha: " << trainable_alpha[0]
                              << std::endl;
                }}
                global_step++;
            }}

            std::cout << "Trainable epoch " << epoch << " average loss "
                      << epoch_loss / samples.size() << std::endl;
            const long epoch_global_step = global_step == 0 ? 0 : (long)global_step - 1;
            fweights << (epoch + 1) << "," << samples.size() << "," << epoch_global_step << ",-1";
{weight_trace_values}            fweights << std::endl;
        }}
    }} else {{
        std::cout << "INFO: Unable to open input/predictions file, using default trainable input." << std::endl;
        const unsigned NUM_TEST_SAMPLES = 5;
        for (unsigned sample_step = 0; sample_step < NUM_TEST_SAMPLES; sample_step++) {{
{''.join(zero_lines)}

            {top_call}

{result_logging}
        }}
    }}

    fout.close();
    floss.close();
    falpha.close();
    fweights.close();
    std::cout << "INFO: Saved trainable inference results to file: " << RESULTS_LOG << std::endl;
    std::cout << "INFO: Saved trainable loss trace to file: tb_data/training/loss.dat" << std::endl;
    std::cout << "INFO: Saved trainable alpha trace to file: tb_data/training/alpha.dat" << std::endl;
    std::cout << "INFO: Saved trainable weights trace to file: tb_data/training/weights.dat" << std::endl;

    return 0;
}}
"""
        )
        fout.close()

    def _make_trainable_bridge_defaults(self, model, indent):
        if not self._is_trainable_model(model):
            return ''

        lines = []
        for endpoint in getattr(model, 'trainable_loss_endpoints', ()):
            lines.append(
                indent
                + '{type} {name}[{size}];\n'.format(
                    type=endpoint['loss_input_type'],
                    name=endpoint['ground_truth_name'],
                    size=endpoint['loss_input_size'],
                )
            )
            lines.append(
                indent
                + 'nnet::fill_zero<{type}, {size}>({name});\n'.format(
                    type=endpoint['loss_input_type'],
                    size=endpoint['loss_input_size'],
                    name=endpoint['ground_truth_name'],
                )
            )
            lines.append(
                indent
                + '{type} {name}[1];\n'.format(
                    type=self._trainable_loss_type_name(model, endpoint),
                    name=endpoint['loss_scalar_name'],
                )
            )

        if getattr(model, 'trainable_backward_order', ()):
            first_layer = model.graph[model.trainable_backward_order[0]]
            lines.append(indent + f'{self._trainable_type_name(first_layer, "alpha_t")} trainable_alpha[1];\n')

            learning_rate_input = self._trainable_learning_rate_input_name(model)
            if learning_rate_input is not None:
                lines.append(
                    indent
                    + f'{self._trainable_type_name(first_layer, "learning_rate_t", self._trainable_type_name(first_layer, "alpha_t"))} '
                    + f'{learning_rate_input} = 0;\n'
                )

        lines.append(indent + 'bool train_enable = false;\n')
        lines.append(indent + 'bool reset_accumulators = false;\n')
        lines.append(indent + 'bool batch_end = false;\n')

        return ''.join(lines)

    def _make_trainable_call_chain(self, model):
        if not self._is_trainable_model(model):
            return ''

        controller_kind = self._normalize_trainable_choice(model.config.get_controller_config().get('Kind', 'none'))
        if controller_kind not in ['none', 'ctrl_none']:
            raise Exception(
                'Trainable Vivado writer currently emits only CTRL-NONE. '
                f'Controller {model.config.get_controller_config().get("Kind")} is not wired yet.'
            )

        lines = ['    // Trainable loss, backprop, optimizer, and update path\n', '    if (train_enable) {\n']

        for endpoint in getattr(model, 'trainable_loss_endpoints', ()):
            lines.append(
                '        nnet::{loss}<{config}>({prediction}, {truth}, {loss_out}, {loss_grad});\n'.format(
                    loss=endpoint['effective_loss_name'],
                    config=self._trainable_loss_config_name(endpoint),
                    prediction=self._trainable_loss_prediction_name(model, endpoint),
                    truth=endpoint['ground_truth_name'],
                    loss_out=endpoint['loss_scalar_name'],
                    loss_grad=endpoint['loss_gradient_name'],
                )
            )

        backward_order = getattr(model, 'trainable_backward_order', ())
        for index, layer_name in enumerate(backward_order):
            layer = model.graph[layer_name]
            if layer.class_name != 'Dense':
                continue

            grad_in = model.trainable_loss_endpoints[0]['loss_gradient_name'] if index == 0 else f'{backward_order[index - 1]}_grad_out'
            prefix = layer.name
            lines.append(
                '        nnet::dense_backpass<{config}>({data_in}, {grad_in}, {weights}, {grad_out}, '
                '{weight_accum}, {bias_accum}, {weight_grad}, {bias_grad}, reset_accumulators, batch_end);\n'.format(
                    config=self._trainable_dense_config_name(layer),
                    data_in=layer.get_input_variable().name,
                    grad_in=grad_in,
                    weights=layer.get_weights('weight').name,
                    grad_out=f'{prefix}_grad_out',
                    weight_accum=f'{prefix}_weight_grad_accum',
                    bias_accum=f'{prefix}_bias_grad_accum',
                    weight_grad=f'{prefix}_weight_grad',
                    bias_grad=f'{prefix}_bias_grad',
                )
            )

        if backward_order:
            lines.append('        if (batch_end) {\n')
            for layer_name in backward_order:
                layer = model.graph[layer_name]
                if layer.class_name != 'Dense':
                    continue
                prefix = layer.name
                lines.append(
                    '            nnet::sgd<{config}>({weight_grad}, {bias_grad}, {weight_update}, {bias_update}, {learning_rate});\n'.format(
                        config=self._trainable_dense_config_name(layer),
                        weight_grad=f'{prefix}_weight_grad',
                        bias_grad=f'{prefix}_bias_grad',
                        weight_update=f'{prefix}_weight_update',
                        bias_update=f'{prefix}_bias_update',
                        learning_rate=self._trainable_learning_rate_expr(model, layer),
                    )
                )

            first_layer = model.graph[backward_order[0]]
            lines.append(f'            nnet::global_throttle_none<{self._trainable_dense_config_name(first_layer)}>(trainable_alpha);\n')

            for layer_name in backward_order:
                layer = model.graph[layer_name]
                if layer.class_name != 'Dense':
                    continue
                prefix = layer.name
                lines.append(
                    '            nnet::apply_dense_update<{config}>({weights}, {biases}, {weight_update}, {bias_update}, trainable_alpha);\n'.format(
                        config=self._trainable_dense_config_name(layer),
                        weights=layer.get_weights('weight').name,
                        biases=layer.get_weights('bias').name,
                        weight_update=f'{prefix}_weight_update',
                        bias_update=f'{prefix}_bias_update',
                    )
                )
            lines.append('        }\n')

        lines.append('    }\n\n')
        return ''.join(lines)

    def print_array_to_cpp(self, var, odir, namespace=None, write_txt_file=True):
        """Write a weights array to C++ header files.

        Args:
            var (WeightVariable): Weight to write
            odir (str): Output directory
            namespace (str, optional): Writes a namespace for the weights to avoid clashes with global variables.
            write_txt_file (bool, optional): Write txt files in addition to .h files. Defaults to True.
        """

        h_file = open(f'{odir}/firmware/weights/{var.name}.h', 'w')
        if write_txt_file:
            txt_file = open(f'{odir}/firmware/weights/{var.name}.txt', 'w')

        # meta data
        h_file.write(f'//Numpy array shape {var.shape}\n')
        h_file.write(f'//Min {np.min(var.min):.12f}\n')
        h_file.write(f'//Max {np.max(var.max):.12f}\n')
        h_file.write(f'//Number of zeros {var.nzeros}\n')
        h_file.write('\n')

        h_file.write(f'#ifndef {var.name.upper()}_H_\n')
        h_file.write(f'#define {var.name.upper()}_H_\n')
        h_file.write('\n')

        if namespace is not None:
            h_file.write(f'namespace {namespace} {{\n\n')

        if write_txt_file:
            h_file.write('#ifndef __SYNTHESIS__\n')
            h_file.write(var.definition_cpp() + ';\n')
            h_file.write('#else\n')

        h_file.write(var.definition_cpp() + ' = {')

        # fill c++ array.
        # not including internal brackets for multidimensional case
        sep = ''
        for x in var:
            h_file.write(sep + x)
            if write_txt_file:
                txt_file.write(sep + x)
            sep = ', '
        h_file.write('};\n\n')

        if write_txt_file:
            h_file.write('#endif\n')
            txt_file.close()

        if namespace is not None:
            h_file.write('}\n\n')

        h_file.write('\n#endif\n')
        h_file.close()

    def write_project_dir(self, model):
        """Write the base project directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if not os.path.isdir(f'{model.config.get_output_dir()}/firmware/weights'):
            os.makedirs(f'{model.config.get_output_dir()}/firmware/weights')

    @staticmethod
    def _make_array_pragma(variable):
        """
        Layers in hls_model.py can specify output array partitioning through the `pragma` attribute.
        If `pragma` is a string: options are 'partition', 'reshape', or 'stream'.
        If `pragma` is a tuple: (mode, type, factor) where mode is 'partition' or 'reshape', type is
        'complete', 'cyclic', or 'block', and factor is an integer only used when the type is not 'complete'.
        """

        config = variable.pragma
        if type(config) is tuple:
            mode = config[0]
            if mode in ['partition', 'reshape']:
                typ = config[1]
                if typ != 'complete':
                    factor = config[2]
            elif mode == 'stream':
                depth = config[1]
        else:
            mode = config
            typ = 'complete'
            factor = 0

        if mode in ['partition', 'reshape']:
            if typ == 'complete':
                template = '#pragma HLS ARRAY_{mode} variable={name} {type} dim={dim}'
            else:
                template = '#pragma HLS ARRAY_{mode} variable={name} {type} factor={factor} dim={dim}'

            return template.format(mode=mode.upper(), name=variable.name, type=typ, factor=factor, dim=0)

        elif mode == 'stream':
            return f'#pragma HLS STREAM variable={variable.name} depth={depth}'

    def write_project_cpp(self, model):
        """Write the main architecture source file (myproject.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))

        f = open(os.path.join(filedir, '../templates/vivado/firmware/myproject.cpp'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            # Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning insert header' in line:
                newline = indent + self._make_top_level_header(model, model_inputs, model_outputs, model_brams, indent)

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            elif '// hls-fpga-machine-learning insert load weights' in line:
                newline = line
                if model.config.get_writer_config()['WriteWeightsTxt']:
                    newline += '#ifndef __SYNTHESIS__\n'
                    newline += '    static bool loaded_weights = false;\n'
                    newline += '    if (!loaded_weights) {\n'

                    for layer in model.get_layers():
                        for w in layer.get_weights():
                            if w.weight_class == 'CompressedWeightVariable':
                                newline += (
                                    indent
                                    + '    nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                        w.type.name, w.nonzeros, w.name, w.name
                                    )
                                )
                            elif w.weight_class == 'ExponentWeightVariable':
                                newline += (
                                    indent
                                    + '    nnet::load_exponent_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                        w.type.name, w.data_length, w.name, w.name
                                    )
                                )
                            else:
                                newline += indent + '    nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                    w.type.name, w.data_length, w.name, w.name
                                )

                    newline += '        loaded_weights = true;'
                    newline += '    }\n'
                    newline += '#endif'

            # Add input/output type
            elif '// hls-fpga-machine-learning insert IO' in line:
                newline = line
                all_inputs = [i.name for i in model_inputs]
                all_outputs = [o.name for o in model_outputs]
                all_brams = [b.name for b in model_brams]
                trainable_ports = self._make_trainable_top_level_ports(model)
                trainable_port_names = [port['name'] for port in trainable_ports]
                io_type = model.config.get_config_value('IOType')

                pipeline_style = model.config.pipeline_style
                pipeline_ii = model.config.pipeline_ii
                pipeline_pragma = indent + f'#pragma HLS {pipeline_style.upper()}'
                if pipeline_style == 'pipeline' and pipeline_ii is not None:
                    pipeline_pragma += f' II={pipeline_ii}\n'
                else:
                    pipeline_pragma += '\n'

                if io_type == 'io_parallel':
                    for i in model_inputs:
                        newline += indent + self._make_array_pragma(i) + '\n'
                    for o in model_outputs:
                        newline += indent + self._make_array_pragma(o) + '\n'
                    for port in trainable_ports:
                        if port['size'] is not None:
                            newline += indent + '#pragma HLS ARRAY_PARTITION variable={} complete dim=0\n'.format(
                                port['name']
                            )
                    # TODO discussed adding a handle for setting the interface mode for individual input and output arrays
                    # Probably the handle doesn't need to be exposed to the user but should be just set in hls_model.py
                    newline += indent + '#pragma HLS INTERFACE ap_vld port={},{} \n'.format(
                        ','.join(all_inputs + trainable_port_names), ','.join(all_outputs)
                    )
                    newline += pipeline_pragma

                if io_type == 'io_stream':
                    newline += indent + '#pragma HLS INTERFACE axis port={},{} \n'.format(
                        ','.join(all_inputs + trainable_port_names), ','.join(all_outputs)
                    )
                    if all_brams:
                        newline += indent + '#pragma HLS INTERFACE bram port={} \n'.format(','.join(all_brams))
                    newline += pipeline_pragma

            elif '// hls-fpga-machine-learning insert layers' in line:
                newline = line + '\n'
                for layer in model.get_layers():
                    vars = layer.get_variables()
                    for var in vars:
                        if var not in model_inputs and var not in model_outputs:
                            def_cpp = var.definition_cpp()
                            if def_cpp is not None:
                                newline += '    ' + def_cpp + ';\n'
                                if var.pragma:
                                    newline += '    ' + self._make_array_pragma(var) + '\n\n'
                newline += self._make_trainable_internal_buffers(model)
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp', None)
                    if func:
                        if not isinstance(func, (list, set)):
                            func = [func]
                        if len(func) == 1:
                            newline += '    ' + func[0] + ' // ' + layer.name + '\n'
                        else:
                            newline += '    // ' + layer.name + '\n'
                            for line in func:
                                newline += '    ' + line + '\n'
                        if model.config.trace_output and layer.get_attr('trace', False):
                            vars = layer.get_variables()
                            newline += '#ifndef __SYNTHESIS__\n'
                            for var in vars:
                                newline += '    nnet::save_layer_output<{}>({}, "{}", {});\n'.format(
                                    var.type.name, var.name, layer.name, var.size_cpp()
                                )
                            newline += '#endif\n'
                        newline += '\n'
                newline += self._make_trainable_call_chain(model)

            # Just copy line
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_project_header(self, model):
        """Write the main architecture header file (myproject.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vivado/firmware/myproject.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.h', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))

            elif 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning insert header' in line:
                newline = indent + self._make_top_level_header(model, model_inputs, model_outputs, model_brams, indent)

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            elif '// hls-fpga-machine-learning insert emulator-defines' in line:
                newline = line

                if model.config.get_writer_config().get('WriteEmulationConstants', False):
                    brams_def_str = ', '.join([b.definition_cpp(as_reference=False) for b in model_brams])
                    brams_call_str = ', '.join([b.name for b in model_brams])

                    if model.config.get_config_value('IOType') == 'io_stream':
                        input_call_str = ', '.join([f'std::get<{n}>(inputs)' for n in range(len(model_inputs))])
                        output_call_str = ', '.join([f'std::get<{n}>(outputs)' for n in range(len(model_outputs))])
                    else:
                        input_call_str = ', '.join([f'std::get<{n}>(inputs).data()' for n in range(len(model_inputs))])
                        output_call_str = ', '.join([f'std::get<{n}>(outputs).data()' for n in range(len(model_outputs))])

                    newline += (
                        f'\ninline void {model.config.get_project_name()}_emulator('
                        'inputs_t& inputs, outputs_t& outputs'  # the inputs_t should ideally be const
                    )
                    if len(model_brams) > 0:
                        newline += ',\n' + brams_def_str
                    newline += ') {\n'
                    newline += indent + model.config.get_project_name() + '(\n'
                    newline += indent + indent + input_call_str + ',\n'
                    newline += indent + indent + output_call_str
                    if len(model_brams) > 0:
                        newline += ',\n' + indent + indent + brams_call_str
                    newline += '\n' + indent + ');\n}\n'

            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_defines(self, model):
        """Write the C++ type definitions file (defines.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vivado/firmware/defines.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/defines.h', 'w')

        for line in f.readlines():
            if '// hls-fpga-machine-learning insert headers' in line:
                uses_stdfloat = False
                uses_apfloat = False
                for layer in model.get_layers():
                    layer_precision = layer.get_layer_precision()
                    for type_var in layer_precision.values():
                        cpp = type_var.definition_cpp()
                        if 'std::' in cpp:
                            uses_stdfloat = True
                            break
                        if 'ap_float' in cpp:
                            uses_apfloat = True
                            break
                if uses_stdfloat:
                    newline = line + '#include <stdfloat>\n'
                if uses_apfloat:
                    newline = line + '#include "ap_float.h"\n'

            elif '// hls-fpga-machine-learning insert layer-precision' in line:
                newline = line
                all_precision = OrderedDict()
                for layer in model.get_layers():
                    layer_precision = layer.get_layer_precision()
                    for type_name, type_var in layer_precision.items():
                        # Ensure that layer's types doesn't override existing types
                        # This can happen in case of InplaceVariable types
                        if type_name not in all_precision:
                            all_precision[type_name] = type_var
                for used_type in all_precision.values():
                    newline += used_type.definition_cpp()

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            elif '// hls-fpga-machine-learning insert emulator-defines' in line:
                newline = line

                if model.config.get_writer_config().get('WriteEmulationConstants', False):
                    if model.config.get_config_value('IOType') == 'io_stream':
                        input_types = [f'hls::stream<{v.type.name}>' for v in model.get_input_variables()]
                        output_types = [f'hls::stream<{v.type.name}>' for v in model.get_output_variables()]
                    else:
                        input_types = [f'std::array<{v.type.name}, {v.size_cpp()}>' for v in model.get_input_variables()]
                        output_types = [f'std::array<{v.type.name}, {v.size_cpp()}>' for v in model.get_output_variables()]
                    input_types_str = ', '.join(input_types)
                    output_types_str = ', '.join(output_types)
                    newline += '\n' + f'using inputs_t = std::tuple<{input_types_str}>;'
                    newline += '\n' + f'using outputs_t = std::tuple<{output_types_str}>;\n'
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_parameters(self, model):
        """Write the C++ layer config file (parameters.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vivado/firmware/parameters.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/parameters.h', 'w')

        for line in f.readlines():
            if '// hls-fpga-machine-learning insert includes' in line:
                newline = line
                for include in sorted(set(sum((layer.get_attr('include_header', []) for layer in model.get_layers()), []))):
                    newline += '#include "%s"\n' % include
                if self._is_trainable_model(model):
                    newline += '#include "trainable/backprop/nnet_dense_backprop.h"\n'
                    newline += '#include "trainable/controllers/global_throttle.h"\n'
                    newline += '#include "trainable/losses/mse.h"\n'
                    newline += '#include "trainable/optimizers/sgd.h"\n'

            elif '// hls-fpga-machine-learning insert weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.storage.lower() != 'bram':
                            newline += f'#include "weights/{w.name}.h"\n'

            elif '// hls-fpga-machine-learning insert layer-config' in line:
                newline = line
                for layer in model.get_layers():
                    config = layer.get_attr('config_cpp', None)
                    if config:
                        newline += '// ' + layer.name + '\n'
                        newline += config + '\n'
                newline += self._make_trainable_configs(model)

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_weights(self, model):
        """Write the weights into header files

        Args:
            model (ModelGraph): the hls4ml model.
        """
        namespace = model.config.get_writer_config().get('Namespace', None)
        write_txt = model.config.get_writer_config().get('WriteWeightsTxt', True)
        for layer in model.get_layers():
            for weights in layer.get_weights():
                self.print_array_to_cpp(
                    weights, model.config.get_output_dir(), namespace=namespace, write_txt_file=write_txt
                )

    def write_multigraph_weights(self, model):
        """Write the weights into header files

        Args:
            model (MultiModelGraph): the hls4ml multigraph model.
        """
        namespace = model.config.get_writer_config().get('Namespace', None)
        write_txt = model.config.get_writer_config().get('WriteWeightsTxt', True)
        for g in model.graphs:
            for layer in g.get_layers():
                for weights in layer.get_weights():
                    self.print_array_to_cpp(
                        weights, model.config.get_output_dir(), namespace=namespace, write_txt_file=write_txt
                    )

    def __make_dat_file(self, original_path, project_path):
        """
        Convert other input/output data types into a dat file, which is
        a text file with the falttened matrix printed out. Note that ' ' is
        assumed to be the delimiter.
        """

        # Take in data from current supported data files
        if original_path[-3:] == 'npy':
            data = np.load(original_path)
        else:
            raise Exception('Unsupported input/output data files.')

        # Faltten data, just keep first dimension
        data = data.reshape(data.shape[0], -1)

        def print_data(f):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    f.write(str(data[i][j]) + ' ')
                f.write('\n')

        # Print out in dat file
        with open(project_path, 'w') as f:
            print_data(f)

    def write_test_bench(self, model):
        """Write the testbench files (myproject_test.cpp and input/output .dat files)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))

        if not os.path.exists(f'{model.config.get_output_dir()}/tb_data/'):
            os.mkdir(f'{model.config.get_output_dir()}/tb_data/')

        input_data = model.config.get_config_value('InputData')
        output_predictions = model.config.get_config_value('OutputPredictions')

        if input_data:
            if input_data[-3:] == 'dat':
                copyfile(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')
            else:
                self.__make_dat_file(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')

        if output_predictions:
            if output_predictions[-3:] == 'dat':
                copyfile(output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat')
            else:
                self.__make_dat_file(
                    output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat'
                )

        if self._is_trainable_model(model):
            os.makedirs(f'{model.config.get_output_dir()}/tb_data/training/', exist_ok=True)
            self._write_trainable_test_bench(model, filedir)
            return

        f = open(os.path.join(filedir, '../templates/vivado/myproject_test.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_test.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            # Insert numbers
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include "firmware/weights/{bram.name}.h"\n'

            elif '// hls-fpga-machine-learning insert data' in line:
                newline = line
                offset = 0
                for inp in model_inputs:
                    newline += '      ' + inp.definition_cpp() + ';\n'
                    newline += '      nnet::copy_data<float, {}, {}, {}>(in, {});\n'.format(
                        inp.type.name, offset, inp.size_cpp(), inp.name
                    )
                    offset += inp.size()
                for out in model_outputs:
                    newline += '      ' + out.definition_cpp() + ';\n'
                newline += self._make_trainable_testbench_data(model, '      ', 'e')

            elif '// hls-fpga-machine-learning insert zero' in line:
                newline = line
                for inp in model_inputs:
                    newline += indent + inp.definition_cpp() + ';\n'
                    newline += indent + f'nnet::fill_zero<{inp.type.name}, {inp.size_cpp()}>({inp.name});\n'
                for out in model_outputs:
                    newline += indent + out.definition_cpp() + ';\n'
                newline += self._make_trainable_zero_data(model, indent, 'i')

            elif '// hls-fpga-machine-learning insert top-level-function' in line:
                newline = line

                input_vars = ','.join([i.name for i in model_inputs])
                output_vars = ','.join([o.name for o in model_outputs])
                bram_vars = ','.join([b.name for b in model_brams])
                trainable_vars = ','.join(self._make_trainable_top_level_call_args(model))

                # Concatenate the input, output, and bram variables. Filter out empty/null values
                all_vars = ','.join(filter(None, [input_vars, output_vars, trainable_vars, bram_vars]))

                top_level = indent + f'{model.config.get_project_name()}({all_vars});\n'

                newline += top_level

            elif '// hls-fpga-machine-learning insert predictions' in line:
                newline = line
                for out in model_outputs:
                    newline += indent + f'for(int i = 0; i < {out.size_cpp()}; i++) {{\n'
                    newline += indent + '  std::cout << pr[i] << " ";\n'
                    newline += indent + '}\n'
                    newline += indent + 'std::cout << std::endl;\n'

            elif '// hls-fpga-machine-learning insert tb-output' in line:
                newline = line
                tb_stream = model.config.get_writer_config().get('TBOutputStream', 'both')
                if tb_stream != 'stdout':
                    for out in model_outputs:
                        newline += indent + 'nnet::print_result<{}, {}>({}, fout);\n'.format(
                            out.type.name, out.size_cpp(), out.name
                        )  # TODO enable this

            elif (
                '// hls-fpga-machine-learning insert output' in line
                or '// hls-fpga-machine-learning insert quantized' in line
            ):
                newline = line
                tb_stream = model.config.get_writer_config().get('TBOutputStream', 'both')
                keep_output = str(tb_stream != 'stdout').lower()  # We keep output if we need to write it to file too.
                if tb_stream != 'file':
                    for out in model_outputs:
                        newline += indent + 'nnet::print_result<{}, {}>({}, std::cout, {});\n'.format(
                            out.type.name, out.size_cpp(), out.name, keep_output
                        )

            elif '// hls-fpga-machine-learning insert namespace' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += indent + f'using namespace {namespace};\n'

            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_bridge(self, model):
        """Write the Python-C++ bridge (myproject_bridge.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vivado/myproject_bridge.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))

            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))

            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include "firmware/weights/{bram.name}.h"\n'

            elif '// hls-fpga-machine-learning insert header' in line:
                dtype = line.split('#', 1)[1].strip()
                inputs_str = ', '.join([f'{dtype} *{i.name}' for i in model_inputs])
                outputs_str = ', '.join([f'{dtype} *{o.name}' for o in model_outputs])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + '\n'

            elif '// hls-fpga-machine-learning insert wrapper' in line:
                dtype = line.split('#', 1)[1].strip()
                newline = ''
                for i in model_inputs:
                    newline += indent + '{var};\n'.format(var=i.definition_cpp(name_suffix='_ap'))
                    newline += indent + 'nnet::convert_data<{}, {}, {}>({}, {}_ap);\n'.format(
                        dtype, i.type.name, i.size_cpp(), i.name, i.name
                    )
                newline += '\n'

                for o in model_outputs:
                    newline += indent + '{var};\n'.format(var=o.definition_cpp(name_suffix='_ap'))

                newline += self._make_trainable_bridge_defaults(model, indent)
                newline += '\n'

                input_vars = ','.join([i.name + '_ap' for i in model_inputs])
                bram_vars = ','.join([b.name for b in model_brams])
                output_vars = ','.join([o.name + '_ap' for o in model_outputs])
                trainable_vars = ','.join(self._make_trainable_top_level_call_args(model))

                # Concatenate the input, output, and bram variables. Filter out empty/null values
                all_vars = ','.join(filter(None, [input_vars, output_vars, trainable_vars, bram_vars]))

                top_level = indent + f'{model.config.get_project_name()}({all_vars});\n'
                newline += top_level

                newline += '\n'

                for o in model_outputs:
                    newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                        o.type.name, dtype, o.size_cpp(), o.name, o.name
                    )

            elif '// hls-fpga-machine-learning insert trace_outputs' in line:
                newline = ''
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp', None)
                    if func and model.config.trace_output and layer.get_attr('trace', False):
                        vars = layer.get_variables()
                        for var in vars:
                            newline += (
                                indent
                                + 'nnet::trace_outputs->insert(std::pair<std::string, void *>('
                                + f'"{layer.name}", (void *) malloc({var.size_cpp()} * element_size)));\n'
                            )

            elif '// hls-fpga-machine-learning insert namespace' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += indent + f'using namespace {namespace};\n'

            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_bridge_multigraph(self, model):
        """Write the Python-C++ bridge (myproject_stitched_bridge.cpp)
        Args:
            model (MultiModelGraph): the hls4ml multigraph model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vivado/myproject_bridge.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp', 'w')
        model_inputs = model.graphs[0].get_input_variables()
        model_outputs = model.graphs[-1].get_output_variables()
        model_brams = [var for var in model.graphs[0].get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            newline = ''
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'firmware/myproject' in line:
                for graph_idx, g in enumerate(model.graphs):
                    newline += '#undef DEFINES_H_\n'
                    if len(g.outputs) == 1:
                        newline += '#define result_t ' + 'result_graph' + str(graph_idx + 1) + '_t\n'
                    newline += line.replace('myproject', format(model.graphs[graph_idx].config.get_project_name()))
                    if len(g.outputs) == 1:
                        newline += (
                            'typedef result_graph' + str(graph_idx + 1) + '_t graph' + str(graph_idx + 1) + '_result_t;\n'
                        )
                        newline += '#undef result_t\n\n' if graph_idx < len(model.graphs) - 1 else '\n'
                newline += '\n'
            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))

            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include "firmware/weights/{bram.name}.h"\n'

            elif '// hls-fpga-machine-learning insert header' in line:
                dtype = line.split('#', 1)[1].strip()
                inputs_str = ', '.join([f'{dtype} {i.name}[{i.size_cpp()}]' for i in model_inputs])
                outputs_str = ', '.join([f'{dtype} {o.name}[{o.size_cpp()}]' for o in model_outputs])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + '\n'

            elif '// hls-fpga-machine-learning insert wrapper' in line:
                dtype = line.split('#', 1)[1].strip()
                newline = ''
                for i in model_inputs:
                    newline += indent + '{var};\n'.format(var=i.definition_cpp(name_suffix='_ap'))
                    newline += indent + 'nnet::convert_data<{}, {}, {}>({}, {}_ap);\n'.format(
                        dtype, i.type.name, i.size_cpp(), i.name, i.name
                    )
                newline += '\n'

                for idx, g in enumerate(model.graphs):
                    for o in g.get_output_variables():
                        definition = o.definition_cpp(name_suffix='_ap')
                        if len(g.outputs) == 1:
                            parts = definition.split(' ', 1)
                            datatype = 'graph' + str(idx + 1) + '_result_t'
                            if parts[0].startswith('hls::stream'):
                                modified_definition = 'hls::stream<' + datatype + '> ' + parts[1]
                            else:
                                modified_definition = datatype + ' ' + parts[1]
                            newline += indent + f'{modified_definition};\n'
                        else:
                            newline += indent + f'{definition};\n'

                newline += '\n'

                top_level = ''
                output_vars = ''
                for idx, g in enumerate(model.graphs):
                    if idx == 0:
                        input_vars = ','.join([i.name + '_ap' for i in g.get_input_variables()])
                    else:
                        input_vars = output_vars
                    bram_vars = ','.join(
                        [b.name for b in [var for var in g.get_weight_variables() if var.storage.lower() == 'bram']]
                    )
                    output_vars = ','.join([o.name + '_ap' for o in g.get_output_variables()])
                    # Concatenate the input, output, and bram variables. Filter out empty/null values
                    all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))
                    top_level += indent + f'{g.config.get_project_name()}({all_vars});\n'
                newline += top_level

                newline += '\n'

                for o in model_outputs:
                    if len(model.graphs[-1].outputs) == 1:
                        newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                            datatype, dtype, o.size_cpp(), o.name, o.name
                        )
                    else:
                        newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                            o.type.name, dtype, o.size_cpp(), o.name, o.name
                        )

            elif '// hls-fpga-machine-learning insert trace_outputs' in line:
                newline = ''
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp', None)
                    if func and model.config.trace_output and layer.get_attr('trace', False):
                        vars = layer.get_variables()
                        for var in vars:
                            newline += (
                                indent
                                + 'nnet::trace_outputs->insert(std::pair<std::string, void *>('
                                + f'"{layer.name}", (void *) malloc({var.size_cpp()} * element_size)));\n'
                            )

            elif '// hls-fpga-machine-learning insert namespace' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += indent + f'using namespace {namespace};\n'

            elif '// hls-fpga-machine-learning insert tb_input_writer' in line:
                funcs = [
                    ('float', 'dump_tb_inputs_float'),
                    ('double', 'dump_tb_inputs_double'),
                ]
                newline = ''
                for dtype, funcname in funcs:
                    newline += f'void {funcname}(\n'
                    newline += '    const char* output_path'
                    for inp in model_inputs:
                        newline += f',\n    {dtype} {inp.name}[{inp.size_cpp()}]'
                    newline += '\n) {\n\n'

                    for inp in model_inputs:
                        decl = inp.definition_cpp(name_suffix='_ap').strip()
                        ap = inp.name + '_ap'
                        if decl.startswith('hls::stream'):
                            newline += f'    {decl};\n'
                        else:
                            newline += f'    {inp.type.name} {ap}[{inp.size_cpp()}];\n'
                        newline += f'    nnet::convert_data<{dtype}, {inp.type.name}, {inp.size_cpp()}>({inp.name}, {ap});\n'
                    newline += '\n'
                    newline += f'    std::ofstream fout(std::string(output_path) + "/{inp.name}_input_data.txt");\n'

                    for inp in model_inputs:
                        decl = inp.definition_cpp(name_suffix='_ap').strip()
                        shape = inp.shape

                        if decl.startswith('hls::stream'):
                            if len(shape) == 1:
                                N = shape[0]
                                newline += f'    for(int i = 0; i < {N}; i++) {{\n'
                                newline += f'        auto temp = {inp.name}_ap.read();\n'
                                newline += f'        ap_uint<{inp.type.name}::value_type::width> bits = temp[0].range();\n'
                                newline += f"        fout << bits.to_uint() << (i+1<{N} ? ' ' : '\\n');\n"
                                newline += '    }\n'
                            else:
                                inputs_list = model.nn_config['inputs']
                                fifo_depth = next((e['fifo_depth'] for e in inputs_list if e['name'] == inp.name), None)
                                batch_size = next((e['batch_size'] for e in inputs_list if e['name'] == inp.name), None)
                                newline += f'    for(int r = 0; r < {fifo_depth}; r++) {{\n'
                                newline += f'        auto temp = {inp.name}_ap.read();\n'
                                newline += f'        for(int c = 0; c < {batch_size}; c++) {{\n'
                                newline += (
                                    f'            ap_uint<{inp.type.name}::value_type::width> bits = temp[c].range();\n'
                                )
                                newline += f"            fout << bits.to_uint() << (c+1<{batch_size} ? ' ' : '\\n');\n"
                                newline += '        }\n'
                                newline += '    }\n'
                        else:
                            ap = inp.name + '_ap'
                            N = inp.size_cpp()
                            newline += f'    for(int i = 0; i < {N}; i++) {{\n'
                            newline += f'        ap_uint<{inp.type.name}::width> bits = {ap}[i].range();\n'
                            newline += f"        fout << bits.to_uint() << (i+1<{N} ? ' ' : '\\n');\n"
                            newline += '    }\n'
                    newline += '    fout.close();\n'
                    newline += '}\n'
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_build_script(self, model):
        """Write the TCL/Shell build scripts (project.tcl, build_prj.tcl, vivado_synth.tcl, build_lib.sh)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = Path(__file__).parent

        # project.tcl
        prj_tcl_dst = Path(f'{model.config.get_output_dir()}/project.tcl')
        with open(prj_tcl_dst, 'w') as f:
            f.write('variable project_name\n')
            f.write(f'set project_name "{model.config.get_project_name()}"\n')
            f.write('variable backend\n')
            f.write('set backend "vivado"\n')
            f.write('variable part\n')
            f.write('set part "{}"\n'.format(model.config.get_config_value('Part')))
            f.write('variable clock_period\n')
            f.write('set clock_period {}\n'.format(model.config.get_config_value('ClockPeriod')))
            f.write('variable clock_uncertainty\n')
            f.write('set clock_uncertainty {}\n'.format(model.config.get_config_value('ClockUncertainty', '12.5%')))
            f.write('variable version\n')
            f.write('set version "{}"\n'.format(model.config.get_config_value('Version', '1.0.0')))
            f.write('variable maximum_size\n')
            f.write('set maximum_size {}\n'.format(model.config.get_config_value('MaximumSize', '4096')))

        # build_prj.tcl
        srcpath = (filedir / '../templates/vivado/build_prj.tcl').resolve()
        dstpath = f'{model.config.get_output_dir()}/build_prj.tcl'
        copyfile(srcpath, dstpath)

        # vivado_synth.tcl
        srcpath = (filedir / '../templates/vivado/vivado_synth.tcl').resolve()
        dstpath = f'{model.config.get_output_dir()}/vivado_synth.tcl'
        copyfile(srcpath, dstpath)

        # build_lib.sh
        build_lib_src = (filedir / '../templates/vivado/build_lib.sh').resolve()
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))

                dst.write(line)
        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

    def write_build_script_multigraph(self, model):
        """Write the build script (build_lib.sh) for stitched multigraph project
        Args:
            model (MultiModelGraph): the hls4ml multigraph model.
        """
        filedir = Path(__file__).parent
        os.makedirs(model.config.get_output_dir(), exist_ok=True)
        build_lib_src = (filedir / '../templates/vivado/build_lib_multigraph.sh').resolve()
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        graph_project_names = ' '.join(f'"{g.config.get_output_dir().split("/")[-1]}"' for g in model.graphs)

        with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.config['OriginalProjectName'])
                line = line.replace('myproject_stitched', model.config.config['ProjectName'])
                line = line.replace('mystamp', model.config.config['Stamp'])
                line = line.replace('mygraph_name_list', graph_project_names)
                dst.write(line)
        os.chmod(build_lib_dst, os.stat(build_lib_dst).st_mode | stat.S_IEXEC)

    def write_nnet_utils(self, model):
        """Copy the nnet_utils, AP types headers and any custom source to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # nnet_utils
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/vivado/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        # ap_types
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/vivado/ap_types/')
        dstpath = f'{model.config.get_output_dir()}/firmware/ap_types/'

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

        # custom source
        filedir = os.path.dirname(os.path.abspath(__file__))

        custom_source = model.config.backend.get_custom_source()
        for dst, srcpath in custom_source.items():
            dstpath = f'{model.config.get_output_dir()}/firmware/{dst}'
            copyfile(srcpath, dstpath)

        if self._is_trainable_model(model):
            self.write_trainable_utils(model)

    def write_trainable_utils(self, model):
        """Copy trainable static headers into the generated firmware directory."""

        filedir = os.path.dirname(os.path.abspath(__file__))
        srcpath = os.path.join(filedir, '../templates/vivado/trainable/')
        dstpath = f'{model.config.get_output_dir()}/firmware/trainable/'

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

    def write_generated_code(self, model):
        """Write the generated code (nnet_code_gen.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        path = f'{model.config.get_output_dir()}/firmware/nnet_utils/nnet_code_gen.h'
        f = open(path)
        contents = f.readlines()
        f.close()
        f = open(path, 'w')
        namespace = model.config.get_writer_config().get('Namespace', None)

        for line in contents:
            if '// hls4ml insert code' in line:
                newline = line
                for layer in model.get_layers():
                    for generated_code in layer.code.values():
                        newline += str(generated_code)
            else:
                newline = line
            if namespace is not None:
                if 'namespace nnet' in newline:
                    newline = newline.replace('namespace nnet', f'namespace {namespace}')
            f.write(newline)
        f.close()

    def write_yml(self, model):
        """Write the config to the YAML file

        Args:
            model (ModelGraph): the hls4ml model.
        """

        def keras_model_representer(dumper, keras_model):
            model_path = model.config.get_output_dir() + '/keras_model.keras'
            keras_model.save(model_path)
            return dumper.represent_scalar('!keras_model', model_path)

        try:
            import keras

            KerasModel = keras.models.Model

            yaml.add_multi_representer(KerasModel, keras_model_representer)
        except Exception:
            pass

        with open(model.config.get_output_dir() + '/' + config_filename, 'w') as file:
            yaml.dump(model.config.config, file)

    def write_tar(self, model):
        """Write the generated project as a .tar.gz archive

        Args:
            model (ModelGraph): the hls4ml model.
        """

        write_tar = model.config.get_writer_config().get('WriteTar', False)
        if write_tar:
            tar_path = model.config.get_output_dir() + '.tar.gz'
            if os.path.exists(tar_path):
                os.remove(tar_path)
            with tarfile.open(tar_path, mode='w:gz') as archive:
                archive.add(model.config.get_output_dir(), recursive=True, arcname='')

    def write_hls(self, model, is_multigraph=False):
        if not is_multigraph:
            self.write_project_dir(model)
            self.write_project_cpp(model)
            self.write_project_header(model)
            self.write_weights(model)
            self.write_defines(model)
            self.write_parameters(model)
            self.write_test_bench(model)
            self.write_bridge(model)
            self.write_build_script(model)
            self.write_nnet_utils(model)
            self.write_generated_code(model)
            self.write_yml(model)
            self.write_tar(model)
        else:
            self.write_project_dir(model)
            self.write_build_script_multigraph(model)
            self.write_bridge_multigraph(model)
            self.write_multigraph_weights(model)
