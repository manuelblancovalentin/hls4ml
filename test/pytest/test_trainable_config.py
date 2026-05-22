import numpy as np
import pytest

from hls4ml.backends import get_backend
from hls4ml.model.flow import get_flow
from hls4ml.model.graph import HLSConfig, ModelGraph
from hls4ml.model.optimizer import get_optimizer


TRAINABLE_PRECISION = {
    'grad_in': 'ap_fixed<18,8>',
    'grad_out': 'ap_fixed<18,8>',
    'weight_grad': 'ap_fixed<20,8>',
    'bias_grad': 'ap_fixed<20,8>',
    'gradient_accum': 'ap_fixed<28,14>',
    'raw_update': 'ap_fixed<20,6>',
    'update': 'ap_fixed<20,6>',
    'optimizer_state': 'ap_fixed<20,6>',
    'controller_metric': 'ap_fixed<32,16>',
    'alpha': 'ap_fixed<16,4>',
}


def make_config(training=None, layer_training=None):
    model_config = {'Precision': 'ap_fixed<16,6>', 'ReuseFactor': 1}
    if training is not None:
        model_config['Training'] = training

    hls_config = {'Model': model_config}
    if layer_training is not None:
        hls_config['LayerName'] = {'dense': {'Training': layer_training}}

    return {
        'Backend': 'Vivado',
        'ProjectName': 'myproject',
        'OutputDir': 'myproject_prj',
        'HLSConfig': hls_config,
    }


class DummyLayer:
    name = 'dense'
    class_name = 'Dense'


def test_training_config_defaults_to_inference():
    config = HLSConfig(make_config())

    assert config.is_trainable() is False
    assert config.get_training_config()['BatchSize'] == 1
    assert config.get_loss_config() == {'Kind': None}
    assert config.get_optimizer_config()['Kind'] == 'sgd'
    assert config.get_controller_config()['Kind'] == 'none'


def test_training_config_merges_model_schema_sections():
    config = HLSConfig(
        make_config(
            {
                'Trainable': True,
                'BatchSize': 8,
                'Loss': {'Kind': 'half_mse', 'Output': 'output'},
                'Optimizer': {'LearningRate': 0.01},
                'Controller': {'Kind': 'ctrl_gt_order_0'},
                'Precision': {'grad_out': 'ap_fixed<18,8>', 'alpha': 'ap_fixed<16,4>'},
            }
        )
    )

    assert config.is_trainable() is True
    assert config.get_training_config()['BatchSize'] == 8
    assert config.get_loss_config() == {'Kind': 'half_mse', 'Output': 'output'}
    assert config.get_optimizer_config() == {'Kind': 'sgd', 'LearningRate': 0.01, 'LearningRateInput': None}
    assert config.get_controller_config()['Kind'] == 'ctrl_gt_order_0'
    assert config.get_controller_config()['SafetyBudget'] == {'Enabled': False}
    assert config.get_trainable_precision_config()['grad_out'] == 'ap_fixed<18,8>'
    assert config.get_trainable_precision_config()['alpha'] == 'ap_fixed<16,4>'


def test_training_accessor_returns_copies():
    config = HLSConfig(make_config({'Trainable': True, 'Loss': {'Kind': 'half_mse'}}))

    loss_config = config.get_loss_config()
    loss_config['Kind'] = 'mse'

    assert config.get_loss_config()['Kind'] == 'half_mse'


def test_boolean_training_config_is_supported():
    config = HLSConfig(make_config(True))

    assert config.is_trainable() is True
    assert config.get_training_config()['Optimizer']['Kind'] == 'sgd'


def test_layer_trainable_config_defaults_to_model_trainable_state():
    config = HLSConfig(make_config({'Trainable': True}))

    assert config.get_layer_trainable_config(DummyLayer()) == {'Trainable': True}


def test_layer_trainable_config_reads_layer_override():
    config = HLSConfig(make_config({'Trainable': True}, {'Trainable': False, 'Precision': {'grad_in': 'ap_fixed<12,4>'}}))

    layer_config = config.get_layer_trainable_config(DummyLayer())

    assert layer_config['Trainable'] is False
    assert layer_config['Precision'] == {'grad_in': 'ap_fixed<12,4>'}


def test_trainable_precision_fields_are_immutable_to_callers():
    config = HLSConfig(make_config())

    fields = config.get_trainable_precision_fields()

    assert isinstance(fields, tuple)
    assert 'grad_in' in fields
    assert 'alpha' in fields


def test_trainable_precision_config_merges_layer_overrides():
    config = HLSConfig(
        make_config(
            {'Trainable': True, 'Precision': {'grad_in': 'ap_fixed<18,8>', 'grad_out': 'ap_fixed<19,9>'}},
            {'Precision': {'grad_in': 'ap_fixed<12,4>'}},
        )
    )

    precision_config = config.get_layer_trainable_precision_config(DummyLayer())

    assert precision_config['grad_in'] == 'ap_fixed<12,4>'
    assert precision_config['grad_out'] == 'ap_fixed<19,9>'


def test_trainable_precision_accessor_converts_to_named_type_parts():
    config = HLSConfig(make_config({'Trainable': True, 'Precision': {'grad_in': 'ap_fixed<18,8>'}}))

    precision, type_name = config.get_trainable_precision(DummyLayer(), 'grad_in')

    assert precision.width == 18
    assert precision.integer == 8
    assert type_name == 'dense_grad_in_t'


def test_trainable_layer_attributes_are_created_from_precision_config():
    layers = [
        {'class_name': 'Input', 'name': 'input_layer', 'input_shape': [1]},
        {
            'class_name': 'Dense',
            'name': 'dense',
            'n_in': 1,
            'n_out': 1,
            'weight_data': np.array([[1.0]]),
            'bias_data': np.array([0.0]),
        },
    ]
    config = make_config(
        {
            'Trainable': True,
            'Precision': {
                'grad_in': 'ap_fixed<18,8>',
                'grad_out': 'ap_fixed<18,8>',
                'raw_update': 'ap_fixed<20,10>',
                'alpha': 'ap_fixed<16,4>',
            },
        },
        {'Precision': {'grad_in': 'ap_fixed<12,4>'}},
    )
    config['HLSConfig']['Flows'] = []

    model = ModelGraph.from_layer_list(config, layers)
    dense = model.graph['dense']

    assert dense.get_attr('trainable') is True
    assert dense.get_attr('grad_in_t').name == 'dense_grad_in_t'
    assert dense.get_attr('grad_in_t').precision.width == 12
    assert dense.get_attr('grad_in_t').precision.integer == 4
    assert dense.get_attr('grad_out_t').precision.width == 18
    assert dense.get_attr('raw_update_t').precision.width == 20
    assert dense.get_attr('alpha_t').precision.width == 16
    assert 'grad_in_t' in dense.types


def test_trainable_layer_attributes_are_skipped_when_layer_is_not_trainable():
    layers = [
        {'class_name': 'Input', 'name': 'input_layer', 'input_shape': [1]},
        {
            'class_name': 'Dense',
            'name': 'dense',
            'n_in': 1,
            'n_out': 1,
            'weight_data': np.array([[1.0]]),
            'bias_data': np.array([0.0]),
        },
    ]
    config = make_config({'Trainable': True, 'Precision': {'grad_in': 'ap_fixed<18,8>'}}, {'Trainable': False})
    config['HLSConfig']['Flows'] = []

    model = ModelGraph.from_layer_list(config, layers)
    dense = model.graph['dense']

    assert dense.get_attr('trainable') is False
    assert dense.get_attr('grad_in_t') is None


def test_vivado_trainable_flow_is_registered_before_writer():
    backend = get_backend('Vivado')

    trainable_flow = get_flow('vivado:trainable')
    ip_flow = get_flow(backend.get_default_flow())

    assert trainable_flow.requires == ['vivado:apply_templates']
    assert trainable_flow.optimizers == [
        'vivado:validate_trainable_config',
        'vivado:resolve_trainable_backward_order',
        'vivado:resolve_trainable_loss_endpoints',
    ]
    assert 'vivado:trainable' in ip_flow.requires


def test_vivado_trainable_validation_accepts_supported_dense_graph():
    layers = [
        {'class_name': 'Input', 'name': 'input_layer', 'input_shape': [1]},
        {
            'class_name': 'Dense',
            'name': 'dense',
            'n_in': 1,
            'n_out': 1,
            'weight_data': np.array([[1.0]]),
            'bias_data': np.array([0.0]),
        },
    ]
    config = make_config(
        {
            'Trainable': True,
            'Loss': {'Kind': 'half_mse'},
            'Optimizer': {'Kind': 'sgd', 'LearningRate': 0.01},
            'Controller': {'Kind': 'CTRL-GT-ORDER-0'},
            'Precision': TRAINABLE_PRECISION,
        }
    )
    config['HLSConfig']['LayerName'] = {'input_layer': {'Training': {'Trainable': False}}}
    config['HLSConfig']['Flows'] = []
    model = ModelGraph.from_layer_list(config, layers)

    validate_trainable_config = get_optimizer('vivado:validate_trainable_config')

    assert validate_trainable_config.transform(model) is False


def test_vivado_trainable_validation_rejects_missing_loss():
    layers = [
        {'class_name': 'Input', 'name': 'input_layer', 'input_shape': [1]},
        {
            'class_name': 'Dense',
            'name': 'dense',
            'n_in': 1,
            'n_out': 1,
            'weight_data': np.array([[1.0]]),
            'bias_data': np.array([0.0]),
        },
    ]
    config = make_config(
        {
            'Trainable': True,
            'Optimizer': {'LearningRate': 0.01},
            'Precision': TRAINABLE_PRECISION,
        }
    )
    config['HLSConfig']['LayerName'] = {'input_layer': {'Training': {'Trainable': False}}}
    config['HLSConfig']['Flows'] = []
    model = ModelGraph.from_layer_list(config, layers)

    validate_trainable_config = get_optimizer('vivado:validate_trainable_config')

    with pytest.raises(Exception, match='supports losses'):
        validate_trainable_config.transform(model)


def test_vivado_trainable_backward_order_resolves_sequential_dense_graph():
    layers = [
        {'class_name': 'Input', 'name': 'input_layer', 'input_shape': [2]},
        {
            'class_name': 'Dense',
            'name': 'dense0',
            'n_in': 2,
            'n_out': 2,
            'weight_data': np.ones((2, 2)),
            'bias_data': np.zeros(2),
        },
        {
            'class_name': 'Dense',
            'name': 'dense1',
            'inputs': ['dense0'],
            'n_in': 2,
            'n_out': 1,
            'weight_data': np.ones((2, 1)),
            'bias_data': np.zeros(1),
        },
    ]
    config = make_config(
        {
            'Trainable': True,
            'Loss': {'Kind': 'half_mse'},
            'Optimizer': {'Kind': 'sgd', 'LearningRate': 0.01},
            'Controller': {'Kind': 'ctrl_gt_order_0'},
            'Precision': TRAINABLE_PRECISION,
        }
    )
    config['HLSConfig']['LayerName'] = {'input_layer': {'Training': {'Trainable': False}}}
    config['HLSConfig']['Flows'] = []
    model = ModelGraph.from_layer_list(config, layers)

    validate_trainable_config = get_optimizer('vivado:validate_trainable_config')
    resolve_backward_order = get_optimizer('vivado:resolve_trainable_backward_order')

    validate_trainable_config.transform(model)

    assert resolve_backward_order.transform(model) is True
    assert model.trainable_forward_path == ('input_layer', 'dense0', 'dense1')
    assert model.trainable_forward_order == ('dense0', 'dense1')
    assert model.trainable_backward_order == ('dense1', 'dense0')
    assert model.trainable_output_layer == 'dense1'


def test_vivado_trainable_backward_order_rejects_branching_graph():
    layers = [
        {'class_name': 'Input', 'name': 'input_layer', 'input_shape': [2]},
        {
            'class_name': 'Dense',
            'name': 'dense0',
            'n_in': 2,
            'n_out': 2,
            'weight_data': np.ones((2, 2)),
            'bias_data': np.zeros(2),
        },
        {
            'class_name': 'Dense',
            'name': 'dense1',
            'inputs': ['dense0'],
            'n_in': 2,
            'n_out': 1,
            'weight_data': np.ones((2, 1)),
            'bias_data': np.zeros(1),
        },
        {
            'class_name': 'Dense',
            'name': 'dense2',
            'inputs': ['dense0'],
            'n_in': 2,
            'n_out': 1,
            'weight_data': np.ones((2, 1)),
            'bias_data': np.zeros(1),
        },
    ]
    config = make_config(
        {
            'Trainable': True,
            'Loss': {'Kind': 'half_mse'},
            'Optimizer': {'Kind': 'sgd', 'LearningRate': 0.01},
            'Controller': {'Kind': 'ctrl_gt_order_0'},
            'Precision': TRAINABLE_PRECISION,
        }
    )
    config['HLSConfig']['LayerName'] = {'input_layer': {'Training': {'Trainable': False}}}
    config['HLSConfig']['Flows'] = []
    model = ModelGraph.from_layer_list(config, layers, outputs=['dense1'])

    validate_trainable_config = get_optimizer('vivado:validate_trainable_config')
    resolve_backward_order = get_optimizer('vivado:resolve_trainable_backward_order')

    validate_trainable_config.transform(model)

    with pytest.raises(Exception, match='supports only sequential graphs'):
        resolve_backward_order.transform(model)


def test_vivado_trainable_loss_endpoint_resolves_half_mse_metadata():
    layers = [
        {'class_name': 'Input', 'name': 'input_layer', 'input_shape': [2]},
        {
            'class_name': 'Dense',
            'name': 'dense',
            'n_in': 2,
            'n_out': 1,
            'weight_data': np.ones((2, 1)),
            'bias_data': np.zeros(1),
        },
    ]
    config = make_config(
        {
            'Trainable': True,
            'Loss': {
                'Kind': 'half_mse',
                'GroundTruthName': 'y_true',
                'LossScalarName': 'train_loss',
                'LossGradientName': 'dL_dy',
            },
            'Optimizer': {'Kind': 'sgd', 'LearningRate': 0.01},
            'Controller': {'Kind': 'ctrl_gt_order_0'},
            'Precision': TRAINABLE_PRECISION,
        }
    )
    config['HLSConfig']['LayerName'] = {'input_layer': {'Training': {'Trainable': False}}}
    config['HLSConfig']['Flows'] = []
    model = ModelGraph.from_layer_list(config, layers)

    get_optimizer('vivado:validate_trainable_config').transform(model)
    get_optimizer('vivado:resolve_trainable_backward_order').transform(model)

    assert get_optimizer('vivado:resolve_trainable_loss_endpoints').transform(model) is True

    assert len(model.trainable_loss_endpoints) == 1
    endpoint = model.trainable_loss_endpoints[0]
    assert endpoint['output_name'] == 'dense'
    assert endpoint['output_layer'] == 'dense'
    assert endpoint['ground_truth_name'] == 'y_true'
    assert endpoint['loss_name'] == 'half_mse'
    assert endpoint['effective_loss_name'] == 'half_mse'
    assert endpoint['loss_input_name'] == 'dense'
    assert endpoint['loss_input_layer'] == 'dense'
    assert endpoint['loss_input_shape'] == (1,)
    assert endpoint['loss_input_size'] == 1
    assert endpoint['loss_scalar_name'] == 'train_loss'
    assert endpoint['loss_scalar_type'] == 'loss0_t'
    assert endpoint['loss_gradient_name'] == 'dL_dy'
    assert endpoint['loss_gradient_type'] == 'dense_loss_grad_t'
    assert endpoint['loss_gradient_scale'] == 1.0
    assert endpoint['skip_backward_layer'] is None


def test_vivado_trainable_loss_endpoint_rejects_unknown_output_mapping():
    layers = [
        {'class_name': 'Input', 'name': 'input_layer', 'input_shape': [2]},
        {
            'class_name': 'Dense',
            'name': 'dense',
            'n_in': 2,
            'n_out': 1,
            'weight_data': np.ones((2, 1)),
            'bias_data': np.zeros(1),
        },
    ]
    config = make_config(
        {
            'Trainable': True,
            'Loss': {'Kind': 'half_mse', 'Output': 'not_an_output'},
            'Optimizer': {'Kind': 'sgd', 'LearningRate': 0.01},
            'Controller': {'Kind': 'ctrl_gt_order_0'},
            'Precision': TRAINABLE_PRECISION,
        }
    )
    config['HLSConfig']['LayerName'] = {'input_layer': {'Training': {'Trainable': False}}}
    config['HLSConfig']['Flows'] = []
    model = ModelGraph.from_layer_list(config, layers)

    get_optimizer('vivado:validate_trainable_config').transform(model)
    get_optimizer('vivado:resolve_trainable_backward_order').transform(model)

    with pytest.raises(Exception, match='not one of the model outputs'):
        get_optimizer('vivado:resolve_trainable_loss_endpoints').transform(model)
