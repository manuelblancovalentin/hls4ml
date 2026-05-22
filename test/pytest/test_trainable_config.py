import numpy as np

from hls4ml.model.graph import HLSConfig, ModelGraph


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
