from hls4ml.model.optimizer import ModelOptimizerPass
from hls4ml.utils.string_utils import convert_to_snake_case


SUPPORTED_LOSSES = {'half_mse'}
SUPPORTED_OPTIMIZERS = {'sgd'}
SUPPORTED_CONTROLLERS = {'none', 'ctrl_gt_order_0', 'ctrl_gt_order_1', 'ctrl_gt_order_2', 'ctrl_gt_order_2_qa'}
SUPPORTED_TRAINABLE_LAYERS = {'Dense'}
REQUIRED_TRAINABLE_PRECISION = (
    'grad_in',
    'grad_out',
    'weight_grad',
    'bias_grad',
    'gradient_accum',
    'raw_update',
    'update',
    'optimizer_state',
    'controller_metric',
    'alpha',
)


def _normalize_choice(value):
    if value is None:
        return None
    return convert_to_snake_case(str(value).lower().replace('-', '_'))


def validate_trainable_config(model):
    """Validate the currently supported trainable hls4ml graph subset."""
    if not model.config.is_trainable():
        return False

    training_config = model.config.get_training_config()
    _validate_batch_size(training_config)
    _validate_loss(model.config.get_loss_config())
    _validate_optimizer(model.config.get_optimizer_config())
    _validate_controller(model.config.get_controller_config())
    _validate_outputs(model)
    _validate_layers(model)

    return False


def _validate_batch_size(training_config):
    batch_size = training_config.get('BatchSize')
    if not isinstance(batch_size, int) or batch_size < 1:
        raise Exception(f'Trainable hls4ml requires BatchSize to be a positive integer, got {batch_size}.')


def _validate_loss(loss_config):
    loss_kind = _normalize_choice(loss_config.get('Kind'))
    if loss_kind not in SUPPORTED_LOSSES:
        raise Exception(
            f'Trainable hls4ml currently supports losses {sorted(SUPPORTED_LOSSES)}, got {loss_config.get("Kind")}.'
        )


def _validate_optimizer(optimizer_config):
    optimizer_kind = _normalize_choice(optimizer_config.get('Kind'))
    if optimizer_kind not in SUPPORTED_OPTIMIZERS:
        raise Exception(
            'Trainable hls4ml currently supports optimizers '
            f'{sorted(SUPPORTED_OPTIMIZERS)}, got {optimizer_config.get("Kind")}.'
        )

    learning_rate = optimizer_config.get('LearningRate')
    learning_rate_input = optimizer_config.get('LearningRateInput')
    if learning_rate is None and learning_rate_input is None:
        raise Exception('Trainable hls4ml requires Optimizer.LearningRate or Optimizer.LearningRateInput.')


def _validate_controller(controller_config):
    controller_kind = _normalize_choice(controller_config.get('Kind'))
    if controller_kind not in SUPPORTED_CONTROLLERS:
        raise Exception(
            'Trainable hls4ml currently supports controllers '
            f'{sorted(SUPPORTED_CONTROLLERS)}, got {controller_config.get("Kind")}.'
        )


def _validate_outputs(model):
    if len(model.outputs) != 1:
        raise Exception(f'Trainable hls4ml currently supports exactly one model output, got {len(model.outputs)}.')


def _validate_layers(model):
    trainable_layers = []
    for layer in model.get_layers():
        if not layer.get_attr('trainable', False):
            continue

        trainable_layers.append(layer)
        if layer.class_name not in SUPPORTED_TRAINABLE_LAYERS:
            raise Exception(
                'Trainable hls4ml currently supports trainable layer types '
                f'{sorted(SUPPORTED_TRAINABLE_LAYERS)}, got {layer.class_name} layer {layer.name}.'
            )

        _validate_trainable_precision(layer)

    if len(trainable_layers) == 0:
        raise Exception('Trainable hls4ml requires at least one trainable layer.')


def _validate_trainable_precision(layer):
    missing = [field for field in REQUIRED_TRAINABLE_PRECISION if layer.get_attr(field + '_t') is None]
    if missing:
        raise Exception(f'Trainable layer {layer.name} is missing precision attributes: {", ".join(missing)}.')


def register_trainable(backend):
    backend.register_pass('validate_trainable_config', ModelOptimizerPass('validate_trainable_config', validate_trainable_config))
