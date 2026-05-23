from hls4ml.model.optimizer import ModelOptimizerPass
from hls4ml.utils.string_utils import convert_to_snake_case


SUPPORTED_LOSSES = {'half_mse'}
SUPPORTED_OPTIMIZERS = {'sgd'}
SUPPORTED_CONTROLLERS = {'none', 'ctrl_none', 'ctrl_gt_order_0', 'ctrl_gt_order_1', 'ctrl_gt_order_2', 'ctrl_gt_order_2_qa'}
SUPPORTED_TRAINABLE_LAYERS = {'Dense'}
LOSS_GRADIENT_SCALE = {
    'half_mse': 1.0,
}
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


def resolve_trainable_backward_order(model):
    """Resolve the supported sequential backward traversal order."""
    if not model.config.is_trainable():
        return False

    producer_by_output = _producer_by_output(model)
    _reject_branching_outputs(model)

    if len(model.outputs) != 1:
        raise Exception(f'Trainable hls4ml currently supports exactly one model output, got {len(model.outputs)}.')

    output_name = model.outputs[0]
    output_layer = producer_by_output.get(output_name)
    if output_layer is None:
        raise Exception(f'Unable to resolve trainable output producer for model output {output_name}.')

    forward_path = _resolve_single_path_to_input(model, producer_by_output, output_layer)
    backward_order = [layer.name for layer in reversed(forward_path) if layer.get_attr('trainable', False)]

    if len(backward_order) == 0:
        raise Exception('Trainable hls4ml requires at least one trainable layer in the backward traversal.')

    model.trainable_forward_path = tuple(layer.name for layer in forward_path)
    model.trainable_forward_order = tuple(reversed(backward_order))
    model.trainable_backward_order = tuple(backward_order)
    model.trainable_output_layer = output_layer.name

    return True


def resolve_trainable_loss_endpoints(model):
    """Attach normalized loss endpoint metadata for trainable writer/template passes."""
    if not model.config.is_trainable():
        return False

    if not hasattr(model, 'trainable_backward_order'):
        raise Exception('Trainable loss endpoint resolution requires trainable backward order to be resolved first.')

    loss_config = model.config.get_loss_config()
    loss_kind = _normalize_choice(loss_config.get('Kind'))

    if len(model.outputs) != 1:
        raise Exception(f'Trainable hls4ml currently supports exactly one model output, got {len(model.outputs)}.')

    output_name = _resolve_loss_output_name(model, loss_config)
    output_variable = model.get_layer_output_variable(output_name)
    if output_variable is None:
        raise Exception(f'Unable to resolve trainable loss output variable {output_name}.')

    output_layer_name = getattr(model, 'trainable_output_layer', None)
    output_layer = model.graph.get(output_layer_name)
    if output_layer is None:
        raise Exception(f'Unable to resolve trainable loss output layer {output_layer_name}.')

    endpoint = {
        'index': 0,
        'output_name': output_name,
        'output_layer': output_layer.name,
        'ground_truth_name': loss_config.get('GroundTruthName', f'{output_name}_truth'),
        'loss_name': loss_kind,
        'effective_loss_name': loss_kind,
        'loss_input_name': output_name,
        'loss_input_layer': output_layer.name,
        'loss_input_type': output_variable.type.name,
        'loss_input_shape': tuple(output_variable.shape),
        'loss_input_size': output_variable.size(),
        'loss_scalar_name': loss_config.get('LossScalarName', 'loss0'),
        'loss_scalar_type': 'loss0_t',
        'loss_gradient_name': loss_config.get('LossGradientName', f'{output_name}_loss_grad'),
        'loss_gradient_type': f'{output_name}_loss_grad_t',
        'loss_gradient_scale': LOSS_GRADIENT_SCALE[loss_kind],
        'skip_backward_layer': None,
    }

    model.trainable_loss_endpoints = (endpoint,)

    return True


def _resolve_loss_output_name(model, loss_config):
    configured_output = loss_config.get('Output')
    if configured_output is None:
        return model.outputs[0]

    if configured_output not in model.outputs:
        raise Exception(
            f'Trainable loss output {configured_output} is not one of the model outputs: {", ".join(model.outputs)}.'
        )

    return configured_output


def _producer_by_output(model):
    producers = {}
    for layer in model.get_layers():
        for output in layer.outputs:
            if output in producers:
                raise Exception(f'Trainable hls4ml does not support duplicate output name {output}.')
            producers[output] = layer
    return producers


def _reject_branching_outputs(model):
    consumers = {}
    for layer in model.get_layers():
        for input_name in layer.inputs:
            consumers.setdefault(input_name, []).append(layer)

    for layer in model.get_layers():
        for output in layer.outputs:
            output_consumers = consumers.get(output, [])
            if len(output_consumers) > 1:
                consumer_names = ', '.join(consumer.name for consumer in output_consumers)
                raise Exception(
                    'Trainable hls4ml currently supports only sequential graphs. '
                    f'Layer {layer.name} output {output} feeds multiple consumers: {consumer_names}.'
                )


def _resolve_single_path_to_input(model, producer_by_output, output_layer):
    reverse_path = []
    current_layer = output_layer

    while current_layer is not None:
        if len(current_layer.outputs) != 1:
            raise Exception(
                'Trainable hls4ml currently supports only single-output layers in backward traversal, '
                f'got layer {current_layer.name}.'
            )

        reverse_path.append(current_layer)

        if current_layer.name in model.inputs:
            break

        if len(current_layer.inputs) != 1:
            raise Exception(
                'Trainable hls4ml currently supports only single-input layers in backward traversal, '
                f'got layer {current_layer.name}.'
            )

        current_layer = producer_by_output.get(current_layer.inputs[0])

    if reverse_path[-1].name not in model.inputs:
        raise Exception('Trainable hls4ml could not resolve a sequential path from output to model input.')

    return list(reversed(reverse_path))


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
    backend.register_pass(
        'resolve_trainable_backward_order',
        ModelOptimizerPass('resolve_trainable_backward_order', resolve_trainable_backward_order),
    )
    backend.register_pass(
        'resolve_trainable_loss_endpoints',
        ModelOptimizerPass('resolve_trainable_loss_endpoints', resolve_trainable_loss_endpoints),
    )
