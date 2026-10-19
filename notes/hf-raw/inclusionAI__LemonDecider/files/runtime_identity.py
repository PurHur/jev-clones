"""Explicit corrected-runtime identity; never mutates source weights/config files."""
import copy

RUNTIME_VERSION = 'lemonjev-qwen-decision-runtime-v2'


def runtime_config(config, model_id, *, step=None, baseline=False):
    if not isinstance(model_id, str) or not model_id.startswith(('lemonjev-', 'lemondecider-')) or model_id == getattr(config, 'model_id', None):
        raise ValueError('A distinct explicit lemonjev- runtime model identity is required')
    result = copy.deepcopy(config)
    result.source_model_id = getattr(config, 'model_id', None)
    result.model_id = model_id
    result.project_name = 'LemonJev'
    result.runtime_version = RUNTIME_VERSION
    result.runtime_override = bool(baseline)
    result.initialization_step = int(getattr(config, 'initialization_step', 0) if step is None else step)
    result.trained = result.initialization_step > 0
    result.temperature = 1.0
    result.calibrated = False
    result.calibration = {'method': 'none', 'temperature': 1.0}
    result.vision_training_mode = 'frozen'  # export/serving does not execute dummy training work
    return result


def decision_metadata(config):
    return {'schema': 'lemonjev-decision-config-v2', 'model_id': config.model_id,
            'source_model_id': config.source_model_id, 'runtime_version': config.runtime_version,
            'runtime_override': config.runtime_override, 'step': config.initialization_step,
            'trained': config.trained, 'calibrated': config.calibrated,
            'calibration': config.calibration, 'temperature': config.temperature,
            'attention_mode': config.attention_mode, 'pooling': config.pooling,
            'max_input_length': config.max_input_length, 'max_options': config.max_options,
            'code_scheme': config.code_scheme, 'codes': config.codes, 'token_ids': config.token_ids,
            'decision_protocol': getattr(config, 'decision_protocol', 'autojev-v1'),
            'readout_dtype': 'float32' if getattr(config, 'decision_protocol', 'autojev-v1') == 'lemondecider-v1' else 'legacy'}
