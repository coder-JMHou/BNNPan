import torch
from packaging import version
from collections import OrderedDict
from safetensors.torch import load_file
from functools import partial
import logging


def module_load(path, 
                model, 
                device, 
                ddp_rank=None, 
                strict=True, 
                spec_key=None,
                logger=None, 
                full_unmatched_log=True):
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    model = model.to(device if ddp_rank is None else ddp_rank)
    place = device if ddp_rank is None else {'cuda:%d' % 0: 'cuda:%d' % ddp_rank}
    if isinstance(place, torch.device):
        place = str(place)
    if path.endswith('pth') or path.endswith('pt'):
        if version.parse(torch.__version__) >= version.parse('2.4.0'):
            load_engine = partial(torch.load, map_location=place, weights_only=False)
        else:
            load_engine = partial(torch.load, map_location=place)
    elif path.endswith('safetensors'):
        load_engine = lambda weight_path, map_location: OrderedDict(load_file(weight_path, device=map_location))
    else:
        raise ValueError
    
    try:
        params = load_engine(path, map_location=place)
    except Exception:
        logger.info('>>> did not find the pth file, try to find in used_weights/ and ununsed_weights/...')
        try:
            path_used = path.replace('weight/', 'weight/used_weights/')
            params = load_engine(path_used, map_location=place)
        except Exception:
            path_ununsed = path.replace('weight/', 'weight/unused_weights/')
            params = load_engine(path_ununsed, map_location=place)
        
    
    # parse key
    if spec_key is not None:
        parsed_keys = spec_key.split('.')
        try:
            for k in parsed_keys:
                params = params[k]
            # _parsed_flag = True
        except KeyError:
            logger.warning(f'>>> not found parsed model `{spec_key}`, load the model directly \n \n')
            # _parsed_flag = False
        
    _load_fail_flag = False

    params_load = params
    # may be tedious but useful and safe to avoid 'module.' prefix caused error
    if not strict:
        logger.warning('model load strict is False, set it to True if you know what you are doing')
        
    def _iter_params_load_fn(model, params_load, strict):
        nonlocal _load_fail_flag
        
        if not isinstance(params_load, (list, tuple)):
            param_load_ziped = list(params_load.items()) if isinstance(params_load, dict) else params_load
            for (s_name, s_param), (name, param) in zip(param_load_ziped, model.named_parameters()):
                saved_shape = tuple(s_param.data.shape)
                required_shape = tuple(param.data.shape)
                if saved_shape != required_shape:
                    if strict:
                        logger.info(
                            f'param shape unmatched, {name} requires: {required_shape}, but got {s_name}: {saved_shape}'
                        )
                        if not full_unmatched_log:
                            logger.info('model load failed! shape of params does not match!')
                            raise RuntimeError('model load failed! shape of params does not match!')
                        else:
                            _load_fail_flag = True
                            continue
                    else:
                        logger.info(f'skip the shape mismatched param, param name {name}, '
                                    + f'current shape {required_shape} but loaded shape {saved_shape}')
                        continue
                param.data.copy_(s_param.data)
        else:
            for s_param, param in zip(params_load, model.parameters()):
                required_shape = tuple(param.data.shape)
                saved_shape = tuple(s_param.data.shape)
                
                if saved_shape != required_shape:
                    if strict:
                        logger.info(
                            f'param shape unmatched, requires: {required_shape}, but got {saved_shape}'
                        )
                        if not full_unmatched_log:
                            logger.info('model load failed! shape of params does not match!')
                            raise RuntimeError('model load failed! shape of params does not match!')
                        else:
                            _load_fail_flag = True
                            continue
                    else:
                        logger.info(f'skip the shape mismatched param, current shape {required_shape} but loaded shape {saved_shape}')
                        continue
                param.data.copy_(s_param.data)
        
    def _load_fn(model, params_load, strict):

        if isinstance(params_load, OrderedDict):  # ordered dict
            model.load_state_dict(params_load, strict=strict)
        else:
            _iter_params_load_fn(model, params_load, strict)


    _load_fn(model, params_load, strict)
    
    if _load_fail_flag:
        raise RuntimeError('model load failed! shape of params does not match!')
        
    logger.info('load pretrain weights')
    return model