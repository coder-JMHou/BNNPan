import os
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import logging


def get_logger(config):
    log_path = Path(config.work_dir) / config.model_name / config.version / config.log_dir
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    file_name_time = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    file_name = f"{log_path}/{file_name_time}"

    if not config.debug:
        fh = logging.FileHandler(file_name + '.log')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_args_and_parameters(logger, args, config):
    logger.info("config_file: ")
    logger.info(args.config_file)
    logger.info("args: ")
    logger.info(args)
    logger.info("config: ")
    logger.info(config)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def yaml_read(yaml_file):
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data
