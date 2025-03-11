from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim, calculate_rmse, calculate_lpips,calculate_abs_rel,calculate_d1,calculate_d2,calculate_d3,caculate_rmse_log

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_lpips','calculate_rmse','calculate_abs_rel','calculate_d1','calculate_d2','calculate_d3','caculate_rmse_log']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
