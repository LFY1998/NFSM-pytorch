import numpy as np
from skimage.metrics import structural_similarity as SSIM


def calculate_nrmse(A, P):
    """
    计算实数或复数张量的 nRMSE。

    参数:
        A (np.ndarray): 目标张量，可以是实数或复数张量。
        P (np.ndarray): 预测张量，可以是实数或复数张量。

    返回:
        float: nRMSE 值。
    """
    if np.iscomplexobj(A) or np.iscomplexobj(P):
        # 复数张量，基于模计算
        rmse = np.sqrt(np.mean(np.abs(A - P) ** 2))
        max_val = np.max(np.abs(A))
        min_val = np.min(np.abs(A))
    else:
        # 实数张量
        rmse = np.sqrt(np.mean((A - P) ** 2))
        max_val = np.max(A)
        min_val = np.min(A)

    # 归一化因子
    norm_factor = max_val - min_val
    nrmse = rmse / norm_factor if norm_factor != 0 else np.inf
    return nrmse


def calculate_psnr(tensorA, tensorB):
    """
    计算实数或复数张量的 PSNR。

    参数:
        tensorA (np.ndarray): 目标张量，可以是实数或复数张量。
        tensorB (np.ndarray): 预测张量，可以是实数或复数张量。

    返回:
        float: PSNR 值。
    """
    if np.iscomplexobj(tensorA) or np.iscomplexobj(tensorB):
        # 复数张量，基于模计算
        data_range = np.max(np.abs(tensorA)) - np.min(np.abs(tensorA))
        mse = np.mean(np.abs(tensorA - tensorB) ** 2)
    else:
        # 实数张量
        data_range = np.max(tensorA) - np.min(tensorA)
        mse = np.mean((tensorA - tensorB) ** 2)

    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def calculate_ssim(imgA, imgB):
    return SSIM(imgA, imgB, data_range=imgA.max() - imgA.min(), win_size=3)
