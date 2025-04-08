from Model.default_config import default_cfg
from Model.NeuralFieldSystemMatrix import SMNeuralFieldModel
import torch
import numpy as np
import random

import os

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# 或
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def set_seed(seed):
    # 1. 固定 Python 内置的 random 库
    random.seed(seed)

    # 2. 固定 numpy 的随机性
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 3. 固定 PyTorch 的 CPU 随机性
    torch.manual_seed(seed)

    # 4. 如果使用 GPU, 固定 PyTorch 的 GPU 随机性
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU

    # 5. 设置 PyTorch 的一些额外配置，确保计算结果一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. 强制使用确定性算法
    # torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    opt = default_cfg()
    set_seed(opt.random_seed)
    model = SMNeuralFieldModel(opt)
    model.fit()
