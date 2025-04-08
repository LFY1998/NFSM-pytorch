import h5py
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def generate_mask(GridSize, sparsity=0.1, sampling_method="random", downsample_factor=None):
    """
    根据给定的稀疏度或下采样倍数生成部分观测张量和观测掩码。

    参数：
    - tensor: 原始目标张量。
    - sparsity: 随机观测时的稀疏度（0到1之间，表示观测的百分比）。
    - sampling_method: 观测方式选择，"random"表示随机观测，"downsample"表示均匀下采样。
    - downsample_factor: 均匀下采样的倍数（仅当sampling_method="downsample"时有效），如2、4、8。

    返回：
    - observed_tensor: 部分观测的张量。
    - mask: 观测掩码（布尔数组），1表示已观测，0表示缺失。
    """
    np.random.seed(0)
    if sampling_method == "random":
        # 随机生成观测掩码
        mask = np.random.rand(*GridSize) < sparsity
    elif sampling_method == "downsample":
        # 生成下采样掩码：每隔downsample_factor取一个值
        mask = np.zeros(GridSize, dtype=bool)
        # 遍历张量每个维度，进行下采样
        mask[::downsample_factor[0], ::downsample_factor[1], ::downsample_factor[2]] = True
    elif sampling_method == "grouped_random":
        n = 1 / sparsity
        assert n % 1 == 0
        n = int(n)
        total_slices = GridSize[0]
        mask = np.zeros(GridSize, dtype=bool)

        # 每组包含 n 个切片，分组采样
        for group_start in range(0, total_slices, n):
            group_end = min(group_start + n, total_slices)
            group_masks = np.zeros((n, GridSize[1], GridSize[2]), dtype=bool)

            for slice_idx in range(group_end - group_start):
                # 每个切片随机采样 1/2^n 的点，并确保组内切片不重叠
                num_points = int(np.prod(GridSize[1:]) * sparsity)
                available_indices = np.argwhere(~group_masks.any(axis=0))  # 获取当前未被占用的点
                if len(available_indices) < num_points:
                    raise ValueError("Not enough points available to sample without overlap within a group.")
                selected_indices = available_indices[
                    np.random.choice(len(available_indices), num_points, replace=False)]
                group_masks[slice_idx, selected_indices[:, 0], selected_indices[:, 1]] = True

            # 将组内的掩码合并到总掩码中
            mask[group_start:group_end, :, :] = group_masks[:group_end - group_start, :, :]

    else:
        raise ValueError(f"Unsupported sampling_method '{sampling_method}'. Use 'random' or 'downsample'.")
    mask_indices = torch.nonzero(torch.from_numpy(mask), as_tuple=False)

    return mask, mask_indices


def generate_normalized_coordinates(Nx, Ny, Nz, uprate=1, normalized=True):
    # 创建索引坐标
    x = torch.linspace(0, 1, int(Nx * uprate))  # 归一化 x 方向
    y = torch.linspace(0, 1, int(Ny * uprate))  # 归一化 y 方向
    z = torch.linspace(0, 1, int(Nz * uprate))  # 归一化 z 方向
    if not normalized:
        x = torch.arange(0, x.size(0), dtype=int)
        y = torch.arange(0, y.size(0), dtype=int)
        z = torch.arange(0, z.size(0), dtype=int)

    # 使用 meshgrid 创建三维坐标网格
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')  # 没有 `indexing` 参数
    # grid_x, grid_y, grid_z = grid_x.permute(1, 0, 2), grid_y.permute(1, 0, 2), grid_z.permute(1, 0, 2)

    # 将坐标合并为形状 [Nx, Ny, Nz, 3]
    P = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # [Nx, Ny, Nz, 3]
    # P = torch.complex(P, torch.zeros_like(P))
    return P


def normalize(tensor, normalize_type='zscore'):
    """
    对单个复数张量进行归一化，并返回归一化参数。  归一化参数仅在观测部分计算
    可选的归一化类型：'zscore'（标准化归一化），'minmax'（0-1归一化）和 'robust'（Robust Scaler）。

    参数：
    - tensor: 输入的复数张量。
    - normalize_type: 归一化的类型，'zscore'、'minmax' 或 'robust'。默认为 'zscore'。

    返回：
    - normalized_tensor: 归一化后的张量。
    - normalization_params: 归一化的参数，包括实部和虚部的均值、标准差、最小值、最大值、四分位数等。
    """
    # 计算实部和虚部的均值、标准差
    mu_real = np.mean(tensor.real, axis=1)[:, None, None, None]
    std_real = np.std(tensor.real, axis=1)[:, None, None, None]
    mu_imag = np.mean(tensor.imag, axis=1)[:, None, None, None]
    std_imag = np.std(tensor.imag, axis=1)[:, None, None, None]

    # 0-1归一化的最小值和最大值
    min_real = np.min(tensor.real, axis=1)[:, None, None, None]
    max_real = np.max(tensor.real, axis=1)[:, None, None, None]
    min_imag = np.min(tensor.imag, axis=1)[:, None, None, None]
    max_imag = np.max(tensor.imag, axis=1)[:, None, None, None]

    # 计算四分位数 (IQR)
    q1_real = np.percentile(tensor.real, 25, axis=1)[:, None, None, None]
    q3_real = np.percentile(tensor.real, 75, axis=1)[:, None, None, None]
    q1_imag = np.percentile(tensor.imag, 25, axis=1)[:, None, None, None]
    q3_imag = np.percentile(tensor.imag, 75, axis=1)[:, None, None, None]

    iqr_real = q3_real - q1_real
    iqr_imag = q3_imag - q1_imag

    if normalize_type == 'zscore':
        # 进行 Z-Score 归一化（标准化归一化）
        normalized_tensor = np.copy(tensor)
        normalized_tensor.real = (tensor.real - mu_real[:, :, 0, 0]) / std_real[:, :, 0, 0]
        normalized_tensor.imag = (tensor.imag - mu_imag[:, :, 0, 0]) / std_imag[:, :, 0, 0]
    elif normalize_type == 'minmax':
        # 进行 0-1 归一化
        normalized_tensor = np.copy(tensor)
        normalized_tensor.real = (tensor.real - min_real[:, :, 0, 0]) / (
                max_real[:, :, 0, 0] - min_real[:, :, 0, 0]) - 0.5
        normalized_tensor.imag = (tensor.imag - min_imag[:, :, 0, 0]) / (
                max_imag[:, :, 0, 0] - min_imag[:, :, 0, 0]) - 0.5
    elif normalize_type == 'robust':
        # 进行 Robust Scaler 归一化
        normalized_tensor = np.copy(tensor)
        normalized_tensor.real = (tensor.real - q1_real[:, :, 0, 0]) / iqr_real[:, :, 0, 0]
        normalized_tensor.imag = (tensor.imag - q1_imag[:, :, 0, 0]) / iqr_imag[:, :, 0, 0]
    else:
        raise ValueError("normalize_type should be 'zscore', 'minmax', or 'robust'")

    # 返回归一化后的张量及其参数
    normalization_params = {
        'mu_real': mu_real,
        'std_real': std_real,
        'mu_imag': mu_imag,
        'std_imag': std_imag,
        'min_real': min_real,
        'max_real': max_real,
        'min_imag': min_imag,
        'max_imag': max_imag,
        'q1_real': q1_real,
        'q3_real': q3_real,
        'iqr_real': iqr_real,
        'q1_imag': q1_imag,
        'q3_imag': q3_imag,
        'iqr_imag': iqr_imag
    }

    return normalized_tensor, normalization_params


def denormalize(tensor, normalization_params, normalize_type='zscore'):
    """
    对单个归一化张量进行解归一化，恢复原始值。

    参数：
    - tensor: 归一化后的复数张量。
    - normalization_params: 归一化时保存的参数，包括实部和虚部的均值、标准差、最小值、最大值、四分位数等。
    - normalize_type: 选择归一化的类型 ('zscore'、'minmax' 或 'robust')。

    返回：
    - denormalized_tensor: 解归一化后的张量。
    """
    # 获取归一化参数
    mu_real = normalization_params['mu_real']
    std_real = normalization_params['std_real']
    mu_imag = normalization_params['mu_imag']
    std_imag = normalization_params['std_imag']
    min_real = normalization_params['min_real']
    max_real = normalization_params['max_real']
    min_imag = normalization_params['min_imag']
    max_imag = normalization_params['max_imag']
    q1_real = normalization_params['q1_real']
    q3_real = normalization_params['q3_real']
    iqr_real = normalization_params['iqr_real']
    q1_imag = normalization_params['q1_imag']
    q3_imag = normalization_params['q3_imag']
    iqr_imag = normalization_params['iqr_imag']

    denormalized_tensor = np.copy(tensor)  # 创建副本避免修改输入张量

    if normalize_type == 'zscore':
        # 解 Z-Score 归一化
        denormalized_tensor.real = (tensor.real * std_real) + mu_real
        denormalized_tensor.imag = (tensor.imag * std_imag) + mu_imag
    elif normalize_type == 'minmax':
        # 解 0-1 归一化
        denormalized_tensor.real = ((tensor.real + 0.5) * (max_real - min_real)) + min_real
        denormalized_tensor.imag = ((tensor.imag + 0.5) * (max_imag - min_imag)) + min_imag
    elif normalize_type == 'robust':
        # 解 Robust Scaler 归一化
        denormalized_tensor.real = (tensor.real * iqr_real) + q1_real
        denormalized_tensor.imag = (tensor.imag * iqr_imag) + q1_imag
    else:
        raise ValueError("normalize_type should be 'zscore', 'minmax', or 'robust'")

    return denormalized_tensor


class CustomArray:
    def __init__(self, dataset: h5py.Dataset, isBG):
        """
        dataset: h5py.Dataset, 存储在 HDF5 文件中的数据
        isBG: 1D 布尔数组，表示哪些点是背景点
        """
        self.dataset = dataset  # h5py.Dataset (未加载到内存)
        self.isBG = np.asarray(isBG, dtype=bool)  # 确保 isBG 是布尔数组
        self.valid_indices = np.where(~self.isBG)[0]  # 预计算非背景点的索引

    def __getitem__(self, key):
        """
        访问 s[c, k] 时，支持 k 为整数或数组，等价于 S[:, c, k, ~isBG].squeeze()
        """
        if isinstance(key, tuple) and len(key) == 2:
            c, k = key
            k = np.atleast_1d(k)  # 确保 k 是 1D 数组

            # 从 h5py 读取完整列，但只选取指定的 k 频率
            full_data = self.dataset[:, c, k, :]  # (N, len(k), M)

            # 过滤掉背景点
            filtered_data = full_data[:, :, ~self.isBG]  # 只保留非背景点
            return np.squeeze(filtered_data)  # squeeze 以去除单一维度
        else:
            raise IndexError("Only indexing as s[c, k] is supported.")


class SystemMatrixLoad(Dataset):
    def __init__(self, SMPath, MeasPath, SNRThreshold, sparsity, sampling_method, K_select, use_coil, upsamplerate,
                 Phantom_type, model_type=None, data_type="OpenMPI", downsample_factor=None):
        super().__init__()
        self.phantomType = None
        self.data_type = data_type
        self.SNR_filter = None
        self.normalized = True
        if model_type == "SimplyNFM":
            self.normalized = False
        self.GridNum = None
        self.K_train_Num = None
        self.DenseViewGridSize = None
        self.datatype = "train"
        self.Su_observed_norm = None
        self.Su_observed = None
        self.Su = None
        self.Su_norm = None
        self.normalizeParam = None
        self.Coord = None
        self.K_filter = None
        self.SNR = None
        self.GridSize = None
        self.S = None
        self.K_select = None
        self.use_coil = use_coil
        self.SNR_train = None
        self.K_train = None
        self.filenameSM = SMPath
        self.Threshold = SNRThreshold
        self.u = None
        self.loadSMh5()
        self.mask, self.mask_indices = generate_mask(self.GridSize, sparsity, sampling_method, downsample_factor)
        self.loadSMData(K_select, use_coil)
        if MeasPath is not None:
            self.loadMeasData(MeasPath, Phantom_type=Phantom_type)
        self.uploadCoord(upsamplerate)
        self.Nk = len(self)
        self.Px = torch.linspace(0, 1, self.GridSize[0])[:, None]
        self.Py = torch.linspace(0, 1, self.GridSize[1])[:, None]
        self.Pz = torch.linspace(0, 1, self.GridSize[2])[:, None]

    def loadSMData(self, K_select, use_coil):
        if self.data_type == "OpenMPI":
            self.K_select = K_select
            K_use__ = []
            for coil, kset in enumerate(self.K_filter):
                for ks in kset:
                    K_use__.append(f"coil:{coil}-k:{ks}")
            use_coil_expand = [use_coil[0]] * len(self.K_filter[0]) + [use_coil[1]] * len(self.K_filter[1]) + [
                use_coil[2]] * len(
                self.K_filter[2])
            K_use = [sublist for sublist, flag in zip(K_use__, use_coil_expand) if flag]
            Su = []
            if K_select is not None:
                assert set(K_select).issubset(
                    K_use), f"Assertion failed: Some elements in K_select are not in K_use: {K_select}"
                K_use = K_select
                for i, ck in enumerate(K_select):
                    c = int(ck.split("-")[0].split(":")[1])
                    k = int(ck.split("-")[1].split(":")[1])
                    Su.append(self.S[c, k].copy())
                Su = np.vstack(Su).reshape(len(K_select), *self.GridSize)
                self.K_train_Num = [len(K_select)]
            else:
                Su = [self.S[idx, ku].copy().reshape(self.S[idx, ku].shape[0], *self.GridSize) for idx, (ku, flag) in
                      enumerate(zip(self.K_filter, use_coil))
                      if flag]
                Su = np.vstack(Su)
                self.K_train_Num = [len(k) for k in self.K_filter]
            self.Su = Su
            self.Su_observed_norm, self.normalizeParam = normalize(Su[:, self.mask], normalize_type="zscore")
            self.SNR_train = [sublist for sublist, flag in zip(self.SNR, use_coil) if flag]
            self.K_train = K_use
        elif self.data_type == "RealData":
            self.K_select = K_select
            self.Su = self.S.reshape(-1, 1, 32, 32)
            self.Su_observed_norm, self.normalizeParam = normalize(self.Su[:, self.mask], normalize_type="zscore")
            K_use__ = []
            for coil, kset in enumerate(self.K_filter):
                for ks in kset:
                    K_use__.append(f"coil:{coil}-k:{ks}")
            use_coil_expand = [use_coil[0]] * len(self.K_filter[0])
            K_use = [sublist for sublist, flag in zip(K_use__, use_coil_expand) if flag]
            self.K_train = K_use
            self.K_train_Num = [len(self.K_train)]
    def loadSMh5(self):
        if self.data_type == "OpenMPI":
            fSM = h5py.File(self.filenameSM, 'r')
            # read the full system matrix
            S = fSM['/measurement/data']
            # reinterpret to complex data
            isBG = fSM['/measurement/isBackgroundFrame'][:].view(bool)
            self.S = CustomArray(S, isBG)
            # get rid of background frames

            # generate frequency vector
            numFreq = round(fSM['/acquisition/receiver/numSamplingPoints'][()] / 2) + 1
            rxBandwidth = fSM['/acquisition/receiver/bandwidth'][()]
            self.freq = np.arange(0, numFreq) / (numFreq - 1) * rxBandwidth
            # # remove frequencies below 80 kHz and use only x/y receive channels
            idxMin = np.nonzero(np.ravel(self.freq[:] > 30e3))[0][0]
            SM_SNR_x = 10 * np.log10(fSM['/calibration/snr'][0, 0])
            SM_SNR_y = 10 * np.log10(fSM['/calibration/snr'][0, 1])
            SM_SNR_z = 10 * np.log10(fSM['/calibration/snr'][0, 2])
            self.GridSize = fSM['/calibration/size'][:]
            self.SNR = [SM_SNR_x, SM_SNR_y, SM_SNR_z]
            K_filter = [np.where(SM_SNR_x >= self.Threshold)[0], np.where(SM_SNR_y >= self.Threshold)[0],
                        np.where(SM_SNR_z >= self.Threshold)[0]]
            self.K_filter = [K_filter[0][K_filter[0] > idxMin], K_filter[1][K_filter[1] > idxMin],
                             K_filter[2][K_filter[2] > idxMin]]

            SNR_filter_list = [SM_SNR_x[self.K_filter[0]], SM_SNR_y[self.K_filter[1]], SM_SNR_z[self.K_filter[2]]]
            self.SNR_filter = 10 * np.log10(
                np.hstack([sublist for sublist, flag in zip(SNR_filter_list, self.use_coil) if flag]))
        elif self.data_type == "RealData":
            self.S = loadmat(self.filenameSM)[self.filenameSM.split('/')[-1].split('.')[0]]
            self.freq = np.arange(self.S.shape[0])
            self.GridSize = [1, 32, 32]
            self.SNR = [self.freq, self.freq, self.freq]
            self.SNR_filter = np.ones(self.S.shape[0])
            self.K_filter = [np.arange(self.S.shape[0])]

    def loadMeasData(self, MeasPath, Phantom_type):
        if self.data_type == "OpenMPI":
            self.u = {}
            assert len(MeasPath) == len(Phantom_type)
            self.phantomType = Phantom_type
            for mea_path, phantomType in zip(MeasPath, Phantom_type):
                filenameMeas = mea_path
                fMeas = h5py.File(filenameMeas, 'r')
                u = fMeas['/measurement/data']
                u = u[:, :, :, :].squeeze()
                if len(u.shape) == 4:
                    u_t = np.mean(u[0], axis=0)
                else:
                    u_t = np.mean(u, axis=0)
                uk = np.fft.rfft(u_t)  # 默认最后一个轴，返回长度为：奇数：（N+1）/2，偶数：N/2 +1
                u = [uk[0, self.K_filter[0]].copy(), uk[1, self.K_filter[1]].copy(), uk[2, self.K_filter[2]].copy()]
                self.u[phantomType] = np.hstack([sublist for sublist, flag in zip(u, self.use_coil) if flag])
        elif self.data_type == "RealData":
            self.u = {}
            assert len(MeasPath) == len(Phantom_type)
            self.phantomType = Phantom_type
            for mea_path, phantomType in zip(MeasPath, Phantom_type):
                self.u[phantomType] = loadmat(mea_path)[phantomType].squeeze()

    def uploadCoord(self, upsamplerate):
        self.Coord = generate_normalized_coordinates(*self.GridSize, uprate=upsamplerate, normalized=self.normalized)
        if upsamplerate > 1:
            self.DenseViewGridSize = self.Coord.shape[0:-1]
            self.GridNum = np.prod(self.DenseViewGridSize)
        else:
            self.DenseViewGridSize = self.GridSize
            self.GridNum = np.prod(self.GridSize)

    def __getitem__(self, idx):
        if self.datatype == "train":
            return self.Su_observed_norm[idx], self.K_train[idx]
        elif self.datatype == "test":
            return self.Su[idx], self.K_train[idx]

    def __len__(self):
        return len(self.K_train)

    def DataTypeSwitch(self, datatype):
        self.datatype = datatype

    @staticmethod
    def collate_fn(batch):
        S = [torch.from_numpy(item[0]) for item in batch]
        K = [item[1] for item in batch]
        return torch.stack(S), K


if __name__ == '__main__':
    SMdata = SystemMatrixLoad(r"H:\MPI-NFM\RealSM\FFP\S.mat", [
        r"H:\MPI-NFM\RealSM\FFP\u_3mm.mat",
        r"H:\MPI-NFM\RealSM\FFP\u_C.mat",
    ], 7, 0.25, "random", None, [
                                  True,
                                  False,
                                  False
                              ], 1, [
                                  "u_3mm",
                                  "u_C",
                              ], model_type="ATCTTC",data_type="RealData")