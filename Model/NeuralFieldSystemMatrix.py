import copy
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch import optim
from torch.utils.data import DataLoader

from Dataset.SystemMatrixLoad import SystemMatrixLoad, denormalize
from Model.NFModel import NFModel, SimplyNFModel
from Model.default_config import cfg_load
from utils.TestResultLogger import ReconStandardView, ReconDenseView, ResultSave, ReconResultSave, LossSaveAndPlot, \
    DenseViewReconPlot, SurfacePlot, SystemMatrixMetricCalculation, ReconResultMetricCalculation
from utils.loggerx import LoggerX


class SMNeuralFieldModel:
    def __init__(self, opt):
        super(SMNeuralFieldModel, self).__init__()
        # Section 配置及日志
        self.opt = opt
        self.opt_temp = copy.deepcopy(opt)
        self.logger = LoggerX(opt)
        if self.opt.save_freq < 0:
            # 不记录中间过程
            self.opt.save_freq = self.opt.max_epochs * 2
        if 'test' in self.opt.mode or 'recon' in self.opt.mode:
            assert self.opt.result_save_path is not None, 'Test result save path is required'

        # Section 读取系统矩阵
        self.load_data()
        self.c = {}

        # Section 模型
        if self.opt.NFMmode != "SimplyNFM":
            self.NFModel = NFModel(hidden_layers=self.opt.hidden_layers_dec, M_order=self.opt.M_order,
                                   hidden_layers_enc=self.opt.hidden_layers_enc, mode=self.opt.NFMmode,
                                   dec_index=self.SMdata.K_train, parallel_method=self.opt.parallel_method,
                                   ChebyOrder=self.opt.ChebyOrder, degree=self.opt.degree).to(self.opt.device)
        else:
            self.NFModel = SimplyNFModel(hidden_layers=self.opt.hidden_layers_dec, M_order=self.opt.M_order,
                                         hidden_layers_enc=self.opt.hidden_layers_enc, mode=self.opt.NFMmode,
                                         dec_index=self.SMdata.K_train, parallel_method=self.opt.parallel_method,
                                         ChebyOrder=self.opt.ChebyOrder, degree=self.opt.degree).to(self.opt.device)
        if self.opt.select_dec_idx is None:
            self.opt.select_dec_idx = self.SMdata.K_train

        # Section 优化器
        self.optimizer_dict = {}
        self.scheduler_dict = {}
        # 共享编码器
        if self.opt.hidden_layers_enc is not None:
            self.optimizer_dict = {
                'encoder': optim.RMSprop(self.NFModel.share_encoder.parameters(), lr=self.opt.init_lr, weight_decay=0,
                                         momentum=0.9)}
            self.scheduler_dict = {
                'encoder': optim.lr_scheduler.CosineAnnealingLR(self.optimizer_dict['encoder'],
                                                                T_max=self.opt.max_epochs,
                                                                eta_min=self.opt.init_lr / 5)}
        # 独立解码器
        for ck in self.opt.select_dec_idx:
            # self.optimizer_dict[f'decoder_{ck}'] = optim.Adam(self.NFModel.model[ck].parameters(),
            #                                                   lr=self.opt.init_lr, weight_decay=1e-3
            #                                                   )
            self.optimizer_dict[f'decoder_{ck}'] = optim.SGD(self.NFModel.model[ck].parameters(),
                                                             lr=self.opt.init_lr, momentum=0.9
                                                             )
            self.scheduler_dict[f'decoder_{ck}'] = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_dict[f'decoder_{ck}'],
                T_max=self.opt.max_epochs, eta_min=self.opt.init_lr / 5)

        self.loss_fn = nn.MSELoss()
        self.logger.modules = [self.NFModel]
        self.logger.module_names = ["NFModel"]
        self.load_model()

        # Section: reconstruction cfg
        # 初始化变量
        idx_snr = []
        start = 0
        # 生成多个 np.arange 数组
        for num in self.SMdata.K_train_Num:
            idx_snr.append(np.arange(start, start + num))
            start += num
        self.K_recon = [sublist for sublist, flag in zip(idx_snr, self.opt.Rx_Recon) if flag]
        pass

    def load_data(self):
        self.SMdata = SystemMatrixLoad(SMPath=self.opt.SM_path, MeasPath=self.opt.Mea_path,
                                       SNRThreshold=self.opt.SNR_Threshold,
                                       sparsity=self.opt.sparsity, sampling_method=self.opt.sampling_method,
                                       K_select=self.opt.select_dec_idx, use_coil=self.opt.Rx,
                                       upsamplerate=1, Phantom_type=self.opt.phantom_type,
                                       downsample_factor=self.opt.downsample_factor, model_type=self.opt.NFMmode, data_type=self.opt.data_type)
        self.TrainSMdataLoader = DataLoader(self.SMdata, batch_size=self.opt.batch_size, shuffle=False,
                                            collate_fn=self.SMdata.collate_fn)
        self.TestSMdataLoader = DataLoader(self.SMdata, batch_size=1, shuffle=False,
                                           collate_fn=self.SMdata.collate_fn)

    def switchReconRange(self, phantom_type, Calibration="5"):
        if Calibration == "5":
            if phantom_type == "Shape":
                self.opt.recon_plot_range_x = [12, 28]
                self.opt.recon_plot_range_y = [14, 21]
                self.opt.recon_plot_range_z = [8, 25]
            elif phantom_type == "Resolution":
                self.opt.recon_plot_range_x = [13, 27]
                self.opt.recon_plot_range_y = [10, 29]
                self.opt.recon_plot_range_z = [16, 19]
            elif phantom_type == "Concentration":
                self.opt.recon_plot_range_x = [13, 26]
                self.opt.recon_plot_range_y = [9, 14]
                self.opt.recon_plot_range_z = [22, 25]
            elif phantom_type == "Custom":
                assert self.recon_plot_range_x[1] <= self.SMdata.GridSize - 1 and self.recon_plot_range_y[
                    1] <= self.SMdata.GridSize - 1 and self.recon_plot_range_z[1] <= self.SMdata.GridSize - 1
        if Calibration == "6":
            if phantom_type == "Shape":
                self.opt.recon_plot_range_x = [12, 23]
                self.opt.recon_plot_range_y = [15, 21]
                self.opt.recon_plot_range_z = [11, 26]
            elif phantom_type == "Resolution":
                self.opt.recon_plot_range_x = [15, 23]
                self.opt.recon_plot_range_y = [11, 30]
                self.opt.recon_plot_range_z = [15, 19]
            elif phantom_type == "Concentration":
                self.opt.recon_plot_range_x = [22, 26]
                self.opt.recon_plot_range_y = [9, 14]
                self.opt.recon_plot_range_z = [22, 25]
            elif phantom_type == "Custom":
                assert self.recon_plot_range_x[1] <= self.SMdata.GridSize - 1 and self.recon_plot_range_y[
                    1] <= self.SMdata.GridSize - 1 and self.recon_plot_range_z[1] <= self.SMdata.GridSize - 1
        if Calibration == "Custom":
            self.opt.recon_plot_range_x = [0, 0]
            self.opt.recon_plot_range_y = [0, -1]
            self.opt.recon_plot_range_z = [0, -1]


    def update_opt(self, ultra_cfg=None):
        # 合并cfg
        if ultra_cfg is not None:
            cfg_load(ultra_cfg, self.opt.__dict__)
            self.logger.save_option(self.opt)

    def reset_opt(self):
        self.opt = copy.deepcopy(self.opt_temp)

    def load_model(self):
        if "train" in self.opt.mode and not self.opt.train_encoder:
            self.logger.load_share_encoder(
                r"H:\MPI-NFM\NeuralFieldSystemMat\HyperParamAblation\Order_train_log\NFM_Cheby-M-order_Calibration-5-Order_8_sparsity-0.0625\save_models\Best Epoch (minimum train loss) - 19890")
        if "test" in self.opt.mode:
            assert self.opt.load_model_path is not None and (
                    self.opt.resume_epochs > 0 or self.opt.load_best_param), "Please confirm that you have made correct configuration on param setting of test model "
        if self.opt.load_best_param and self.opt.load_model_path is not None:
            self.logger.load_best_checkpoints(self.opt.load_model_path)
        elif self.opt.resume_epochs > 0 and self.opt.load_model_path is not None:
            self.logger.load_checkpoints(self.opt.resume_epochs, self.opt.load_model_path)

    def train(self):
        best_loss = float('inf')  # 初始化最小 loss 为无穷大
        best_model_state = None  # 用于保存最优模型的状态
        loss_log = []

        with (tqdm.tqdm(total=self.opt.max_epochs - self.opt.resume_epochs, desc=f'training.....') as T):
            for epoch in range(self.opt.resume_epochs + 1, self.opt.max_epochs + 1):
                self.NFModel.train()
                self.SMdata.DataTypeSwitch("train")
                loss_temp = 0
                constraint_loss_temp = 0
                l2_loss = 0
                constraint_loss = 0
                l1_loss = 0
                for iters, inputs in enumerate(self.TrainSMdataLoader):
                    # 1. 随机打乱 tensor 和 coord
                    # 首先生成一个随机的索引
                    N = inputs[0].shape[1]
                    indices = torch.randperm(N)
                    # 在每个批次开始前禁用未用解码器的梯度
                    tensor = inputs[0].to(self.opt.device)[:, indices]
                    used_decoders = inputs[1]  # 获取当前批次用到的解码器索引
                    coord = self.SMdata.Coord.to(self.opt.device)[self.SMdata.mask, :][indices, :]
                    step = int(N * self.opt.train_percent)
                    start_idx = 0
                    while start_idx < N:
                        end_idx = min(start_idx + step, N)  # 计算当前步长下的结束索引
                        # 重置优化器的梯度
                        if self.opt.hidden_layers_enc is not None:
                            self.optimizer_dict['encoder'].zero_grad()
                        for decoder_idx in used_decoders:
                            self.optimizer_dict[f'decoder_{decoder_idx}'].zero_grad()
                        if "M-order" in self.opt.NFMmode:
                            # 前向传播计算输出
                            A_pred, Cp = self.NFModel(coord[start_idx:end_idx],
                                                      used_decoders, self.opt.train_encoder)
                            constraint_loss = self.opt.l2_struct * Cp[0]
                            l1_loss = Cp[1]
                            constraint_loss_temp += constraint_loss.sum(0).item()
                        elif "standard" in self.opt.NFMmode:
                            A_pred, _ = self.NFModel(coord[start_idx:end_idx], used_decoders, self.opt.train_encoder)
                        elif "Simply" in self.opt.NFMmode:
                            A_pred = self.NFModel(coord[start_idx:end_idx], self.SMdata.Px.to(self.opt.device),
                                                  self.SMdata.Py.to(self.opt.device),
                                                  self.SMdata.Pz.to(self.opt.device), used_decoders,
                                                  self.opt.train_encoder)

                        loss = (torch.norm(A_pred - torch.cat(
                            (tensor[:, start_idx:end_idx, None].real, tensor[:, start_idx:end_idx, None].imag),
                            dim=2)) ** 2) / (end_idx - start_idx + 1) + constraint_loss.sum(0)
                        # + 0.003 * torch.stack(l1_loss).sum(0)
                        # loss反向传播主要受到jit的分支个数影响，与输入大小关系不大
                        loss.backward()
                        # 更新共享编码器的梯度
                        if self.opt.hidden_layers_enc is not None and self.opt.train_encoder:
                            self.optimizer_dict['encoder'].step()
                        # 更新当前批次解码器的梯度
                        for decoder_idx in used_decoders:
                            torch.nn.utils.clip_grad_value_(self.NFModel.model[decoder_idx].parameters(), 1)
                            self.optimizer_dict[f'decoder_{decoder_idx}'].step()
                        loss_temp += loss.mean().item()
                        # 更新开始索引
                        start_idx += step


                T.update(1)
                # 打印当前 epoch 的 loss 和学习率
                if self.opt.hidden_layers_enc is not None and self.opt.train_encoder:
                    self.scheduler_dict['encoder'].step()
                for decoder_idx in self.opt.select_dec_idx:
                    self.scheduler_dict[f'decoder_{decoder_idx}'].step()
                current_lr = self.scheduler_dict[f'decoder_{self.opt.select_dec_idx[0]}'].get_last_lr()[0]
                # 计算 epoch 的平均损失
                avg_loss = loss_temp / len(self.opt.select_dec_idx)
                avg_constraint_loss = constraint_loss_temp / len(self.opt.select_dec_idx)
                loss_log.append(float(avg_loss))
                #     更新绘制损失图
                self.logger.summer.add_scalar("train/loss", avg_loss, global_step=epoch)
                # 如果当前 epoch 的 loss 小于之前的最小 loss，更新最优模型
                if self.opt.epoch_l2 is not None:
                    if avg_loss < best_loss and epoch >= self.opt.epoch_l2:
                        best_loss = avg_loss
                        best_epoch = epoch
                        best_model_state = copy.deepcopy(self.NFModel.state_dict())  # 保存当前模型的状态
                else:
                    if avg_loss < best_loss and epoch >= self.opt.max_epochs // 2:
                        best_loss = avg_loss
                        best_epoch = epoch
                        best_model_state = copy.deepcopy(self.NFModel.state_dict())  # 保存当前模型的状态
                print(
                    f"[Loss: {avg_loss:.6f}, Constrain Loss: {avg_constraint_loss:.6f}, LR: {current_lr:.6f}")

                if epoch % self.opt.save_freq == 0:
                    it = epoch
                    self.logger.checkpoints(it)
                    completed_tensor = self.test()
                    if self.opt.Rx_Recon != [False, False, False]:
                        for phantom in self.opt.phantom_type:
                            self.switchReconRange(phantom, self.opt.Calibration_num)
                            cfm = ReconStandardView(completed_tensor, phantom, c=self.c, S=self.SMdata.Su,
                                                    u=self.SMdata.u,
                                                    GridSize=self.SMdata.GridSize, Nk=self.SMdata.Nk,
                                                    K_recon=self.K_recon,SNR=self.SMdata.SNR_filter)
                            ReconMetric = ReconResultMetricCalculation(cfm, dx=self.opt.recon_plot_range_x,
                                                                       dy=self.opt.recon_plot_range_y,
                                                                       dz=self.opt.recon_plot_range_z, phantom=phantom,
                                                                       c=self.c)
                            savepath = f"{self.logger.current_save_dir}/{phantom}"
                            ReconResultSave(savepath=savepath, ReconMetric_dict=ReconMetric, cfm=cfm, phantom=phantom,
                                            c_gt=self.c, recon_plot_range_x=self.opt.recon_plot_range_x,
                                            recon_plot_range_y=self.opt.recon_plot_range_y,
                                            recon_plot_range_z=self.opt.recon_plot_range_z)
                    SMMetric = SystemMatrixMetricCalculation(completed_tensor, mask=self.SMdata.mask, Su=self.SMdata.Su)
                    ResultSave(self.logger.current_save_dir, completed_tensor, SMMetric, save_sm=False)
                    # SurfacePlot(completed_tensor, SMMetric, self.opt.plot_plane, self.SMdata.K_train, self.SMdata.Su,
                    #             self.SMdata.mask, self.SMdata.GridSize, self.SMdata.SNR, self.SMdata.freq,
                    #             self.logger.current_save_dir)
                    pass

            save_dir = osp.join(self.logger.models_save_dir, f'Best Epoch (minimum train loss) - {best_epoch}')
            self.logger.checkpoints(best_epoch, save_dir)
            self.NFModel.load_state_dict(best_model_state)
            completed_tensor = self.test()
            if self.opt.Rx_Recon != [False, False, False]:
                for phantom in self.opt.phantom_type:
                    self.switchReconRange(phantom, self.opt.Calibration_num)
                    cfm = ReconStandardView(completed_tensor, phantom, c=self.c, S=self.SMdata.Su, u=self.SMdata.u,
                                            GridSize=self.SMdata.GridSize, Nk=self.SMdata.Nk, K_recon=self.K_recon, SNR=self.SMdata.SNR_filter)
                    ReconMetric = ReconResultMetricCalculation(cfm, dx=self.opt.recon_plot_range_x,
                                                               dy=self.opt.recon_plot_range_y,
                                                               dz=self.opt.recon_plot_range_z, phantom=phantom,
                                                               c=self.c)
                    savepath = f"{self.logger.current_save_dir}/{phantom}"
                    ReconResultSave(savepath=savepath, ReconMetric_dict=ReconMetric, cfm=cfm, phantom=phantom,
                                    c_gt=self.c, recon_plot_range_x=self.opt.recon_plot_range_x,
                                    recon_plot_range_y=self.opt.recon_plot_range_y,
                                    recon_plot_range_z=self.opt.recon_plot_range_z)
            SMMetric = SystemMatrixMetricCalculation(completed_tensor, mask=self.SMdata.mask, Su=self.SMdata.Su)
            LossSaveAndPlot(self.logger.models_save_dir, loss_log, SMMetric)
            ResultSave(self.logger.current_save_dir, completed_tensor, SMMetric, save_sm=True)
            SurfacePlot(completed_tensor, SMMetric, self.opt.plot_plane, self.SMdata.K_train, self.SMdata.Su,
                        self.SMdata.mask, self.SMdata.GridSize, self.SMdata.SNR, self.SMdata.freq,
                        self.logger.current_save_dir)

    @staticmethod
    def project(weights, c):
        # 计算当前权重的L2范数
        norm = torch.norm(weights, p=2)

        # 如果L2范数超过了阈值c，则进行投影操作
        if norm > c:
            weights = weights * (c / norm)

        return weights

    @torch.no_grad()
    def test(self):
        self.NFModel.eval()
        A_pred = []
        self.SMdata.DataTypeSwitch("test")
        loss_temp = 0
        for iters, inputs in enumerate(self.TestSMdataLoader):
            N = self.SMdata.GridNum
            indices = torch.randperm(N)
            # 在每个批次开始前禁用未用解码器的梯度
            P = self.SMdata.Coord.to(self.opt.device)
            P_flat = P.view(-1, 3)[indices]
            used_decoders = inputs[1]  # 获取当前批次用到的解码器索引
            A = torch.zeros(len(used_decoders), N, 2)
            step = int(N * self.opt.test_percent)
            start_idx = 0
            while start_idx < N:
                end_idx = min(start_idx + step, N)  # 计算当前步长下的结束索引
                # TODO
                if self.opt.NFMmode == "SimplyNFM":
                    A[:, start_idx:end_idx, :] = self.NFModel(P_flat[start_idx:end_idx],
                                                              self.SMdata.Px.to(self.opt.device),
                                                              self.SMdata.Py.to(self.opt.device),
                                                              self.SMdata.Pz.to(self.opt.device),
                                                              used_decoders).detach()
                else:
                    A[:, start_idx:end_idx, :] = self.NFModel(P_flat[start_idx:end_idx], used_decoders)[0].detach()
                start_idx += step
            A_pred.append(A[:, torch.argsort(indices), :].view(-1, *P.shape[:-1], 2))
        completed_tensor = torch.cat(A_pred)  # 取消梯度追踪
        completed_tensor = torch.complex(completed_tensor[:, :, :, :, 0], completed_tensor[:, :, :, :, 1])
        completed_tensor = denormalize(completed_tensor.cpu().numpy(), self.SMdata.normalizeParam,
                                       normalize_type="zscore")
        return completed_tensor

    def fit(self):
        opt = self.opt
        if 'train' in opt.mode:
            self.train()

        elif 'test' in opt.mode:
            self.test()

        elif 'recon' in opt.mode:
            assert self.opt.UpsampleRate >= 1 and isinstance(self.opt.UpsampleRate,
                                                             int), 'Upsampling rate should be an integer greater than 1'
            self.SMdata.uploadCoord(self.opt.UpsampleRate)
            t1 = time.time()
            completed_tensor = self.test()
            t2 = time.time()
            print(f"[Test time: {t2 - t1:}")
            if self.opt.UpsampleRate == 1:
                for phantom in self.opt.phantom_type:
                    self.switchReconRange(phantom, self.opt.Calibration_num)
                    cfm = ReconStandardView(completed_tensor, phantom, c=self.c, S=self.SMdata.Su, u=self.SMdata.u,
                                            GridSize=self.SMdata.GridSize, Nk=self.SMdata.Nk, K_recon=self.K_recon,SNR=self.SMdata.SNR_filter)
                    ReconMetric = ReconResultMetricCalculation(cfm, dx=self.opt.recon_plot_range_x,
                                                               dy=self.opt.recon_plot_range_y,
                                                               dz=self.opt.recon_plot_range_z, phantom=phantom,
                                                               c=self.c)
                    savepath = f"{self.logger.save_root}/StandardView/{phantom}"
                    ReconResultSave(savepath=savepath, ReconMetric_dict=ReconMetric, cfm=cfm, phantom=phantom,
                                    c_gt=self.c, recon_plot_range_x=self.opt.recon_plot_range_x,
                                    recon_plot_range_y=self.opt.recon_plot_range_y,
                                    recon_plot_range_z=self.opt.recon_plot_range_z)
                    print(f"{phantom}-PSNR:", ReconMetric['psnr_3d_batch'])
                    print(f"{phantom}-SSIM:", ReconMetric['ssim_3d_batch'])
                SMMetric = SystemMatrixMetricCalculation(completed_tensor, mask=self.SMdata.mask, Su=self.SMdata.Su)
                ResultSave(self.logger.save_root, completed_tensor, SMMetric, save_sm=False)
                print("nRMSE", SMMetric["total_nrmse"])
                print("PSNR", SMMetric["total_psnr"])
                # self.SurfacePlot(completed_tensor, SMMetric)

            else:
                for phantom in self.opt.phantom_type:
                    self.switchReconRange(phantom, self.opt.Calibration_num)
                    cfm = ReconDenseView(completed_tensor, phantom, c=self.c, S=self.SMdata.Su, u=self.SMdata.u,
                                         GridSize=self.SMdata.GridSize, DenseViewGridSize=self.SMdata.DenseViewGridSize,
                                         Nk=self.SMdata.Nk, K_recon=self.K_recon, SNR=self.SMdata.SNR_filter)
                    savepath = f"{self.logger.save_root}/DenseView-{self.opt.UpsampleRate}x/{phantom}"
                    DenseViewReconPlot(savepath, cfm, dx=self.opt.recon_plot_range_x, dy=self.opt.recon_plot_range_y,
                                       dz=self.opt.recon_plot_range_z, phantom=phantom, c=self.c,
                                       UpsampleRate=self.opt.UpsampleRate)

                # DenseViewSurfacePlot(completed_tensor, self.opt.plot_plane, self.SMdata.K_train, self.SMdata.Su,
                #                      self.SMdata.mask, self.SMdata.GridSize, self.SMdata.SNR, self.SMdata.freq,
                #                      self.logger.save_root, self.opt.UpsampleRate, self.SMdata.DenseViewGridSize)


if __name__ == '__main__':
    NFM = SMNeuralFieldModel(None)
    NFM.fit()
