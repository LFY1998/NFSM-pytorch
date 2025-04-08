import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import json

from numpy.linalg import norm

from Recon.kaczmarzReg import kaczmarzReg
from utils.metric_calculator import calculate_nrmse, calculate_psnr, calculate_ssim

matplotlib.use('Agg')


def ReconStandardView(completed_tensor, phantom, c, S, u, GridSize, Nk, K_recon, SNR):
    # lam = norm(S.reshape(Nk, -1), ord='fro') * 1e-9
    # lam = norm(S.reshape(Nk, -1), ord='fro') * 2e-1
    lam = norm(S.reshape(Nk, -1), ord='fro') * 1e-7
    # lam = 0
    if phantom not in c.keys():
        c_gt = kaczmarzReg(ExtractSelectedFreq(S.reshape(Nk, -1), K_recon),
                           ExtractSelectedFreq(u[phantom][:, None], K_recon), GridSize, 3, lam,
                           True, True, False,
                           tv_iters=0, SNR=None, power=0).real
        # ExtractSelectedFreq(SNR[:, None], K_recon)[:, 0]
        c[phantom] = np.reshape(c_gt, GridSize)
    c_nfm = kaczmarzReg(ExtractSelectedFreq(completed_tensor.reshape(Nk, -1), K_recon),
                        ExtractSelectedFreq(u[phantom][:, None], K_recon), GridSize, 3, lam,
                        True, True, False,
                        tv_iters=0, SNR=None, power=0).real

    c_nfm = np.reshape(c_nfm, GridSize)
    return c_nfm


def ReconDenseView(completed_tensor, phantom, c, S, u, GridSize, DenseViewGridSize, Nk, K_recon, SNR):
    lam = norm(S.reshape(Nk, -1), ord='fro') * 1e-7
    if phantom not in c.keys():
        c_gt = kaczmarzReg(ExtractSelectedFreq(S.reshape(Nk, -1), K_recon),
                           ExtractSelectedFreq(u[phantom][:, None], K_recon), GridSize, 3, lam,
                           True, True, False,
                           tv_iters=0, SNR=ExtractSelectedFreq(SNR[:, None], K_recon)[:, 0], power=0).real
        c[phantom] = np.reshape(c_gt, GridSize)
    c_nfm = kaczmarzReg(ExtractSelectedFreq(completed_tensor.reshape(Nk, -1), K_recon),
                        ExtractSelectedFreq(u[phantom][:, None], K_recon), DenseViewGridSize,
                        3, lam, True, True, False,
                        tv_iters=0, SNR=ExtractSelectedFreq(SNR[:, None], K_recon)[:, 0], power=0.5).real

    c_nfm = np.reshape(c_nfm, DenseViewGridSize)
    return c_nfm


def ExtractSelectedFreq(systemMatrix, K_recon):
    return np.vstack([systemMatrix[idk,] for idk in K_recon])


def ResultSave(current_save_dir, completed_tensor, SMMetric_dict, save_sm=True):
    savepath = current_save_dir
    if save_sm:
        np.save(f"{savepath}/CompletedSystemMatrix.npy", completed_tensor)
    with open(f'{savepath}/SMMetric.json', 'w') as json_file:
        json.dump(SMMetric_dict, json_file, indent=4, ensure_ascii=False)


def ReconResultSave(savepath, ReconMetric_dict, cfm, phantom, c_gt, recon_plot_range_x, recon_plot_range_y,
                    recon_plot_range_z):
    os.makedirs(savepath, exist_ok=True)
    np.save(f"{savepath}/ReconResult.npy", cfm)
    with open(f'{savepath}/ReconMetric.json', 'w') as json_file:
        json.dump(ReconMetric_dict, json_file, indent=4, ensure_ascii=False)
    ReconPlot(savepath, cfm, ReconMetric_dict, recon_plot_range_x, recon_plot_range_y,
              recon_plot_range_z, phantom, c_gt)


def LossSaveAndPlot(models_save_dir, loss_log, SMMetric_dict):
    """
    绘制损失曲线并在右上角显示平均指标。
    """
    # 获取平均指标
    avg_total_error_abs, avg_total_error_real, avg_total_error_imag = SMMetric_dict["total_error"]
    _, avg_total_nrmse_real, avg_total_nrmse_imag, _ = SMMetric_dict["total_nrmse"]
    _, avg_total_psnr_real, avg_total_psnr_imag, _ = SMMetric_dict["total_psnr"]

    # 绘制 loss_log 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(loss_log[10:], label="Loss over Epochs", color="blue")

    # 格式化平均指标文本，增加对齐效果
    avg_text = (
        # f"Error (abs) :{avg_total_error_abs}\n"
        # f"Error (real):{avg_total_error_real}\n"
        # f"Error (imag):{avg_total_error_imag}\n"
        f"NRMSE (real):{avg_total_nrmse_real}\n"
        f"NRMSE (imag):{avg_total_nrmse_imag}\n"
        f"PSNR  (real):{avg_total_psnr_real}\n"
        f"PSNR  (imag):{avg_total_psnr_imag}"
    )

    # 在右上角显示平均指标，设置等宽字体和左对齐
    plt.text(0.95, 0.95, avg_text, ha='right', va='top', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7), family='monospace', multialignment='left')

    # 添加标题和标签
    plt.title("Loss Curve with Average Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # 显示图像
    plt.legend()
    plt.grid(True)
    savepath = f"{models_save_dir}/Loss.png"
    plt.savefig(savepath)
    plt.close()


def ReconPlot(save_path, cfm, ReconMetric_dict, dx, dy, dz, phantom, c):
    c_gt = c[phantom]
    savepath_root = save_path

    def plot_and_save(slice_gt, slice_recon, metric_psnr, metric_ssim, slice_type, index):
        """ Helper function to plot and save slice images. """
        res = np.abs(slice_gt - slice_recon)
        fig, ax = plt.subplots(1, 3, figsize=(30, 10))

        # Ground truth slice
        ax[0].set_title(f"Recon by Origin SM", fontsize=35, y=1.1)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].imshow(slice_gt, "gray", vmin=0, vmax=slice_gt.max())

        # Reconstructed slice
        ax[1].set_title(f"Recon by NFM SM", fontsize=35, y=1.1)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        s = f'PSNR={metric_psnr:.2f}dB, SSIM={metric_ssim:.4f}'
        ax[1].text(x=0.5, y=-0.1, s=s, fontsize=25, horizontalalignment='center', transform=ax[1].transAxes)
        ax[1].imshow(slice_recon, "gray", vmin=0, vmax=slice_recon.max())

        # Residual map
        ax[2].set_title("Residual Map", fontsize=35, y=1.1)
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].imshow(res, cmap="viridis", vmin=0, vmax=slice_gt.max())

        # Save the plot
        save_path = f"{savepath_root}/{slice_type}-slice/"
        os.makedirs(save_path, exist_ok=True)
        save_path = f"{savepath_root}/{slice_type}-slice/{slice_type}={index}.png"
        plt.savefig(save_path, dpi=75)
        plt.close(fig)

    # Plot yz slices
    for step, x in enumerate(range(dx[0], dx[1] + 1)):
        cyz_gt = c_gt[x, :, :]
        cyz = cfm[x, :, :]
        plot_and_save(cyz_gt, cyz, ReconMetric_dict["psnr_yz"][step], ReconMetric_dict["ssim_yz"][step], "yz", x)

    # Plot xz slices
    for step, y in enumerate(range(dy[0], dy[1] + 1)):
        cxz_gt = c_gt[:, y, :]
        cxz = cfm[:, y, :]
        plot_and_save(cxz_gt, cxz, ReconMetric_dict["psnr_xz"][step], ReconMetric_dict["ssim_xz"][step], "xz", y)

    # Plot xy slices
    for step, z in enumerate(range(dz[0], dz[1] + 1)):
        cxy_gt = c_gt[:, :, z]
        cxy = cfm[:, :, z]
        plot_and_save(cxy_gt, cxy, ReconMetric_dict["psnr_xy"][step], ReconMetric_dict["ssim_xy"][step], "xy", z)


def DenseViewReconPlot(savepath, cfm, dx, dy, dz, phantom, c, UpsampleRate):
    c_gt = c[phantom]
    savepath_root = savepath

    def plot_and_save(slice_gt, slice_recon, slice_type, index):
        """ Helper function to plot and save slice images. """
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        # Ground truth slice
        ax[0].set_title(f"Recon by Origin SM", fontsize=35, y=1.1)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].imshow(slice_gt, cmap="gray", vmin=0, vmax=slice_gt.max())

        # 上采样的 slice 显示，调整 extent 使其在画布上大小一致
        height, width = slice_gt.shape  # 获取原始图像尺寸
        ax[1].set_title(f"Recon by NFM SM \n [UpsampleRate={UpsampleRate}]", fontsize=35, y=1.05)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # 设置 extent 为原始图像的尺寸
        ax[1].imshow(slice_recon, cmap="gray", vmin=0, vmax=slice_recon.max(),
                     extent=[0, width, 0, height])

        # Save the plot
        save_path = f"{savepath_root}/{slice_type}-slice/"
        os.makedirs(save_path, exist_ok=True)
        save_path = f"{save_path}{slice_type}={index}.png"
        plt.savefig(save_path, dpi=75)
        plt.close(fig)

    # Plot yz slices
    for step, x in enumerate(range(dx[0], dx[1] + 1)):
        cyz_gt = c_gt[x, :, :]
        cyz = cfm[UpsampleRate * x, :, :]
        plot_and_save(cyz_gt, cyz, "yz", x)

    # Plot xz slices
    for step, y in enumerate(range(dy[0], dy[1] + 1)):
        cxz_gt = c_gt[:, y, :]
        cxz = cfm[:, UpsampleRate * y, :]
        plot_and_save(cxz_gt, cxz, "xz", y)

    # Plot xy slices
    for step, z in enumerate(range(dz[0], dz[1] + 1)):
        cxy_gt = c_gt[:, :, z]
        cxy = cfm[:, :, UpsampleRate * z]
        plot_and_save(cxy_gt, cxy, "xy", z)


def SurfacePlot(completed_tensor, SMMetric_dict, plot_plane, K_train, Su, mask, GridSize, SNR, freq, current_save_dir):
    for i, ck in enumerate(K_train):
        plane = plot_plane
        plot_tensor = Su[i, plane, :, :].real
        plot_observed_tensor = plot_tensor.copy()
        plot_observed_tensor[~mask[plane, :, :]] = np.nan
        plot_completed_tensor = completed_tensor[i, plane, :, :].real
        X = np.arange(GridSize[1])
        Y = np.arange(GridSize[2])
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(241, projection='3d')
        ax1.plot_surface(X, Y, plot_tensor, cmap='viridis')
        ax1.title.set_text('origin-real')
        ax2 = fig.add_subplot(242, projection='3d')
        ax2.plot_surface(X, Y, plot_completed_tensor, cmap='viridis')
        ax2.title.set_text('completed-real')
        ax3 = fig.add_subplot(243, projection='3d')
        ax3.scatter(X, Y, plot_observed_tensor, c=plot_observed_tensor[mask[plane, :, :]], marker='o',
                    s=3, alpha=1)
        ax3.title.set_text('observed-real')
        # 添加颜色条
        ax4 = fig.add_subplot(244, projection='3d')
        ax4.plot_surface(X, Y, (plot_completed_tensor - plot_tensor), cmap='viridis')
        ax4.title.set_text('error-real')
        Min = plot_tensor.min()
        Max = plot_tensor.max()
        AbsMax = np.abs(plot_tensor).max()
        ax1.set_zlim(Min.min(), Max.max())
        ax2.set_zlim(Min.min(), Max.max())
        ax3.set_zlim(Min.min(), Max.max())
        ax4.set_zlim(-AbsMax, AbsMax)

        plot_tensor = Su[i, plane, :, :].imag
        plot_observed_tensor = plot_tensor.copy()
        plot_observed_tensor[~mask[plane, :, :]] = np.nan
        plot_completed_tensor = completed_tensor[i, plane, :, :].imag
        ax5 = fig.add_subplot(245, projection='3d')
        ax5.plot_surface(X, Y, plot_tensor, cmap='viridis')
        ax5.title.set_text('origin-imag')
        ax6 = fig.add_subplot(246, projection='3d')
        ax6.plot_surface(X, Y, plot_completed_tensor, cmap='viridis')
        ax6.title.set_text('completed-imag')
        ax7 = fig.add_subplot(247, projection='3d')
        ax7.scatter(X, Y, plot_observed_tensor, c=plot_observed_tensor[mask[plane, :, :]], marker='o',
                    s=3, alpha=1)
        ax7.title.set_text('observed-imag')
        # 添加颜色条
        ax8 = fig.add_subplot(248, projection='3d')
        # fig.colorbar(scatter, ax=ax4, label='Z Value')
        ax8.plot_surface(X, Y, (plot_completed_tensor - plot_tensor), cmap='viridis')
        ax8.title.set_text('error-imag')
        Min = plot_tensor.min()
        Max = plot_tensor.max()
        AbsMax = np.abs(plot_tensor).max()
        ax5.set_zlim(Min.min(), Max.max())
        ax6.set_zlim(Min.min(), Max.max())
        ax7.set_zlim(Min.min(), Max.max())
        ax8.set_zlim(-AbsMax, AbsMax)
        nrmse_real = SMMetric_dict["batch_nrmse_real"][i]
        nrmse_imag = SMMetric_dict["batch_nrmse_imag"][i]
        psnr_real = SMMetric_dict["batch_psnr_real"][i]
        psnr_imag = SMMetric_dict["batch_psnr_imag"][i]
        c = int(ck.split("-")[0].split(":")[1])
        coil = ["rx", "ry", "rz"][c]
        k = int(ck.split("-")[1].split(":")[1])
        plt.suptitle(
            f"plot plane=xy-z{plot_plane}  SNR={SNR[c][k]:.2f}dB  Freq={freq[k] / 1e3:.2f}kHZ \n  nRMSEr: {nrmse_real * 100:.2f}%, PSNRr:{psnr_real:.2f}, nRMSEi: {nrmse_imag * 100:.2f}%, PSNRi:{psnr_imag:.2f}")
        savepath_root = current_save_dir
        save_path = f"{savepath_root}/coil-{coil}/"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig("{}K={}.png".format(save_path, k))
        plt.close(fig)


def DenseViewSurfacePlot(completed_tensor, plot_plane, K_train, Su, mask, GridSize, SNR, freq, save_root, UpsampleRate,
                         DenseViewGridSize):
    for i, ck in enumerate(K_train):
        plane = plot_plane
        plot_tensor = Su[i, plane, :, :].real
        plot_observed_tensor = plot_tensor.copy()
        plot_observed_tensor[~mask[plane, :, :]] = np.nan
        plot_completed_tensor = completed_tensor[i, UpsampleRate * plane, :, :].real
        X = np.arange(GridSize[0])
        Y = np.arange(GridSize[1])
        X, Y = np.meshgrid(X, Y)
        XD = np.linspace(0, GridSize[0] - 1, DenseViewGridSize[0])
        YD = np.linspace(0, GridSize[1] - 1, DenseViewGridSize[1])
        XD, YD = np.meshgrid(XD, YD)
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot_surface(X, Y, plot_tensor, cmap='viridis', rstride=1, cstride=1)
        ax1.title.set_text('origin-real')
        ax2 = fig.add_subplot(232, projection='3d')
        ax2.plot_surface(XD, YD, plot_completed_tensor, cmap='viridis', rstride=1, cstride=1)
        ax2.title.set_text('completed-real (upsample)')
        ax3 = fig.add_subplot(233, projection='3d')
        ax3.scatter(X, Y, plot_observed_tensor, c=plot_observed_tensor[mask[plane, :, :]], marker='o',
                    s=3, alpha=1)
        ax3.title.set_text('observed-real')

        Min = plot_tensor.min()
        Max = plot_tensor.max()
        ax1.set_zlim(Min.min(), Max.max())
        ax2.set_zlim(Min.min(), Max.max())
        ax3.set_zlim(Min.min(), Max.max())

        plot_tensor = Su[i, plane, :, :].imag
        plot_observed_tensor = plot_tensor.copy()
        plot_observed_tensor[~mask[plane, :, :]] = np.nan
        plot_completed_tensor = completed_tensor[i, UpsampleRate * plane, :, :].imag
        ax5 = fig.add_subplot(234, projection='3d')
        ax5.plot_surface(X, Y, plot_tensor, cmap='viridis', rstride=1, cstride=1)
        ax5.title.set_text('origin-imag')
        ax6 = fig.add_subplot(235, projection='3d')
        ax6.plot_surface(XD, YD, plot_completed_tensor, cmap='viridis', rstride=1, cstride=1)
        ax6.title.set_text('completed-imag (upsample)')
        ax7 = fig.add_subplot(236, projection='3d')
        ax7.scatter(X, Y, plot_observed_tensor, c=plot_observed_tensor[mask[plane, :, :]], marker='o',
                    s=3, alpha=1)
        ax7.title.set_text('observed-imag')

        Min = plot_tensor.min()
        Max = plot_tensor.max()
        ax5.set_zlim(Min.min(), Max.max())
        ax6.set_zlim(Min.min(), Max.max())
        ax7.set_zlim(Min.min(), Max.max())
        c = int(ck.split("-")[0].split(":")[1])
        coil = ["rx", "ry", "rz"][c]
        k = int(ck.split("-")[1].split(":")[1])
        plt.suptitle(
            f"[plot plane=xy-z{plot_plane}]  [SNR={SNR[c][k]:.2f}dB]  [Freq={freq[k] / 1e3:.2f}kHZ] [Upsample rate={UpsampleRate}] ")
        savepath_root = save_root
        save_path = f"{savepath_root}/DenseView/coil-{coil}/"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig("{}K={}.png".format(save_path, k))
        plt.close(fig)


def SystemMatrixMetricCalculation(completed_tensor, mask, Su):
    error_mask = ~mask  # 计算缺失部分的误差
    tensor = Su[:, error_mask]
    completed_tensor = completed_tensor[:, error_mask]
    B = tensor.shape[0]  # 批量大小

    # 初始化列表来存储每个张量的误差
    batch_errors = []
    batch_nrmse_real = []
    batch_nrmse_imag = []
    batch_nrmse_abs = []
    batch_psnr_real = []
    batch_psnr_imag = []
    batch_psnr_abs = []
    batch_mae_real = []
    batch_mae_imag = []
    batch_mae_abs = []
    batch_r2_real = []
    batch_r2_imag = []
    batch_r2_abs = []

    # 遍历每个张量计算误差
    for b in range(B):
        # 计算每个张量的 real、imag 和 abs 的当前误差
        current_error_real = float(np.linalg.norm(
            (tensor[b].real - completed_tensor[b].real)) / np.linalg.norm(
            tensor[b].real))
        current_error_imag = float(np.linalg.norm(
            (tensor[b].imag - completed_tensor[b].imag)) / np.linalg.norm(
            tensor[b].imag))
        current_error_abs = float(np.linalg.norm(
            (np.abs(tensor[b]) - np.abs(completed_tensor[b]))) / np.linalg.norm(
            np.abs(tensor[b])))

        # 记录当前张量的误差
        batch_errors.append([current_error_abs, current_error_real, current_error_imag])

        # 计算 NRMSE 和 PSNR
        nrmse_real = float(calculate_nrmse(tensor[b].real, completed_tensor[b].real))
        nrmse_imag = float(calculate_nrmse(tensor[b].imag, completed_tensor[b].imag))
        nrmse_abs = float(calculate_nrmse(np.abs(tensor[b]), np.abs(completed_tensor[b])))
        psnr_real = float(calculate_psnr(tensor[b].real, completed_tensor[b].real))
        psnr_imag = float(calculate_psnr(tensor[b].imag, completed_tensor[b].imag))
        psnr_abs = float(calculate_psnr(np.abs(tensor[b]), np.abs(completed_tensor[b])))
        # 计算 MAE
        mae_real = float(np.mean(np.abs(tensor[b].real - completed_tensor[b].real)))
        mae_imag = float(np.mean(np.abs(tensor[b].imag - completed_tensor[b].imag)))
        mae_abs = float(np.mean(np.abs(np.abs(tensor[b]) - np.abs(completed_tensor[b]))))
        # 计算 R²
        r2_real = 1 - np.sum((tensor[b].real - completed_tensor[b].real) ** 2) / np.sum(
            (tensor[b].real - np.mean(tensor[b].real)) ** 2)
        r2_imag = 1 - np.sum((tensor[b].imag - completed_tensor[b].imag) ** 2) / np.sum(
            (tensor[b].imag - np.mean(tensor[b].imag)) ** 2)
        r2_abs = 1 - np.sum((np.abs(tensor[b]) - np.abs(completed_tensor[b])) ** 2) / np.sum(
            (np.abs(tensor[b]) - np.mean(np.abs(tensor[b]))) ** 2)

        # 记录每个张量的 NRMSE、PSNR、MAE 和 R²
        batch_nrmse_real.append(nrmse_real)
        batch_nrmse_imag.append(nrmse_imag)
        batch_psnr_real.append(psnr_real)
        batch_psnr_imag.append(psnr_imag)
        batch_nrmse_abs.append(nrmse_abs)
        batch_psnr_abs.append(psnr_abs)
        batch_mae_real.append(mae_real)
        batch_mae_imag.append(mae_imag)
        batch_mae_abs.append(mae_abs)
        batch_r2_real.append(r2_real)
        batch_r2_imag.append(r2_imag)
        batch_r2_abs.append(r2_abs)

    # 计算整体的平均误差
    total_error_abs = float(np.mean([error[0] for error in batch_errors]))
    total_error_real = float(np.mean([error[1] for error in batch_errors]))
    total_error_imag = float(np.mean([error[2] for error in batch_errors]))

    total_nrmse_real = float(np.mean(batch_nrmse_real)) * 1e2
    total_nrmse_imag = float(np.mean(batch_nrmse_imag)) * 1e2
    total_nrmse_abs = float(np.mean(batch_nrmse_abs)) * 1e2
    total_psnr_real = float(np.mean(batch_psnr_real))
    total_psnr_imag = float(np.mean(batch_psnr_imag))
    total_psnr_abs = float(np.mean(batch_psnr_abs))
    total_mae_real = float(np.mean(batch_mae_real))
    total_mae_imag = float(np.mean(batch_mae_imag))
    total_mae_abs = float(np.mean(batch_mae_abs))
    total_r2_real = float(np.mean(batch_r2_real))
    total_r2_imag = float(np.mean(batch_r2_imag))
    total_r2_abs = float(np.mean(batch_r2_abs))

    # 计算每个总误差指标的标准差
    std_error_abs = float(np.std([error[0] for error in batch_errors]))
    std_error_real = float(np.std([error[1] for error in batch_errors]))
    std_error_imag = float(np.std([error[2] for error in batch_errors]))

    std_nrmse_real = float(np.std(batch_nrmse_real)) * 1e2
    std_nrmse_imag = float(np.std(batch_nrmse_imag)) * 1e2
    std_nrmse_abs = float(np.std(batch_nrmse_abs)) * 1e2
    std_psnr_real = float(np.std(batch_psnr_real))
    std_psnr_imag = float(np.std(batch_psnr_imag))
    std_psnr_abs = float(np.std(batch_psnr_abs))
    std_mae_real = float(np.std(batch_mae_real))
    std_mae_imag = float(np.std(batch_mae_imag))
    std_mae_abs = float(np.std(batch_mae_abs))
    std_r2_real = float(np.std(batch_r2_real))
    std_r2_imag = float(np.std(batch_r2_imag))
    std_r2_abs = float(np.std(batch_r2_abs))

    # 格式化输出字符串
    results = {
        "total_error": [
            f"{total_error_abs:.4f}±{std_error_abs:.4f}",
            f"{total_error_real:.4f}±{std_error_real:.4f}",
            f"{total_error_imag:.4f}±{std_error_imag:.4f}"
        ],
        "total_nrmse": [
            f"Abs={total_nrmse_abs:.2f}%±{std_nrmse_abs:.2f}%",
            f"Real={total_nrmse_real:.2f}%±{std_nrmse_real:.2f}%",
            f"Imag={total_nrmse_imag:.2f}%±{std_nrmse_imag:.2f}%",
            f"Mean={(total_nrmse_real + total_nrmse_imag) / 2:.2f}%±{(std_nrmse_imag + std_nrmse_real) / 2:.2f}%",
        ],
        "total_psnr": [
            f"Abs={total_psnr_abs:.2f}±{std_psnr_abs:.2f}",
            f"Real={total_psnr_real:.2f}±{std_psnr_real:.2f}",
            f"Imag={total_psnr_imag:.2f}±{std_psnr_imag:.2f}",
            f"Mean={(total_psnr_real + total_psnr_imag) / 2:.2f}±{(std_psnr_imag + std_psnr_real) / 2:.2f}",
        ],
        "total_mae": [
            f"Abs={total_mae_abs:.4f}±{std_mae_abs:.4f}",
            f"Real={total_mae_real:.4f}±{std_mae_real:.4f}",
            f"Imag={total_mae_imag:.4f}±{std_mae_imag:.4f}",
            f"Mean={(total_mae_real + total_mae_imag) / 2:.4f}±{(std_mae_imag + std_mae_real) / 2:.4f}",
        ],
        "total_r2": [
            f"Abs={total_r2_abs:.4f}±{std_r2_abs:.4f}",
            f"Real={total_r2_real:.4f}±{std_r2_real:.4f}",
            f"Imag={total_r2_imag:.4f}±{std_r2_imag:.4f}",
            f"Mean={(total_r2_real + total_r2_imag) / 2:.4f}±{(std_r2_imag + std_r2_real) / 2:.4f}",
        ],
        "batch_errors": batch_errors,
        "batch_nrmse_real": batch_nrmse_real,
        "batch_nrmse_imag": batch_nrmse_imag,
        "batch_psnr_real": batch_psnr_real,
        "batch_psnr_imag": batch_psnr_imag,
    }

    # 返回格式化字符串结果
    return results


def ReconResultMetricCalculation(cfm, dx, dy, dz, phantom, c):
    c_gt = c[phantom]

    # 初始化存储每个平面的指标列表
    psnr_yz = []
    ssim_yz = []
    psnr_xy = []
    ssim_xy = []
    psnr_xz = []
    ssim_xz = []

    # 计算 yz 平面上的 PSNR 和 SSIM
    for x in range(dx[0], dx[1] + 1):
        cyz_gt = c_gt[x, :, :]
        cyz = cfm[x, :, :]
        psnr_yz.append(calculate_psnr(cyz, cyz_gt))
        ssim_yz.append(calculate_ssim(cyz, cyz_gt))

    # 计算 xy 平面上的 PSNR 和 SSIM
    for z in range(dz[0], dz[1] + 1):
        cxy_gt = c_gt[:, :, z]
        cxy = cfm[:, :, z]
        psnr_xy.append(calculate_psnr(cxy, cxy_gt))
        ssim_xy.append(calculate_ssim(cxy, cxy_gt))

    # 计算 xz 平面上的 PSNR 和 SSIM
    for y in range(dy[0], dy[1] + 1):
        cxz_gt = c_gt[:, y, :]
        cxz = cfm[:, y, :]
        psnr_xz.append(calculate_psnr(cxz, cxz_gt))
        ssim_xz.append(calculate_ssim(cxz, cxz_gt))

    results = {
        "psnr_3d_batch": [
            f"{(np.mean(psnr_yz) + np.mean(psnr_xz) + np.mean(psnr_xy)) / 3:.2f}±{(np.std(psnr_yz) + np.std(psnr_xz) + np.std(psnr_xy)) / 3:.2f}"
        ],
        "ssim_3d_batch": [
            f"{(np.mean(ssim_yz) + np.mean(ssim_xz) + np.mean(ssim_xy)) / 3:.4f}±{(np.std(ssim_yz) + np.std(ssim_xz) + np.std(ssim_xy)) / 3:.4f}"
        ],
        "psnr_yz_batch": [
            f"{np.mean(psnr_yz):.2f}±{np.std(psnr_yz):.2f}"
        ],
        "ssim_yz_batch": [
            f"{np.mean(ssim_yz):.4f}±{np.std(ssim_yz):.4f}"
        ],
        "psnr_xy_batch": [
            f"{np.mean(psnr_xy):.2f}±{np.std(psnr_xy):.2f}"
        ],
        "ssim_xy_batch": [
            f"{np.mean(ssim_xy):.4f}±{np.std(ssim_xy):.4f}"
        ],
        "psnr_xz_batch": [
            f"{np.mean(psnr_xz):.2f}±{np.std(psnr_xz):.2f}"
        ],
        "ssim_xz_batch": [
            f"{np.mean(ssim_xz):.4f}±{np.std(ssim_xz):.4f}"
        ],
        "psnr_yz": psnr_yz,
        "ssim_yz": ssim_yz,
        "psnr_xy": psnr_xy,
        "ssim_xy": ssim_xy,
        "psnr_xz": psnr_xz,
        "ssim_xz": ssim_xz,
    }

    # 返回每个平面的指标
    return results


def switchReconRange(phantom_type, Calibration="5", range=None):
    if Calibration == "5":
        if phantom_type == "Shape":
            recon_plot_range_x = [12, 28]
            recon_plot_range_y = [14, 21]
            recon_plot_range_z = [8, 25]
        elif phantom_type == "Resolution":
            recon_plot_range_x = [13, 27]
            recon_plot_range_y = [10, 29]
            recon_plot_range_z = [16, 19]
        elif phantom_type == "Concentration":
            recon_plot_range_x = [13, 26]
            recon_plot_range_y = [9, 14]
            recon_plot_range_z = [22, 25]
    if Calibration == "6":
        if phantom_type == "Shape":
            recon_plot_range_x = [12, 23]
            recon_plot_range_y = [15, 21]
            recon_plot_range_z = [11, 26]
        elif phantom_type == "Resolution":
            recon_plot_range_x = [15, 23]
            recon_plot_range_y = [11, 30]
            recon_plot_range_z = [15, 19]
        elif phantom_type == "Concentration":
            recon_plot_range_x = [22, 26]
            recon_plot_range_y = [9, 14]
            recon_plot_range_z = [22, 25]
    if Calibration == "Custom":
        recon_plot_range_x = [0, 0]
        recon_plot_range_y = [0, -1]
        recon_plot_range_z = [0, -1]
    return recon_plot_range_x, recon_plot_range_y, recon_plot_range_z
