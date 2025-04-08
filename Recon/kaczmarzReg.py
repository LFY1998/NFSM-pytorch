import math

import numpy as np


def computeRowEnergy(A):
    """
  Calculate the energy of the system matrix rows
  """
    M = A.shape[0]
    energy = np.zeros(M, dtype=np.double)

    for m in range(M):
        energy[m] = np.linalg.norm(A[m, :])
    return energy


def kaczmarzReg(A, b, shape, iterations=10, lambd=0, enforceReal=True, enforcePositive=True, shuffle=True,
                lambda_tv=0.1, tv_iters=1, SNR=None, power=0.3):
    M = A.shape[0]
    N = A.shape[1]

    x = np.zeros(N, dtype=b.dtype)
    residual = np.zeros(M, dtype=x.dtype)

    energy = computeRowEnergy(A)

    if SNR is None:
        rowIndexCycle = np.argsort(energy)[::-1]
    else:
        rowIndexCycle = np.argsort(SNR)[::-1]

    if shuffle:
        np.random.shuffle(rowIndexCycle)

    lambdIter = lambd

    for l in range(iterations):
        for m in range(M):
            k = rowIndexCycle[m]
            if energy[k] > 0:
                # denominator = (1.0 / energy[k]) + lambd
                # beta = (b[k] - A[k, :].dot(x) - np.sqrt(lambdIter) * residual[k]) / denominator
                # weights = sigmoid(torch.tensor(energy[k] / np.max(energy))+0.5)**0.5
                if SNR is not None:
                    weights = (SNR[k] / np.max(SNR))**power
                else:
                    # weights = (energy[k] / np.max(energy)) ** power
                    weights = 1
                beta = weights*(b[k] - A[k, :].dot(x) - np.sqrt(lambdIter) * residual[k]) / (energy[k] ** 2 + lambd)

                x[:] += beta * A[k, :].conjugate()

                residual[k] += np.sqrt(lambdIter) * beta

        if enforceReal and np.iscomplexobj(x):
            x.imag = 0
        if enforcePositive:
            x = x * (x.real > 0)

        x = x.real

        # # 重塑x以进行TV去噪
        x_image = x.reshape(shape)
        if tv_iters > 0:
            for z in range(shape[2]):
                x_image[:, :, z] = TV(x_image[:, :, z], tv_iters, 0.1, 1, np.zeros([shape[0], shape[1]]))
        x = x_image.flatten().astype(np.complex64)  # Flatten back for next ART iteration



    return x


def TV(m_imgData, iter, dt, epsilon, lamb):
    NX = m_imgData.shape[0]
    NY = m_imgData.shape[1]
    ep2 = epsilon * epsilon
    I_t = m_imgData.astype(np.float64)
    I_tmp = m_imgData.astype(np.float64)
    for t in range(0, iter):
        for i in range(0, NY):  # 一次迭代
            for j in range(0, NX):
                iUp = i - 1
                iDown = i + 1
                jLeft = j - 1
                jRight = j + 1  # 边界处理
                if i == 0:
                    iUp = i
                if NY - 1 == i:
                    iDown = i
                if j == 0:
                    jLeft = j
                if NX - 1 == j:
                    jRight = j
                tmp_x = (I_t[i][jRight] - I_t[i][jLeft]) / 2.0
                tmp_y = (I_t[iDown][j] - I_t[iUp][j]) / 2.0
                tmp_xx = I_t[i][jRight] + I_t[i][jLeft] - 2 * I_t[i][j]
                tmp_yy = I_t[iDown][j] + I_t[iUp][j] - 2 * I_t[i][j]
                tmp_xy = (I_t[iDown][jRight] + I_t[iUp][jLeft] - I_t[iUp][jRight] - I_t[iDown][jLeft]) / 4.0
                tmp_num = tmp_yy * (tmp_x * tmp_x + ep2) + tmp_xx * (tmp_y * tmp_y + ep2) - 2 * tmp_x * tmp_y * tmp_xy
                tmp_den = math.pow(tmp_x * tmp_x + tmp_y * tmp_y + ep2, 1.5)
                I_tmp[i][j] += dt * (tmp_num / tmp_den + (0.5 + lamb[i][j]) * (m_imgData[i][j] - I_t[i][j]))

        for i in range(0, NY):
            for j in range(0, NX):
                I_t[i][j] = I_tmp[i][j]  # 迭代结束

    return I_t  # 返回去噪图
