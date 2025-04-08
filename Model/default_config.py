import argparse
import json
import sys


def default_cfg(argv=None):
    parser = argparse.ArgumentParser('Default arguments for training or test')
    # section: train/test cfg
    parser.add_argument('--random_seed', type=int, default=19980523,
                        help='random seed for reproducibility')
    parser.add_argument('--save_freq', type=int, default=5000,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=75,
                        help='batch_size')
    parser.add_argument('--train_percent', type=float, default=1,
                        help='percent*batch_size is the number of data used for training in each iter')
    parser.add_argument('--test_percent', type=float, default=1,
                        help='percent*batch_size is the number of data used for testing in each iter')
    parser.add_argument('--max_epochs', type=int, default=20000,
                        help='number of training epochs')
    parser.add_argument("--init_lr", default=1e-3, type=float)
    parser.add_argument("--mode", type=str, default='train',
                        help='train/test')
    # run_name and model_name
    parser.add_argument('--run_name', type=str, default='Calibration-5',
                        help='each run name')
    parser.add_argument('--model_name', type=str, default='NFM',
                        help='model name')
    parser.add_argument('--NFMmode', type=str, default='Cheby-M-order',
                        help='the type of network (M-order/standard)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='GPU device id')
    parser.add_argument('--load_option_path', type=str, default=None,
                        help='json options for loading')
    parser.add_argument('--load_model_path', type=str, default=None,
                        help='model params for loading')
    parser.add_argument('--load_best_param', action='store_true',
                        help='whether to load best model')
    parser.set_defaults(load_best_param=False)
    parser.add_argument('--resume_epochs', type=int, default=0,
                        help='number of epochs of model params for resuming')
    parser.add_argument('--display_result', action='store_true', help='display the results')
    parser.set_defaults(display_result=False)  # 默认值为 False
    parser.add_argument('--test_result_data_save', type=bool, default=False,
                        help='save the data of test results')
    parser.add_argument('--metrics', nargs='+', type=str, default=['psnr', 'ssim'],
                        help='metrics for test')
    parser.add_argument('--result_save_path', type=str, default=None,
                        help='result save path')
    parser.add_argument('--pos_emb_dim', type=int, default=10,
                        help='calibration position embedding dimension')
    parser.add_argument('--hidden_layers_enc', nargs='+', type=int_list_or_none, default=[3, 16, 16, 32],
                        help='pos encoder hidden size')
    parser.add_argument('--hidden_layers_dec', nargs='+', type=int, default=[32, 32, 48, 2],
                        help='condition encoder hidden size')
    parser.add_argument('--M_order', type=int, default=16,
                        help='order of separable representation')
    parser.add_argument('--ChebyOrder', type=str, default="first",
                        help='order of Chebyshev polynomial')
    parser.add_argument('--degree', type=int, default=9,
                        help='degree of Chebyshev polynomial')
    parser.add_argument('--epoch_l2', type=int, default=None,
                        help='number of epochs when add l2 regularization')
    parser.add_argument('--l2_struct', type=float, default=1,
                        help='constrain of structural consistency')
    parser.add_argument('--parallel_method', type=str, default="jit",
                        help='parallel method of training stage (jit or stream)')
    parser.add_argument('--recon_plot_range_x', nargs='+', type=int, default=[7, 30],
                        help='recon plot range of yz slices')
    parser.add_argument('--recon_plot_range_y', nargs='+', type=int, default=[7, 30],
                        help='recon plot range of xz slices')
    parser.add_argument('--recon_plot_range_z', nargs='+', type=int, default=[7, 30],
                        help='recon plot range of xy slices')
    parser.add_argument('--default_recon_range', type=str2bool, default=True,
                        help='whether to use default reconstruction range')
    parser.add_argument('--UpsampleRate', type=int, default=1,
                        help='rate of upsampling in dense view reconstruction')
    parser.add_argument('--train_encoder', type=str2bool, default=True
                        )

    # section: dataset cfg
    parser.add_argument('--data_type', type=str, default="OpenMPI")
    parser.add_argument('--SM_path', type=str, default=r"C:\OpenMPIData\calibrations\5.mdf")
    parser.add_argument('--Calibration_num', type=str, default="5",
                        help='the number of the Calibration file')
    parser.add_argument('--Mea_path', nargs='+', type=str, default=["C:\\OpenMPIData\\Measurements\\shape\\2.mdf",
                                                                    "C:\\OpenMPIData\\Measurements\\resolution\\2.mdf",
                                                                    "C:\\OpenMPIData\\Measurements\\concentration\\2.mdf"])
    parser.add_argument('--num_workers', type=int, default=1,
                        help='dataloader num_workers')

    parser.add_argument('--SNR_Threshold', type=float, default=7,
                        help='SNR Threshold for freq filter')
    parser.add_argument('--sampling_method', type=str, default="random")
    parser.add_argument('--downsample_factor', nargs='+', type=int, default=[2, 2, 2],
                        help='downsample factor')
    parser.add_argument('--sparsity', type=float, default=0.125)
    parser.add_argument('--plot_plane', type=int, default=19)
    parser.add_argument('--Rx', nargs='+', type=str2bool, default=[True, True, False],
                        help='List of boolean values for Rx')
    parser.add_argument('--Rx_Recon', nargs='+', type=str2bool, default=[True, False, False],
                        help='List of boolean values for Rx for reconstruction')
    parser.add_argument('--phantom_type', nargs='+', type=str, default=["Shape", "Resolution", "Concentration"],
                        help='phantom type')

    parser.add_argument('--select_dec_idx', nargs='+', type=str, default=None,
                        help='selected decoders for training or testing')


    if argv is not None:
        opt = parser.parse_args(argv)
    else:
        argv = sys.argv[1:]
        opt = parser.parse_args(argv)

    # 注意：如果使用option.json覆盖，会导致命令行输入的参数被覆盖
    # 如果想让输入优先级大于option覆盖，需要获取命令行输入参数，在加载时排除这些参数
    args_input = [item[2:] for item in argv if "--" in item]
    if opt.load_option_path is not None:
        print("internal options are loading...")
        print("loading cfg except external options {}".format(args_input))
        load_option(opt, opt.load_option_path, args_input)
        print("internal and external options were loaded successfully!")
    if opt.hidden_layers_enc == [None]:
        opt.hidden_layers_enc = None
    return opt


# 字典传的是指针，会随函数内修改而变化
# new_cfg是新增修改的部分，old_cfg是需要修改值的
def cfg_load(new_cfg, old_cfg):
    for key in new_cfg.keys():
        if isinstance(new_cfg[key], dict):
            cfg_load(new_cfg[key], old_cfg[key])
        else:
            if key in old_cfg.keys():
                old_cfg[key] = new_cfg[key]
            else:
                print(f"no key names {key} in config\n")
                # sys.exit()


def load_option(opt, load_path, exception):
    f = open(load_path, 'r')
    opt_load = json.load(f)
    for key in exception:
        if key in opt_load.keys():
            del opt_load[key]
    cfg_load(opt_load, opt.__dict__)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def int_list_or_none(value):
    if value.lower() == 'none':  # 如果输入是 'none'（不区分大小写），返回 None
        return None
    try:
        # 尝试将输入值解析为逗号分隔的整数列表
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid value: {value}. Expected an integer or 'None'."
        )
