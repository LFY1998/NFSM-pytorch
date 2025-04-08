import glob
import json

import torch
import os.path as osp
import os
import time

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import torch.distributed as dist
import inspect
from matplotlib import pyplot as plt
from datetime import datetime


def get_varname(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class LoggerX(object):
    def __init__(self, opt):
        self.current_save_dir = None
        timestamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        self.opt = opt
        if opt.result_save_path is None:
            self.save_root = osp.join(osp.dirname(osp.abspath(__file__)), '../ModelTrainLog/',
                                      '{}_{}_{}_sparsity-{}/{}'.format(opt.model_name, opt.NFMmode, opt.run_name, opt.sparsity, timestamp))
        else:
            self.save_root = osp.join(opt.result_save_path, '{}_{}_{}_sparsity-{}'.format(opt.model_name, opt.NFMmode, opt.run_name, opt.sparsity))
        self.models_save_dir = osp.join(self.save_root, 'save_models')
        self.curve_save_dir = osp.join(self.save_root, 'save_curve')
        os.makedirs(self.models_save_dir, exist_ok=True)
        self.modules = []
        self.module_names = []
        self.world_size = 1
        self.local_rank = 0
        self.curve_data = dict()
        if "train" in opt.mode:
            self.summer = SummaryWriter(log_dir=self.save_root + '/trainSummary')
        # Section:test result and metric save path
        # self.save_result_path = osp.join(self.save_root, 'save_test_results')
        # if not os.path.exists(self.save_result_path):
        #     os.makedirs(self.save_result_path)
        self.save_option(opt)

    # @property
    # def modules(self):
    #     return self._modules
    #
    # @property
    # def module_names(self):
    #     return self._module_names

    # @modules.setter
    # def modules(self, modules):
    #     for i in range(len(modules)):
    #         self._modules.append(modules[i])

    # @modules.setter
    # def modules_names(self,names):
    #     for i in range(len(names)):
    #         self._module_names.append(names[i])

    def checkpoints(self, epoch, save_path=None):
        if self.local_rank != 0:
            return
        if save_path is None:
            self.current_save_dir = osp.join(self.models_save_dir, 'Epoch-{}'.format(epoch))
        else:
            self.current_save_dir = save_path
        os.makedirs(self.current_save_dir, exist_ok=True)
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            if module is not None:
                torch.save(module.state_dict(), self.current_save_dir+"/{}.pth".format(module_name))

    def load_checkpoints(self, epoch, model_load_path):
        print("load model...")
        load_dir = osp.join(model_load_path, 'Epoch-{}'.format(epoch))
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            if module is not None:
                params_path = load_dir + "/{}.pth".format(module_name)
                if osp.exists(params_path):
                    module.load_state_dict(torch.load(params_path))
                    print("load {} finished!".format(module_name))

    def load_best_checkpoints(self, model_load_path):
        print("load model...")
        load_dir = osp.join(model_load_path, 'Best Epoch (minimum train loss)*')
        load_dir = glob.glob(load_dir)[0]
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            if module is not None:
                params_path = load_dir + "/{}.pth".format(module_name)
                if osp.exists(params_path):
                    module.load_state_dict(torch.load(params_path))
                    print("load {} finished!".format(module_name))

    def load_share_encoder(self, model_load_path):
        print("load encoder...")
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            if module is not None:
                params_path = model_load_path + "/{}.pth".format(module_name)
                if osp.exists(params_path):
                    checkpoint = torch.load(params_path, map_location=torch.device('cpu'))
                    share_encoder_state_dict = {k.replace('share_encoder.', ''): v for k, v in checkpoint.items() if
                                                k.startswith('share_encoder.')}
                    module.share_encoder.load_state_dict(share_encoder_state_dict)
                    print("load encoder finished!".format(module_name))

    def save_option(self, opt):
        info_json = json.dumps(opt.__dict__, sort_keys=False, indent=4, separators=(',', ': '))
        f = open(self.models_save_dir + '/option.json', 'w')
        f.write(info_json)
        f.close()
        for k in opt.__dict__:
            if k!="select_dec_idx":
                print(k + ": " + str(opt.__dict__[k]))

    def msg(self, stats, step, mode="train", epoch=0):
        output_str = 'Epoch:{}/{} {}_[{}] {:05d}, '.format(epoch, self.opt.max_epochs, mode,
                                                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)

        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)

        if self.local_rank == 0:
            print(output_str)

    def save_image(self, grid_img, n_iter, sample_type):
        save_image(grid_img, osp.join(self.images_save_dir,
                                      '{}_{}_{}.png'.format(n_iter, self.local_rank, sample_type)), nrow=1)

    def curve_print(self, data_name, data):
        if data_name in self.curve_data.keys():
            self.curve_data[data_name].append(data)
        else:
            self.curve_data[data_name] = [data]
        plt.plot(self.curve_data[data_name])
        plt.savefig(self.curve_save_dir + '/' + data_name + '.png')

    # @module_names.setter
    # def module_names(self, value):
    #     self._module_names = value
    #
    # @modules.setter
    # def modules(self, value):
    #     self._modules = value


def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`
        new_state_dict[namekey] = v
    return new_state_dict


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    if dist.is_initialized():
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    else:
        world_size = 1
    if world_size is not None:
        rt /= world_size
    return rt
