from typing import List

import pynvml
import logging
import sys
import os
import datetime

import torch
import yaml
import numpy as np
import time
from subprocess import Popen, PIPE


def sync_template():
    """
    Sync deep_sort_template.py with a commented one `my_template.py`
    """
    with open('./template/deep_sort_template.py', 'r') as f:
        _lines = [line.rstrip() for line in f.readlines()]
        _lines.insert(0, '"""')
        _lines.insert(len(_lines), '"""')
    with open('./template/my_template.py', 'w') as f:
        f.write(os.linesep.join(_lines))


def Singleton(cls):
    """
    A decorator for Singleton support
    """
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


def choose_device(device: int = 0):
    """
    Display & return certain CUDA device
    :param device: starts with 'cpu' or digits(CUDA device id)
    :return: torch.device
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    # cuda part
    c = 1024 ** 2
    x = torch.cuda.get_device_properties(device)
    Log.info("Using _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
             (x.name, x.total_memory / c))
    return torch.device('cuda:%d' % device)


def select_device(device: str = ''):
    cpu = device.lower() == 'cpu'
    if device and not cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), 'CUDA Unavailable...'
    cuda = False if cpu else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2
        num_cuda = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(num_cuda)]
        s = 'Using CUDA '
        for i in range(0, num_cuda):
            if i == 1:
                s = ' ' * len(s)
            Log.info("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                     (s, i, x[i].name, x[i].total_memory / c))
    else:
        Log.info('Using CPU')
    return torch.device('cuda:0' if cuda else 'cpu')


class ConfigManager:
    """
    Rewrite StatusUpdateTool with yaml support
    """
    config_dict = {}
    config_path = ''
    layer = ['conv', 'res', 'pool']

    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError("{} not found!".format(config_path))
        ConfigManager.config_path = config_path
        self.read_config()

        # Calculate the maximum down-sample times
        _judge = lambda size: size[0] / size[1]  # type: size: list
        calc_down_cnt = lambda input_size, output_size: int(np.log2(input_size[0] / output_size[0]))

        self.input_size: List[int] = ConfigManager.config_dict['network']['input_size']
        self.output_size: List[int] = ConfigManager.config_dict['network']['output_size']

        if _judge(self.input_size) != _judge(self.output_size):  # if ratio not same
            raise ValueError("Input size {} should keep the same size ratio with output size {}!"
                             .format(str(self.input_size), str(self.output_size)))
        self.downsample_count_max = calc_down_cnt(self.input_size, self.output_size)
        if ConfigManager.config_dict['network']['pool_limit'][1] > self.downsample_count_max:
            Log.warn("Upper bound of pool_limit should not exceed `downsample_count_max`!")
            ConfigManager.config_dict['network']['pool_limit'][1] = self.downsample_count_max
        self.add_kv_and_save('downsample_max', self.downsample_count_max)

    @classmethod
    def read_config(cls):
        with open(cls.config_path, 'r') as f:
            cls.config_dict = yaml.load(f, Loader=yaml.FullLoader)

    @classmethod
    def write_config(cls):
        with open(cls.config_path, 'w') as f:
            f.write(yaml.dump(cls.config_dict))

    @classmethod
    def add_kv_and_save(cls, key, value):
        cls.config_dict['network'][key] = value
        cls.write_config()

    @classmethod
    def set_evolution(cls, status: bool):
        cls.config_dict['status']['is_running'] = status
        cls.write_config()

    @classmethod
    def is_running(cls):
        return cls.config_dict['status']['is_running']


class ExpLogger(logging.Logger):
    log_dir = './logs/'

    def __init__(self, name, level='DEBUG', file=True):
        super().__init__(name)
        self.setLevel(level)
        ft = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        if file:
            now = datetime.datetime.now()
            time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
            file_handler = logging.FileHandler(os.path.join(self.log_dir, "%s-%s.log" % (time_str, name)))
            file_handler.setFormatter(ft)
            self.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = ft
        self.addHandler(console_handler)
        self.propagate = False


class Log(object):
    _logger = None
    log_dir = './logs/'

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            if not os.path.exists(Log.log_dir):
                os.mkdir(Log.log_dir)
            logger = logging.getLogger("evolving-sort")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            now = datetime.datetime.now()
            time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
            file_handler = logging.FileHandler(os.path.join(Log.log_dir, "%s.log" % time_str))
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False  # disable child logger message to root logger
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)

    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warning(_str)


class GPUTool:
    @classmethod
    def get_free_gpu(cls, keyword: str = 'python', gpu_list=None) -> List[str]:
        p = Popen('nvidia-smi', stdout=PIPE)
        out = p.stdout.read().decode('UTF-8')
        lines = out.split(os.linesep)
        if gpu_list is not None:
            equipped_gpu_ids = gpu_list
        else:
            equipped_gpu_ids: List[str] = []
            for line in lines:
                if line.startswith(' '):
                    break
                if 'GeForce' in line:
                    equipped_gpu_ids.append(line.strip().split(' ', 4)[3])
        gpu_info_list = []
        for line_idx in range(len(lines) - 3, -1, -1):
            if lines[line_idx].startswith('|==='):
                break
            gpu_info_list.append(lines[line_idx][1:-1].strip())

        used_gpu_ids = []
        for gpu_info in gpu_info_list:
            if keyword in gpu_info:
                used_gpu_ids.append(gpu_info.strip().split(' ', 1)[0])
        free_gpu_ids: List[str] = [_id for _id in equipped_gpu_ids if _id not in used_gpu_ids]
        return free_gpu_ids


# !Abandoned! This manager brings unpredictable exceptions when training for a long period!
# Trying to use a Popen-based GPUTool...
@Singleton
class GPUManager:
    """
    PYNVML Wrapper
    More compatible GPUTools
    """

    def __init__(self):
        pynvml.nvmlInit()
        self.num_gpu = pynvml.nvmlDeviceGetCount()

    def get_free_gpu(self, keyword: str = 'python', gpu_list=None) -> List[int]:
        """
        :param keyword:
        :param gpu_list: List[int] candidate GPU List
        :return:
        """
        free_list = []
        if gpu_list is None:
            gpu_list = range(self.num_gpu)
        for gpu_id in gpu_list:
            handle = self._get_handle(gpu_id)
            free_flag = True
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                process_name = pynvml.nvmlSystemGetProcessName(process.pid)
                if keyword in str(process_name):
                    free_flag = False
                    break
            if free_flag:
                free_list.append(gpu_id)
        return free_list

    def _get_handle(self, gpu_id: int):
        assert gpu_id <= self.num_gpu - 1
        return pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

    @staticmethod
    def _get_gpu_memory_percentage(handle):
        """
        :param handle: Handle for nvidia-ml-py3
        :return: (float) memory ratio of the specific GPU
        """
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return meminfo.free / meminfo.total

    @staticmethod
    def _get_gpu_util_rate(handle):
        """
        :param handle: Handle for nvidia-ml-py3
        :return: (float) utilization ratio of the specific GPU
        """
        res = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return res.gpu


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000
        if self.verbose:
            print('elapsed time: %f ms' % self.msecs)


if __name__ == '__main__':
    print(GPUTool.get_free_gpu())