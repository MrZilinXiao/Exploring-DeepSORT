"""
Genetic Evolve Strategy for Exploring:
1. Candidate Block List
    Normal Conv2D with the `SAME` padding strategy
    Residual Block
    Pooling Layer with maxpooling/meanpooling, 50/50 chances each.

2. Evaluation
    Each individual will be trained for 30 epochs on a MOT16 held-out Dataset
    CMC_top_1, mAP are both evaluation metrics

"""

import numpy as np
from utils.general import Log
from typing import Any, List, Union
import hashlib
import copy
import random


class Unit:
    def __init__(self, layer_id: int):
        self.id = layer_id  # layer_id in each individual


class ConvUnit(Unit):
    def __init__(self, layer_id, in_channels: int, out_channels: int, kernel_size: int = 3, stride=1):
        super(ConvUnit, self).__init__(layer_id)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride


class ResUnit(Unit):
    def __init__(self, layer_id, in_channels: int, out_channels: int):
        super(ResUnit, self).__init__(layer_id)
        self.in_channels = in_channels
        self.out_channels = out_channels


class PoolUnit(Unit):
    threshold = 0.5

    def __init__(self, layer_id, prob, kernel_size=2, stride=2):
        super(PoolUnit, self).__init__(layer_id)
        self.prob = prob
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_type = 'max' if self.prob < self.threshold else 'mean'


class LAYER_TYPE:
    RESIDUAL_LAYER = 1
    CONV_LAYER = 2
    POOLING_LAYER = 3
    mapping = {
        ConvUnit: CONV_LAYER,
        ResUnit: RESIDUAL_LAYER,
        PoolUnit: POOLING_LAYER
    }


class Individual:
    def __init__(self, params: dict = None, idx: str = None):
        self.metric = -1.0  # eval metric
        # for MOT16 tracking task, we use Multi-Object Tracking Accuracy(MOTA) as metric
        self.id = idx  # individual id `indi0001`
        self.num_id = 0  # to record the latest block_id
        if params is not None:
            self.min_conv, self.max_conv = params['network']['conv_limit']  # type: int, int
            self.min_pool, self.max_pool = params['network']['pool_limit']  # type: int, int
            self.min_res, self.max_res = params['network']['res_limit']  # type: int, int
            self.max_len: int = params['network']['max_length']
            self.input_channel: int = params['network']['input_channel']
            self.output_channel: List[int] = params['network']['output_channel']
            self.input_size: List[int] = params['network']['input_size']
            self.output_size: List[int] = params['network']['output_size']
            self.downsample_count_max = params['network']['downsample_max']

        self.num_res_downsample_count = 0
        self.units: List[Union[Unit, ResUnit, ConvUnit, PoolUnit]] = []
        self.real_output_size = [0, 0]

    def reset_metric(self):
        self.metric = -1.0

    def _adjust_successor(self, successor_list, target_channels):
        for i, unit in enumerate(successor_list):
            if isinstance(unit, ConvUnit):
                unit.in_channels = target_channels
                break
            elif isinstance(unit, ResUnit):
                unit.in_channels = unit.out_channels = target_channels
                self._adjust_successor(successor_list[i + 1:], target_channels)
                break
            elif isinstance(unit, PoolUnit):
                continue
            else:
                return NotImplemented

    def _count_individual_layer(self, individual_units: List[Union[ConvUnit, PoolUnit, ResUnit]]) -> int:
        """
        Count the times when feature map size shrinks to its quarter
        It happens when forward process encounters any of following:
        1. A pooling layer with default params
        2. A Residual block with in/out `not equal`
        :param individual_units:
        :return:
        """
        return int(np.sum(list(map(lambda x: LAYER_TYPE.mapping[type(x)] == LAYER_TYPE.POOLING_LAYER or
                                             (LAYER_TYPE.mapping[
                                                  type(
                                                      x)] == LAYER_TYPE.RESIDUAL_LAYER and x.in_channels != x.out_channels),
                                   individual_units))))

    def initialize(self):
        # individual initialization should follow some global rules:
        # 1. Here we allow size shrinking when going through residual block,
        # but limitation exists to control the final feature map
        num_conv = np.random.randint(self.min_conv, self.max_conv + 1)
        num_pool = min(self.downsample_count_max,
                       np.random.randint(self.min_pool, self.max_pool + 1))  # pool kernel_size fixed to 2
        if num_pool == self.downsample_count_max:
            Log.warn("Init: Pooling layers consume all down-sample count of this individual {}.".format(self.id))
        num_res = np.random.randint(self.min_res, self.max_res + 1)
        self.num_res_downsample_count = np.random.randint(0,
                                                          self.downsample_count_max - num_pool + 1)  # restriction on downsample count
        avail_pos_list = list(range(1, num_conv + num_res + 1))  # drop 0 since no individuals start with pooling layer
        np.random.shuffle(avail_pos_list)
        pool_pos = sorted(avail_pos_list[0:num_pool])  # [3, 4, 6]
        layer_list = []
        for conv_id in range(num_conv):
            layer_list.append(LAYER_TYPE.CONV_LAYER)
        for res_id in range(num_res):
            layer_list.append(LAYER_TYPE.RESIDUAL_LAYER)
        for pool_id in pool_pos:  # insert pooling layer
            layer_list.insert(pool_id, LAYER_TYPE.POOLING_LAYER)

        in_channels = self.input_channel
        for layer_id in layer_list:
            new_block = self.init_block(layer_id, in_channels=in_channels, out_channels=None)
            if layer_id == LAYER_TYPE.CONV_LAYER or layer_id == LAYER_TYPE.RESIDUAL_LAYER:
                in_channels = new_block.out_channels
            self.units.append(new_block)

        # need to change some out_channels to False
        res_idx = [i for i, unit in enumerate(self.units) if isinstance(unit, ResUnit)]
        for downsample_id in random.choices(res_idx, k=self.num_res_downsample_count):
            out_list = [out_ch for out_ch in self.output_channel if out_ch != self.units[downsample_id].in_channels]
            # in/out channels of a Residual block with `same_shape` False can't be the same
            self.units[downsample_id].out_channels = np.random.choice(out_list)
            self._adjust_successor(self.units[downsample_id + 1:], self.units[downsample_id].out_channels)

        downsample_cnt = self._count_individual_layer(self.units)
        self.real_output_size = list(map(lambda input_size: int(input_size / np.power(2, downsample_cnt))
                                         , self.input_size))

    def init_block(self, layer_id: int, in_channels: Union[None, int] = None,
                   out_channels: Union[None, int] = None, max_or_mean: Union[None, float] = None, num_id=None):
        if num_id is not None:
            _num_id = num_id
        else:
            _num_id = self.num_id
            self.num_id += 1
        if layer_id == LAYER_TYPE.CONV_LAYER or layer_id == LAYER_TYPE.RESIDUAL_LAYER:
            assert in_channels is not None
            if out_channels is None:
                out_channels = self.output_channel[np.random.randint(0, len(self.output_channel))]
            return ConvUnit(_num_id, in_channels, out_channels) \
                if layer_id == LAYER_TYPE.CONV_LAYER else ResUnit(_num_id, in_channels, in_channels)
            # ResUnit keep `same_shape` True when init
        elif layer_id == LAYER_TYPE.POOLING_LAYER:
            if max_or_mean is None:
                max_or_mean = np.random.rand()
            return PoolUnit(_num_id, max_or_mean)
        else:
            return NotImplemented

    def uuid(self) -> (str, str):
        """
        Returns detailed network definition and its hash224 key for cache usage.
        :return: hash_key, network_definition
        """
        _str = []
        for unit in self.units:
            unit_str = []
            if isinstance(unit, ConvUnit):
                unit_str.append('conv')
                unit_str.append('num: %d, in: %d, out: %d' % (unit.id, unit.in_channels, unit.out_channels))
            elif isinstance(unit, PoolUnit):
                unit_str.append('pool')
                unit_str.append('num: %d, type: %s' % (unit.id, unit.pool_type))
            elif isinstance(unit, ResUnit):
                unit_str.append('res')
                unit_str.append('num: %d, in: %d, out: %d' % (
                    unit.id, unit.in_channels, unit.out_channels))
            else:
                return NotImplemented
            _str.append("%s%s%s" % ('[', ','.join(unit_str), ']'))
        _final_str = '-'.join(_str)
        _final_str_utf8 = _final_str.encode('utf-8')
        _key = hashlib.sha224(_final_str_utf8).hexdigest()
        return _key, _final_str

    def __str__(self) -> str:
        _str = ['individual id: %d' % self.id, 'curr metric: %.3f' % self.metric]
        _, net_str = self.uuid()
        return '\n'.join(_str) + '\n' + net_str

    def __eq__(self, other):
        _, _this_str = self.uuid()
        _, other_str = other.uuid()
        return _this_str == other_str


class Population:
    def __init__(self, params: dict, gen_no: int):
        self.gen_no: int = gen_no  # generation number
        self.indi_id = 0
        self.pop_size = params['general']['pop_size']
        self.params = params
        self.individuals: List[Individual] = []

    def _add_individual(self, individual_name: str):
        self.indi_id += 1
        individual = Individual(self.params, individual_name)
        individual.initialize()
        self.individuals.append(individual)

    def initialize(self):
        for i in range(self.pop_size):
            individual_name = 'indi%02d%02d' % (self.gen_no, self.indi_id)
            self._add_individual(individual_name)

    def create_from_offspring(self, offsprings: List[Individual]) -> None:
        """
        Create population from a list of offspring individuals
        We copy each individual and rename it with new generation id
        :param offsprings: List[Individual]
        :return: None
        """
        for individual in offsprings:
            indi = copy.deepcopy(individual)
            indi.id = 'indi%02d%02d' % (self.gen_no, self.indi_id)
            self.indi_id += 1
            indi.num_id = len(indi.units)
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for individual in self.individuals:
            _str.append(str(individual))
            _str.append('-' * 100)
        return '\n'.join(_str)
