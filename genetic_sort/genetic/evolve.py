import copy
import random
from typing import List, Union

import numpy as np

from genetic_sort.genetic.population import Individual, ConvUnit, PoolUnit, ResUnit, LAYER_TYPE, Population
from utils.evolve import DataManager
from utils.general import Log


class Evolve(object):
    MUTATION_TYPE = {
        0: 'add res',
        1: 'add conv',
        2: 'add pooling',
        3: 'remove layer',
        4: 'alter params of layer'
    }

    def __init__(self, config_dict: dict, individuals: List[Individual], **kwargs):
        self.config = config_dict
        self.cross_prob = self.config['general']['genetic_prob'][0]
        self.mutation_prob = self.config['general']['genetic_prob'][1]
        self.detailed_mutation_prob = self.config['general']['mutation_probs']
        self.out_channels_list = self.config['network']['output_channel']
        assert np.isclose(np.sum(self.detailed_mutation_prob), 1)
        # add res, add conv, add pooling, remove layer , alter params of layer
        self.downsample_max = self.config['network']['downsample_max']
        self.individuals = individuals
        self.offsprings = []
        self.params = {k: v for k, v in kwargs.items()}  # some special params
        assert 'gen_no' in self.params  # have to include `gen_no` params

    def start(self):
        # crossover
        crossover_offsprings = self.crossover()
        self.offsprings = crossover_offsprings
        crossover_pops = Population(self.config, self.params['gen_no'])
        crossover_pops.create_from_offspring(self.offsprings)
        DataManager.save_pops_at(crossover_pops, 'crossover')

        # mutation
        self.mutation()
        mutation_pops = Population(self.config, self.params['gen_no'])
        mutation_pops.create_from_offspring(self.offsprings)

        for i, individual in enumerate(self.offsprings):  # update individual id
            individual.id = 'indi%02d%02d' % (self.params['gen_no'], i)
            downsample_factor = self._count_individual_layer(individual.units)
            individual.real_output_size = list(map(lambda size: size / np.power(2, downsample_factor),
                                                   individual.input_size))  # update real_output_size

        DataManager.save_pops_at(mutation_pops, 'mutation')
        return self.offsprings

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

    def _adjust_successor(self, successor_list, target_channels):
        """
        Adjust successor units to adapt to ResUnit `same_shape`
        :param successor_list:
        :param target_channels:
        :return:
        """
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

    def _adjust_channels(self, unit_list: List[Union[ConvUnit, PoolUnit, ResUnit]], split_pos, add_mutation=False):
        _log_str = '\nAdjust Channels:\n'
        latest_output_channels = self.config['network']['input_channel']
        for i in range(split_pos - 1, -1, -1):  # find the rightmost Conv/Res Unit of split1
            if isinstance(unit_list[i], ConvUnit) or isinstance(unit_list[i], ResUnit):
                latest_output_channels = unit_list[i].out_channels
                _log_str += 'Found the rightmost Conv/Res Unit({}) of split1, \n'.format(i)
                break

        for j in range(split_pos,
                       len(unit_list)):  # change the `in_channels` of the leftmost Conv/Res Unit of split2
            if isinstance(unit_list[j], ConvUnit) or isinstance(unit_list[j], ResUnit):
                unit_list[j].in_channels = latest_output_channels
                if add_mutation and isinstance(unit_list[j], ResUnit) and j == split_pos:  # adding a ResUnit
                    unit_list[j].out_channels = unit_list[j].in_channels
                    self._adjust_successor(unit_list[j + 1:], unit_list[j].out_channels)
                _log_str += 'Alter the in_channels of the leftmost Conv/Res Unit({}) of split2.\n'.format(j)
                if isinstance(unit_list[j], ResUnit) and self._count_individual_layer(unit_list) > self.downsample_max:
                    # if introducing new down-sample factor leading to exceeding limit
                    _log_str += 'Adjust successor units at {} to adapt ResUnit.\n'.format(j)
                    self._adjust_successor(unit_list[j:], unit_list[j].in_channels)
                break

        if add_mutation:  # if mutation, we have to alter out_channels of mutated unit to latest_input_channels
            latest_input_channels = -1
            for k in range(split_pos + 1, len(unit_list)):
                if isinstance(unit_list[k], ConvUnit) or isinstance(unit_list[k], ResUnit):
                    latest_input_channels = unit_list[k].in_channels
                    _log_str += 'Mutation: Found the leftmost Conv/Res Unit({}) of split2.\n'.format(k)
                    break
            if isinstance(unit_list[split_pos], ConvUnit) and latest_input_channels != -1:
                unit_list[split_pos].out_channels = latest_input_channels

        Log.info(_log_str)

    def _adjust_num_id(self, unit_list: List[Union[ConvUnit, PoolUnit, ResUnit]]):
        for i, unit in enumerate(unit_list):
            unit.id = i

    def crossover(self):
        """
        Crossover strategy for BetterSORT:
        Any offsprings from crossover stage
            1. should not start with a Pooling layer;
            2. should not possess more (Pooling Layer/Residual Layer with in/ont not equal) than config defines
            (change crossover positions if so)
        Also, there are some rules:
            1. if no new offspring generated compared with parents, we straightly move to mutation stage

        :return:
        """

        def _fixed_random_split(parent1: Individual, parent2: Individual):
            """
            This function makes sure that offsprings from crossover possess enough down-sample factor
            :param parent1:
            :param parent2:
            :return:
            """

            def _split_helper(split_pos1, split_pos2):
                p1_left, p1_right, p2_left, p2_right = self._count_individual_layer(parent1.units[:split_pos1]), \
                                                       self._count_individual_layer(parent1.units[split_pos1:]), \
                                                       self._count_individual_layer(parent2.units[:split_pos2]), \
                                                       self._count_individual_layer(parent2.units[split_pos2:])
                offspring1 = p1_left + p2_right
                offspring2 = p2_left + p1_right
                return split_pos1, split_pos2, offspring1, offspring2

            p1_len, p2_len = len(parent1.units), len(parent2.units)
            return [_split_helper(split_pos1, split_pos2) for split_pos1 in range(p1_len) for split_pos2 in
                    range(p2_len)]

        def _select_parent() -> int:
            random_idx = list(range(len(self.individuals)))
            np.random.shuffle(random_idx)
            lhs, rhs = random_idx[:2]
            return lhs if self.individuals[lhs].metric > \
                          self.individuals[rhs].metric else rhs

        def _choose_two_parents() -> (int, int):
            lhs, rhs = _select_parent(), _select_parent()
            while lhs == rhs:
                rhs = _select_parent()
            return lhs, rhs

        _evo_counter = {
            'evolved': 0,
            'from parent': 0
        }
        offspring_list = []
        for _ in range(len(self.individuals) // 2):  # crossover for half the number of individuals
            p1_idx, p2_idx = _choose_two_parents()
            parent1, parent2 = copy.deepcopy(self.individuals[p1_idx]), copy.deepcopy(self.individuals[p2_idx])
            if np.random.random() < self.cross_prob:  # begin crossover
                _evo_counter['evolved'] += 2  # increase counter, decrease it if no offspring after crossover
                pool_first1, pool_first2 = True, True  # Pool indicator of offsprings

                offspring1_units, offspring2_units = None, None
                choice_split = None
                recrossover_counter = 0
                try:
                    while pool_first1 or pool_first2:
                        # p1_pos, p2_pos, offspring1_cnt, offspring2_cnt = _random_split(parent1, parent2)
                        # while offspring1_cnt > self.downsample_max or offspring2_cnt > self.downsample_max:  # unstable method, may lead to infinite loop
                        split_list = _fixed_random_split(parent1, parent2)
                        filtered_split_list = [[split_pos1, split_pos2, offspring1, offspring2]
                                               for split_pos1, split_pos2, offspring1, offspring2 in split_list
                                               if offspring1 <= self.downsample_max and offspring2 <= self.downsample_max]
                        assert len(filtered_split_list) > 0, "no available split way when split {} and {}".format(
                            parent1.id, parent2.id)
                        # assertion may fail if no available split ways

                        choice_split = filtered_split_list[np.random.randint(0, len(filtered_split_list))][:2]
                        # random select a split plan out of available choices
                        Log.info(
                            "Crossover: Found a way {} to split {} and {}".format(choice_split, parent1.id, parent2.id))
                        offspring1_units, offspring2_units = [], []

                        offspring1_units.extend(parent1.units[:choice_split[0]])
                        offspring1_units.extend(parent2.units[choice_split[1]:])
                        offspring2_units.extend(parent2.units[:choice_split[1]])
                        offspring2_units.extend(parent1.units[choice_split[0]:])

                        pool_first1 = isinstance(offspring1_units[0], PoolUnit)
                        pool_first2 = isinstance(offspring2_units[0], PoolUnit)
                        if pool_first1 or pool_first2:
                            Log.warn('Crossover: Offsprings after crossover start with a pooling layer. Re-crossover...')
                            recrossover_counter += 1
                            if recrossover_counter > 10:
                                raise RuntimeError("Too many retries! Abandon this crossover...")

                    #  adjust conv/res/pool channels
                    self._adjust_channels(offspring1_units, choice_split[0])
                    self._adjust_channels(offspring2_units, choice_split[1])
                    self._adjust_num_id(offspring1_units)
                    self._adjust_num_id(offspring2_units)

                    parent1.units = offspring1_units  # okay since parent1, parent2 are deepcopy
                    parent2.units = offspring2_units
                    if parent1 == self.individuals[p1_idx] and parent2 == self.individuals[
                        p2_idx]:  # okay since we overwrite `__eq__` of Individual
                        pass  # TODO: submit to Mutation Process

                    parent1.reset_metric()
                    parent2.reset_metric()
                except RuntimeError as e:
                    pass
                finally:
                    offspring_list.append(parent1)
                    offspring_list.append(parent2)
            else:  # inherit same from parents
                _evo_counter['from parent'] += 2
                offspring_list.append(parent1)
                offspring_list.append(parent2)
        Log.info('Crossover: %d offspring(s) generated, where new: %d, inherited same: %d' %
                 (len(offspring_list), _evo_counter['evolved'], _evo_counter['from parent']))
        return offspring_list

    def mutation(self):
        """
        Mutation strategy for BetterSORT:
        1. Mutation plan was chose via random.choices() with weights
        2. No mutation occurs at individual.units[0]
        3. `_adjust_channels` is required after each mutation
        :return:
        """

        def _select_choice() -> str:
            return Evolve.MUTATION_TYPE[random.choices(
                list(range(len(self.detailed_mutation_prob))),
                weights=self.detailed_mutation_prob)[0]]

        def _add_layer(individual: Individual, mutation_type: str):
            mutation_pos = np.random.randint(0, len(individual.units))
            Log.info('Mutation: add layer mutation will occurs at %d for individual %s' % (mutation_pos, individual.id))
            type_mapping = {
                'add res': ResUnit,
                'add conv': ConvUnit,
                'add pooling': PoolUnit
            }
            unit_type = type_mapping[mutation_type]
            curr_downsample_cnt = self._count_individual_layer(individual.units)
            if unit_type == ConvUnit:
                add_unit = individual.init_block(LAYER_TYPE.mapping[ConvUnit], in_channels=1, out_channels=None,
                                                 num_id=mutation_pos)
                _mutation_counter['add conv'] += 1
                # `in_channels` not relevant since it will be overwritten by `_adjust_channels`

            elif unit_type == ResUnit:
                add_unit = individual.init_block(LAYER_TYPE.mapping[ResUnit], in_channels=1, out_channels=None,
                                                 num_id=mutation_pos)
                _mutation_counter['add res'] += 1
                # `in_channels` not relevant since it will be overwritten by `_adjust_channels`
            elif unit_type == PoolUnit:
                if curr_downsample_cnt == self.downsample_max:
                    Log.warn('Mutation: Mutation ceased due to no more down-sample chances...')
                    return
                if len(individual.units) == 1:
                    Log.warn('Mutation: Mutation ceased since too few units remaining...')
                    return
                add_unit = individual.init_block(LAYER_TYPE.mapping[PoolUnit], num_id=mutation_pos)
                _mutation_counter['add pooling'] += 1
            else:
                return NotImplemented

            individual.units.insert(mutation_pos, add_unit)
            self._adjust_channels(individual.units, mutation_pos, add_mutation=True)
            self._adjust_num_id(individual.units)
            individual.num_id += 1
            individual.reset_metric()

        def _add_res(individual: Individual):
            _add_layer(individual, 'add res')

        def _add_conv(individual: Individual):
            _add_layer(individual, 'add conv')

        def _add_pooling(individual: Individual):
            _add_layer(individual, 'add pooling')

        def _remove_layer(individual: Individual):
            if len(individual.units) <= 1:
                Log.warn("Mutation: Few units detected! Remove operation is not possible, skipping...")
                return
            remove_pos = np.random.randint(1, len(individual.units))
            Log.info(
                'Mutation: removing layer mutation will occurs at %d for individual %s' % (remove_pos, individual.id))
            individual.units.pop(remove_pos)
            self._adjust_channels(individual.units, remove_pos)
            self._adjust_num_id(individual.units)
            individual.num_id -= 1
            individual.reset_metric()
            _mutation_counter['remove layer'] += 1

        def _alter_params(individual: Individual):
            """
            Related to 2 operations with 50/50 chances:
            1. change conv/res in/out channels
            2. change type of pooling
            :param individual:
            :return:
            """
            _alter_type = 'channels' if np.random.random() < 0.5 else 'pooling_type'
            if _alter_type == 'channels':
                conv_idx_list = [i for i, unit in enumerate(individual.units)
                                 if isinstance(unit, ConvUnit) or isinstance(unit, ResUnit)]
                if len(conv_idx_list) == 0:
                    Log.warn(
                        'Mutation: No Conv/Res blocks in Individual {}, alteration is not possible, skipping...'.format(
                            individual.id))
                    return

                # change in_channels
                sel_conv_idx = random.choice(list(range(len(conv_idx_list))))
                sel_channel_idx = np.random.randint(0, len(self.out_channels_list))
                if individual.units[sel_conv_idx].in_channels != self.out_channels_list[sel_channel_idx]:
                    individual.reset_metric()
                    Log.info("Mutation: Individual %s will change its input channel at %d from %d to %d..." %
                             (individual.id, sel_conv_idx, individual.units[sel_conv_idx].in_channels,
                              self.out_channels_list[sel_channel_idx]))
                    individual.units[sel_conv_idx].in_channels = self.out_channels_list[sel_channel_idx]
                    if conv_idx_list[sel_conv_idx] > 0:
                        individual.units[sel_conv_idx - 1].out_channels = self.out_channels_list[sel_channel_idx]

                # change out_channels
                sel_channel_idx = np.random.randint(0, len(self.out_channels_list))
                if individual.units[sel_conv_idx].out_channels != self.out_channels_list[sel_channel_idx]:
                    individual.reset_metric()
                    Log.info("Mutation: Individual %s will change its output channel at %d from %d to %d..." %
                             (individual.id, sel_conv_idx, individual.units[sel_conv_idx].in_channels,
                              self.out_channels_list[sel_channel_idx]))
                    individual.units[sel_conv_idx].out_channels = self.out_channels_list[sel_channel_idx]
                    if conv_idx_list[sel_conv_idx] < len(individual.units) - 1:
                        individual.units[sel_conv_idx + 1].in_channels = self.out_channels_list[sel_channel_idx]
            elif _alter_type == 'pooling_type':
                pool_idx_list = [i for i, unit in enumerate(individual.units)
                                 if isinstance(unit, PoolUnit)]
                if len(pool_idx_list) == 0:
                    Log.warn(
                        'Mutation: No Pool blocks in Individual {}, alteration is not possible, skipping...'.format(
                            individual.id))
                    return
                sel_pool_idx = np.random.randint(0, len(pool_idx_list))
                sel_pool_idx = pool_idx_list[sel_pool_idx]
                if individual.units[sel_pool_idx].pool_type == 'max':
                    individual.units[sel_pool_idx].pool_type = 'mean'
                else:
                    individual.units[sel_pool_idx].pool_type = 'max'
                Log.info('Mutation: Individual %s will swap pool type at %d' % (individual.id, sel_pool_idx))
                individual.reset_metric()
            _mutation_counter['alter params of layer'] += 1

        _mutation_counter = {
            'new': 0,
            'from parent': 0,
            'add res': 0,
            'add conv': 0,
            'add pooling': 0,
            'remove layer': 0,
            'alter params of layer': 0
        }

        _mutation_handler = {
            'add res': lambda individual: _add_conv(individual),
            'add conv': lambda individual: _add_res(individual),
            'add pooling': lambda individual: _add_pooling(individual),
            'remove layer': lambda individual: _remove_layer(individual),
            'alter params of layer': lambda individual: _alter_params(individual),
        }

        for individual in self.offsprings:
            if np.random.random() < self.mutation_prob:
                _mutation_counter['new'] += 1
                choice_str = _select_choice()
                try:
                    _mutation_handler[choice_str](individual)  # call lambda
                except KeyError as e:
                    raise NotImplementedError from e
            else:
                _mutation_counter['from parent'] += 1
        Log.info("Mutation: %s" % (', '.join(['%s: %d' % (k, v) for k, v in _mutation_counter.items()])))


if __name__ == '__main__':
    pass
