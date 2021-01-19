import time
from genetic_sort.genetic.population import Individual, Population
from typing import List

from utils.evolve import DataManager
from utils.general import Log, GPUManager, GPUTool
from utils.network_builder import NetworkBuilder
import importlib
import os
from multiprocessing import Process


class Evaluator(object):
    # some params about evaluation (in seconds)
    GPU_QUERY_CYCLE = 20  # interval of checking gpu status when all GPUs are busy
    EVAL_CHECK_CYCLE = 20  # interval of checking whether this population gets fully evaluated
    # GPU_QUERY_CYCLE = 1
    # EVAL_CHECK_CYCLE = 1

    def __init__(self, population: Population):
        self.pops = population
        # self.gpu_manager = GPUManager()
        pop_name = '%02d' % self.pops.gen_no
        self.eval_result = './populations/after_%s.txt' % pop_name
        self.max_epoch: int = self.pops.params['network']['epoch']
        self.batch_size: int = self.pops.params['general']['batch_size']
        self.num_workers: int = self.pops.params['general']['num_workers']
        if os.path.exists(self.eval_result):
            os.remove(self.eval_result)  # remove previous evaluation results

    def generate_scripts(self):
        for individual in self.pops.individuals:
            NetworkBuilder.generate_py_file(individual)

    def evaluate(self):
        # load cache first
        _cache = DataManager.load_cache_fitness()
        _hit_counter = 0
        for individual in self.pops.individuals:
            _key, _str = individual.uuid()
            if _key in _cache:
                _hit_counter += 1
                _metric = float(_cache[_key])
                individual.metric = _metric
                Log.info('Hit individual-%s for metric %.5f' % (individual.id, _metric))

        Log.info('Cache status: hit %d individuals! ' % _hit_counter)
        evaluating_offsprings = False
        for individual in self.pops.individuals:
            filename = individual.id
            if individual.metric < 0:  # need evaluation
                evaluating_offsprings = True
                # gpu_list = self.gpu_manager.get_free_gpu('python')
                gpu_list = GPUTool.get_free_gpu(keyword='python')
                # abandon since it will introduce `pynvml.NVMLError: b'Unknown Error'`
                while len(gpu_list) == 0:  # loop until a GPU is set free
                    Log.warn('All GPUs are busy! Waiting for next round...')
                    time.sleep(self.GPU_QUERY_CYCLE)
                    # gpu_list = self.gpu_manager.get_free_gpu('python')
                    gpu_list = GPUTool.get_free_gpu(keyword='python')
                gpu_id = gpu_list[-1]

                module_name = 'scripts.%s' % filename
                _module = importlib.import_module('.', module_name)
                _class = getattr(_module, 'RunModel')
                class_obj = _class()
                # p = Process(target=class_obj.unittest,
                #             args=(gpu_id, self.eval_result, self.max_epoch, self.batch_size, self.num_workers))
                p = Process(target=class_obj.do_work,
                            args=(gpu_id, self.eval_result, self.max_epoch, self.batch_size, self.num_workers))
                p.start()
                time.sleep(self.GPU_QUERY_CYCLE)  # wait for loading model
            else:
                Log.info('Individual %s has been evaluated with fitness %.5f, skipping evaluation...' %
                         (filename, individual.metric))
                with open(self.eval_result, 'a+') as f:
                    f.write('%s=%.5f\n' % (filename, individual.metric))

        if evaluating_offsprings:
            all_finished = False
            Log.info('Checking evaluation process...')
            while not all_finished:
                try:
                    line_num = 0
                    with open(self.eval_result, 'r') as f:
                        for line in f.readlines():
                            if line.strip() != '':
                                line_num += 1
                    all_finished = line_num >= len(self.pops.individuals)
                    if not all_finished:
                        Log.info('Not finished, continue checking...')
                except FileNotFoundError as e:  # if pop_size <= GPU_NUM, after_xx.txt should not exist for a while
                    Log.info("No eval_result yet...")
                    all_finished = False
                time.sleep(self.EVAL_CHECK_CYCLE)

        if evaluating_offsprings:
            fitness_dict = {}
            with open(self.eval_result, 'r') as f:
                for line in f.readlines():
                    if len(line.strip()) > 0:
                        line = line.strip().split('=')
                        fitness_dict[line[0]] = float(line[1])
            for individual in self.pops.individuals:
                if individual.metric < 0:
                    assert individual.id in fitness_dict
                    print("%s: %f" % (individual.id, fitness_dict[individual.id]))
                    individual.metric = fitness_dict[individual.id]
        else:
            Log.warn('No offsprings got evaluated!')

        DataManager.save_cache_fitness(self.pops.individuals)
