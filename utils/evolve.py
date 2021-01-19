import pickle
import random
from typing import List
import glob
import os

from genetic_sort.genetic.population import Population, Individual


class DataManager:
    POPULATION_DIR = './populations'

    @classmethod
    def init(cls):
        if not os.path.exists(cls.POPULATION_DIR):
            os.mkdir(cls.POPULATION_DIR)
        # else:
        #     txt_files = glob.glob(os.path.join(cls.POPULATION_DIR, '*.txt'))
        #     for txt in txt_files:
        #         os.remove(txt)

    @classmethod
    def _save_pickle(cls, object, path):
        with open(path, 'wb') as f:
            pickle.dump(object, f)

    @classmethod
    def save_pops_at(cls, pops: Population, stage: str):
        cls._save_pickle(pops, os.path.join(cls.POPULATION_DIR, '%s_%02d.bin' % (stage, pops.gen_no)))

    @classmethod
    def load_pickle(cls, stage: str = None, gen_no: int = None, rela_path=None):
        if rela_path:
            path = rela_path
        else:
            path = os.path.join(cls.POPULATION_DIR, '%s_%02d.bin' % (stage, gen_no))
        obj: Population
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        # TODO: load fitness when loading population
        return obj

    @classmethod
    def load_population_path(cls, stage: str) -> List[str]:
        return sorted(glob.glob(os.path.join(cls.POPULATION_DIR, '%s_*.bin' % stage)))

    @classmethod
    def load_cache_fitness(cls, file_name='cache.txt'):
        file_name = os.path.join(cls.POPULATION_DIR, file_name)
        _dict = {}
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split(';')
                    _dict[data[0]] = '%.5f' % (float(data[1]))

        return _dict

    @classmethod
    def save_cache_fitness(cls, individuals: List[Individual], file_name='cache.txt'):
        file_name = os.path.join(cls.POPULATION_DIR, file_name)
        _dict = cls.load_cache_fitness()
        for individual in individuals:
            _key, _str = individual.uuid()
            _metric = individual.metric
            if _key not in _dict:
                with open(file_name, 'a+') as f:
                    f.write('%s;%.5f;%s\n' % (_key, _metric, _str))
                _dict[_key] = _metric

class Selector:
    @classmethod
    def RouletteSelection(cls, wait_list: list, weights: List[float], times: int):
        return random.choices(range(len(wait_list)), weights=weights, k=times)
