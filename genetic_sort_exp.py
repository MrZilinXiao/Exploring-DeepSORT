from genetic_sort.genetic.evolve import Evolve
from utils.general import ConfigManager, Log, sync_template
from utils.evolve import DataManager, Selector
from genetic_sort.genetic.population import Population
from genetic_sort.genetic.evaluate import Evaluator
import copy
import numpy as np


class ExploringEvolve(object):
    def __init__(self):
        self.population = None
        self.config_manager = ConfigManager('./config.yaml')
        DataManager.init()

    def init_population(self):
        self.config_manager.set_evolution(True)
        self.population = Population(self.config_manager.config_dict, 0)
        self.population.initialize()
        DataManager.save_pops_at(self.population, 'begin')

    def evaluate(self):
        evaluator = Evaluator(self.population)
        evaluator.generate_scripts()
        evaluator.evaluate()

    def evolve(self):
        evolve = Evolve(self.config_manager.config_dict,
                        self.population.individuals, gen_no=self.population.gen_no)
        offsprings = evolve.start()
        self.parental_pops: Population = copy.deepcopy(self.population)
        self.population.individuals = copy.deepcopy(offsprings)

    def selection(self, loaded=False):  # to avoid early mature
        if loaded:
            next_pop = Population(self.config_manager.config_dict, self.population.gen_no + 1)
            next_pop.create_from_offspring(self.population.individuals)
            self.population = next_pop
            return
        sel_list, metric_list = [], []
        sel_list.extend(self.population.individuals)
        metric_list.extend([individual.metric for individual in self.population.individuals])
        sel_list.extend(self.parental_pops.individuals)
        metric_list.extend([individual.metric for individual in self.parental_pops.individuals])

        # Log part, skip for now.

        max_metric_idx = np.argmax(metric_list)
        sel_idx_list = Selector.RouletteSelection(metric_list, metric_list,
                                                  times=self.config_manager.config_dict['general']['pop_size'])
        if max_metric_idx not in sel_idx_list:
            # if max_metric_idx not get chosen, replace the one with minimum metric
            sel_metric_list = [metric_list[i] for i in sel_idx_list]
            min_metric_idx = np.argmin(sel_metric_list)
            sel_idx_list[min_metric_idx] = max_metric_idx

        next_gen_individuals = [sel_list[i] for i in sel_idx_list]
        # initialize a Population instance
        next_pop = Population(self.config_manager.config_dict, self.population.gen_no + 1)
        next_pop.create_from_offspring(next_gen_individuals)
        self.population = next_pop
        DataManager.save_pops_at(self.population, 'begin')  # save when next gen gets inited

    def do_work(self, max_gen: int):
        Log.info('-' * 50)
        # load or init population
        if self.config_manager.is_running():
            try:
                Log.info('Evolution is running, continue with latest `begin` population...')
                path = DataManager.load_population_path('begin')[-1]
                # if path is None:
                #     raise IOError('No population found...Try to set `is_running` to False to begin from scratch!')
                self.population = DataManager.load_pickle(rela_path=path)
                # gen_no = int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
                gen_no = self.population.gen_no
                Log.info('Loaded latest population #%d bin: %s...' % (gen_no, path))
                # DataManager.load_fitness_txt(self.population, gen_no)
                if gen_no == 0:  # have no parents, interruption happens in eval#0 or eval#1
                    Log.info("Exploring Evolve Gen-0: Initial Evaluation!")
                    self.evaluate()  # gen_no = 0
                else:  # have parents, load it
                    self.parental_pops = DataManager.load_pickle(rela_path=DataManager.load_population_path('begin')[-2])
                    # DataManager.load_fitness_txt(self.parental_pops, gen_no - 1)
                    self.evaluate()
            except IndexError:
                raise IOError('No population found...Try to set `is_running` to False to begin from scratch!')

        else:
            gen_no = 0
            Log.info('Initializing population...')
            self.init_population()
            Log.info("Exploring Evolve Gen-0: Initial Evaluation!")
            self.evaluate()  # gen_no = 0

        for curr_gen in range(gen_no + 1, max_gen):
            Log.info("Exploring Evolve Gen-%d: Crossover & Mutation!" % curr_gen)
            self.evolve()
            if curr_gen == 1:
                self.selection(loaded=True)
            Log.info("Exploring Evolve Gen-%d: Fitness Evaluation!" % curr_gen)
            self.evaluate()
            Log.info("Exploring Evolve Gen-%d: Environment Selection!" % curr_gen)
            self.selection()
        self.config_manager.set_evolution(False)


if __name__ == '__main__':
    sync_template()
    evo = ExploringEvolve()
    evo.do_work(max_gen=10)
    # DONE: resume from latest population
