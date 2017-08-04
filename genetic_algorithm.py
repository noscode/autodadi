from demographic_model import Demographic_model, Period, Split
import dadi
import time 
import copy
import pickle

def cross_two_models(model_1, model_2):
    position = random.randint(1, model_1.number_of_periods-1)
    child_1 = model(model.number_of_populations)
    child_2 = model(model.number_of_populations)
    child_1.add_list_of_periods(copy.deepcopy(model_1.periods[:position]))
    child_2.add_list_of_periods(copy.deepcopy(model_2.periods[:position]))
    child_1.add_list_of_periods(copy.deepcopy(model_2.periods[position:]))
    child_2.add_list_of_periods(copy.deepcopy(model_1.periods[position:]))
    if model_1.periods[position].is_split_of_population:
        child_1.periods[position].update(sum(child_1.periods[position-1].sizes_of_populations))
        child_2.periods[position].update(sum(child_2.periods[position-1].sizes_of_populations))
    child_1.normalize_time()
    child_2.normalize_time()
    child_1.fitness_func_value = None
    child_2.fitness_func_value = None

    return [child_1, child_2]

def fitness_function(model, data, theta, ns, pts):
    func_ex = dadi.Numerics.make_extrap_log_func(model.dadi_code)
    model.sfs = func_ex(theta, ns, pts)
    model.fitness_func_value = dadi.Inference.ll(model.sfs, data)
    return model.fitness_func_value


class GA:
    class population_of_models:
        def __init__(self, size, number_of_populations, total_time, time_per_generation, min_N, max_N):
            self.data = []
            for i in xrange(size):
                self.data.append(Demographic_model(number_of_populations, total_time, time_per_generation))
                self.data[-1].init_simple_model(min_N, max_N)
        def cross(self):
            size = len(self.data)
            for i in xrange(size/2):
                self.data.extend(cross_two_models(self.data[2*i], self.data[2*i+1]))
        def select(self, size):
            self.data = sorted(self.data, key=lambda x: fitness_function(x, GA.data, GA.theta, GA.ns, GA.pts), reverse=True)[:size]
        def mutate(self, mutation_rate):
            size = len(self.data)
            for i in xrange(size):
                self.data.append(copy.deepcopy(self.data[i]))
                self.data[i].mutate(mutation_rate)


    def __init__(self, num_of_generations = 100, s_of_population = 50, file_to_write_models=None):
        self.number_of_generations = num_of_generations
        self.size_of_population = s_of_population
        self.progress = []
        self.cur_iteration = 0
        self.work_time = 0 
        self.models = None
        if file_to_write_models is not None:
            self.output = open(file_to_write_models, 'w')
        else:
            self.output = None
        
    def init_first_population_of_models(self, number_of_populations, total_time, time_per_generation, min_N, max_N):
        self.models = self.population_of_models(self.size_of_population * 5, number_of_populations, total_time, time_per_generation, min_N, max_N)
        self.models.select(self.size_of_population)
        
        fitness_function(self.models.data[0], GA.data, GA.theta, GA.ns, GA.pts)
        
        print "[-1]", self.best_fitness_value(),  self.models.data[0]

        if self.output is not None:
            pickle.dump(self.best_model(), self.output)

    def set_params_for_dadi_scene(self, data, theta, ns, pts):
        GA.data = data
        GA.theta = theta
        GA.ns = ns
        GA.pts = pts

    def mean_time(self):
        return self.work_time / self.cur_iteration
    
    def best_model(self):
        return self.models.data[0]

    def best_fitness_value(self):
        return self.best_model().fitness_func_value
    
    def run_one_iteration(self):
        start = time.time()

        self.models.mutate(0.8)
        self.models.select(self.size_of_population)

        print "[" + str(self.cur_iteration) + "]", self.best_fitness_value(), self.models.data[0]
        stop = time.time()
        self.work_time += stop - start
        self.cur_iteration += 1
        print "mean time: ", self.mean_time()
        
        if self.output is not None:
            pickle.dump(self.best_model(), self.output)
        
    def is_stoped(self):
        return (self.cur_iteration > self.number_of_generations)

    def run(self):
        while (not self.is_stoped()):
            self.run_one_iteration()

