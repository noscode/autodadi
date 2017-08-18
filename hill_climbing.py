from demographic_model import Demographic_model, Period, Split
import dadi
import time 
import copy
import pickle
import random
import numpy as np
import time
import progressbar

def fitness_function(model, data, theta, ns, pts):
    if model.fitness_func_value is not None:
        return model.fitness_func_value
    func_ex = dadi.Numerics.make_extrap_log_func(model.dadi_code)
    model.sfs = func_ex(theta, ns, pts)
    model.fitness_func_value = dadi.Inference.ll(model.sfs, data)
    model.ll_multinom = dadi.Inference.ll_multinom(model.sfs, data)
    return model.fitness_func_value

def upgrade_model(model, mutation_rate):
    epsilon = 1e-4
    new_model = copy.deepcopy(model)
#    print new_model
    index, sign = new_model.mutate_clever(mutation_rate)
    print new_model.as_vector()
#    print "WAS: ", model.__str__(end=" | ")
    if fitness_function(new_model, HC.data, HC.theta, HC.ns, HC.pts) >= (model.fitness_func_value):# - epsilon):
        new_new_model = copy.deepcopy(new_model)
        index, sign = new_new_model.mutate_clever(mutation_rate, (index, sign))
        print index, ("+" if sign==1 else "-")
        counter = 0
        bar = progressbar.ProgressBar(maxval=50).start()
        while (fitness_function(new_new_model, HC.data, HC.theta, HC.ns, HC.pts) > (new_model.fitness_func_value)) and counter < 50:
#            print index
#            print new_model.as_vector()
            new_model = new_new_model
            new_model.number_of_changes[index] -= 1
            new_new_model = copy.deepcopy(new_new_model)
            index, sign = new_new_model.mutate_clever(mutation_rate, (index, sign))
            counter += 1
            bar.update(counter)
        bar.finish()
#        print "GOT: ", model.__str__(end=" | ")
        print "changed", new_model.fitness_func_value, (new_model.ll_multinom,)
        return (True, new_model)

    index, sign = new_model.mutate_clever(mutation_rate, index_and_sign=(index, -sign))
    if fitness_function(new_model, HC.data, HC.theta, HC.ns, HC.pts) >= (model.fitness_func_value):# - epsilon):
        new_model.number_of_changes[index] -= 1
        new_new_model = copy.deepcopy(new_model)
        index, sign = new_new_model.mutate_clever(mutation_rate, (index, sign))

        while fitness_function(new_new_model, HC.data, HC.theta, HC.ns, HC.pts) > (new_model.fitness_func_value):
#            print index
#            print new_model
            new_model = new_new_model
            new_model.number_of_changes[index] -= 1
            new_new_model = copy.deepcopy(new_new_model)
            index, sign = new_new_model.mutate_clever(mutation_rate, (index, sign))
#        print "GOT: ", model.__str__(end=" | ")
        print "changed", new_model.fitness_func_value
        return (True, new_model)

    print "not changed", new_model.fitness_func_value
#    print "TRY: ", model.__str__(end=" | ")
    return (False, model)
        
class HC:

    def __init__(self, file_to_write_models=None):
        self.cur_iteration = 0
        self.work_time = 0 
        self.model = None
        if file_to_write_models is not None:
            self.output = open(file_to_write_models, 'w')
        else:
            self.output = None
        self.without_changes = 0 # for stop
        
    def init_first_model(self, number_of_populations, total_time, time_per_generation):
        self.model = Demographic_model(number_of_populations, total_time, time_per_generation)
        self.model.init_simple_model()
        
        fitness_function(self.model, HC.data, HC.theta, HC.ns, HC.pts)
        
#        print "[-1]", self.model.fitness_func_value
#        print self.model.as_vector()

        if self.output is not None:
            pickle.dump(total_time, self.output)
            pickle.dump(time_per_generation, self.output)
            pickle.dump(self.model, self.output)
            
    @staticmethod
    def set_params_for_dadi_scene(data, theta, ns, pts):
        HC.data = data
        HC.theta = theta
        HC.ns = ns
        HC.pts = pts

    def mean_time(self):
        return self.work_time / self.cur_iteration
        
    def run_one_iteration(self):
        start = time.time()
        
        is_changed, self.model = upgrade_model(self.model, mutation_rate=0.1)

#        print "[" + str(self.cur_iteration) + "]", self.model.fitness_func_value
#        print self.model.as_vector()
        stop = time.time()
        self.work_time += stop - start
        self.cur_iteration += 1
#        print "mean time: ", self.mean_time()
        
        if self.output is not None:
            pickle.dump(self.model, self.output)

        if not is_changed:
            self.without_changes += 1
        else:
            self.without_changes = 0
        
    def is_stoped(self):
#        return (self.cur_iteration > 10)
        return self.without_changes > 100

    def run(self):
        while (not self.is_stoped()):
            self.run_one_iteration()

class HC_multi:
    def __init__(self, number_of_threads, out_file=None):
        self.array_of_hc = []
        self.size = number_of_threads * 20
        self.number_of_threads = number_of_threads
        self.cur_iteration = 0
        for i in xrange(self.size):
            self.array_of_hc.append(HC(file_to_write_models=out_file))

    def init_first_models(self, number_of_populations, total_time, time_per_generation):
        for i in xrange(self.size):
            self.array_of_hc[i].init_first_model(number_of_populations, total_time, time_per_generation)
        self.select()

    def select(self):
        self.array_of_hc = sorted(self.array_of_hc, key=lambda x: fitness_function(x.model, HC.data, HC.theta, HC.ns, HC.pts), reverse=True)[:self.number_of_threads]
        self.size = self.number_of_threads

    def run(self):
        is_stoped = False
        time_of_work = 0
        while not is_stoped:
            start = time.time()
            is_stoped = True
            for i in xrange(self.number_of_threads):
                self.array_of_hc[i].run_one_iteration()
                is_stoped = is_stoped and (self.array_of_hc[i]).is_stoped()
            print "start_sort"
            self.select()
            stop = time.time()
            print self.cur_iteration, self.array_of_hc[0].model.fitness_func_value
            print self.array_of_hc[0].model.as_vector()
            time_of_work += stop - start
            self.cur_iteration += 1
            print "Mean time", time_of_work / self.cur_iteration
            print
        return self.array_of_hc



