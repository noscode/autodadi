from demographic_model import Demographic_model, Period, Split
import dadi
import time 
import copy
import pickle
import random
import numpy as np

def cross_two_models(model_1, model_2):
    position = random.randint(1, model_1.number_of_periods-1)
    child_1 = Demographic_model()
    child_2 = Demographic_model()
    child_1.add_list_of_periods(copy.deepcopy(model_1.periods[:position]))
    child_2.add_list_of_periods(copy.deepcopy(model_2.periods[:position]))
    child_1.add_list_of_periods(copy.deepcopy(model_2.periods[position:]))
    child_2.add_list_of_periods(copy.deepcopy(model_1.periods[position:]))
    if model_1.periods[position].is_split_of_population:
        child_1.periods[position].update(sum(child_1.periods[position-1].sizes_of_populations))
        child_2.periods[position].update(sum(child_2.periods[position-1].sizes_of_populations))
    if position != (model_1.number_of_periods - 1):
        if model_1.periods[index+1].is_split_of_population:
            child_1.periods[index+1].update(sum(child_1.periods[index].sizes_of_populations))
            child_2.periods[index+1].update(sum(child_2.periods[index].sizes_of_populations))
    child_1.fitness_func_value = None
    child_2.fitness_func_value = None
    return [child_1, child_2]

def fitness_function(model, data, theta, ns, pts):
    if model.fitness_func_value is not None:
        return model.fitness_func_value
    func_ex = dadi.Numerics.make_extrap_log_func(model.dadi_code)
    model.sfs = func_ex(theta, ns, pts)
    model.fitness_func_value = dadi.Inference.ll(model.sfs, data)
#    model.ll_multinom = dadi.Inference.ll_multinom(model.sfs, data)
    return model.fitness_func_value


class GA:
    class population_of_models:
        def __init__(self, size, number_of_populations=None, total_time=None, time_per_generation=None, some_models = []):
            self.data = some_models
            self.size = size
            for i in xrange(size - len(some_models)):
                self.data.append(Demographic_model(number_of_populations, total_time, time_per_generation))
                self.data[-1].init_simple_model()

        def add_model(self, model=None):
            if model == None:
                self.data.append(Demographic_model())
                self.data[-1].init_simple_model()
            else:
                self.data.append(model)
            self.size += 1

    def select(self, size):
        self.models.data = sorted(self.models.data, key=lambda x: fitness_function(x, GA.data, GA.theta, GA.ns, GA.pts), reverse=True)[:size]
        print "CURRENT POPULATION OF MODELS:"
        for i, model in enumerate(self.models.data):
            print i, model.as_vector()




    def get_mutated_model(self, mutation_rate, p=None):
        model = np.random.choice(self.models.data, p=p)
        new_model_1 = copy.deepcopy(model)
        index, sign = new_model_1.mutate_clever(mutation_rate)
        new_model_2 = copy.deepcopy(model)
        new_model_2.mutate_clever(mutation_rate, index_and_sign=[index, -sign])
        if (fitness_function(new_model_1, GA.data, GA.theta, GA.ns, GA.pts) > fitness_function(new_model_2, GA.data, GA.theta, GA.ns, GA.pts)):
#            print index, new_model_1.as_vector()
            if model.fitness_func_value < new_model_1.fitness_func_value:
                new_model_1.number_of_changes[index] -= 1
            return new_model_1
        else:
#            print index, new_model_2.as_vector()
            if model.fitness_func_value < new_model_2.fitness_func_value:
                new_model_2.number_of_changes[index] -= 1
            return new_model_2


    def __init__(self, num_of_generations = 100, s_of_population = 20, procent_of_old_models=0.2, procent_of_mutated_models=0.3, procent_of_crossed_models=0.3, file_to_write_models=None):
        self.number_of_generations = num_of_generations
        self.size_of_population = s_of_population
        self.number_of_old_models = int(self.size_of_population*procent_of_old_models)
        self.number_of_mutated_models  = int(self.size_of_population * procent_of_mutated_models)
        self.number_of_crossed_models = (int(self.size_of_population * procent_of_crossed_models) / 2) * 2
        self.number_of_random_models = self.size_of_population - self.number_of_old_models - self.number_of_mutated_models - self.number_of_crossed_models
        self.progress = []
        self.cur_iteration = 0
        self.work_time = 0 
        self.models = None
        if file_to_write_models is not None:
            self.output = open(file_to_write_models, 'w')
        else:
            self.output = None

        self.best = None
        self.without_changes = 0 # for stop
        
    def init_first_population_of_models(self, number_of_populations, total_time, time_per_generation):
        self.models = self.population_of_models(max(250, self.size_of_population * 5), number_of_populations, total_time, time_per_generation)
        self.select(self.size_of_population)
        
        fitness_function(self.models.data[0], GA.data, GA.theta, GA.ns, GA.pts)
        
        print
        print "[-1]", self.best_fitness_value()
        print self.models.data[0].as_vector()

        if self.output is not None:
            pickle.dump(total_time, self.output)
            pickle.dump(time_per_generation, self.output)
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

        new_models = self.population_of_models(self.number_of_old_models, some_models=self.models.data[:self.number_of_old_models])

        p = []
        for m in self.models.data:
            p.append(m.fitness_func_value)
        p = np.array(p)
        p -= min(p)
        p /= sum(p)
#        print p

        if self.without_changes == 50:
            for i in xrange(self.number_of_mutated_models):
                model = self.get_mutated_model(0.1, p=p)
                new_models.add_model(model)
        else:
            for i in xrange(self.number_of_mutated_models):
                model = self.get_mutated_model(0.1, p=p)
                new_models.add_model(model)


        for i in xrange(self.number_of_crossed_models / 2):
            for x in cross_two_models(np.random.choice(self.models.data, p=p), np.random.choice(self.models.data, p=p)):
                new_models.add_model(x) 

        for i in xrange(self.number_of_random_models):
            new_models.add_model()

        self.models = new_models

        self.select(self.size_of_population)

        print
        print "[" + str(self.cur_iteration) + "]", self.best_fitness_value()
        print self.models.data[0].as_vector()
        
        stop = time.time()
        self.work_time += stop - start
        self.cur_iteration += 1
        print "mean time: ", self.mean_time()
        
        if self.output is not None:
            pickle.dump(self.best_model(), self.output)

        if self.best is None:
            self.best = self.best_fitness_value()
        if self.best_fitness_value() - self.best < 0.8:
            self.without_changes += 1
        else:
            self.best = self.best_fitness_value()
            self.without_changes = 0
        
    def is_stoped(self):
#        return (self.cur_iteration > self.number_of_generations)
        return self.without_changes > 50

    def run(self, draw_pictures_dir=None):
        while (not self.is_stoped()):
            self.run_one_iteration()
            if draw_pictures_dir is not None:
                self.save_picture_to_dir(draw_pictures_dir)


    def save_picture_to_dir(self, draw_pictures_dir):
        import PIL
        import io
        import pylab
        import PIL.Image
        import os

        
        m = self.best_model()
        
        fig = pylab.figure(1, figsize=(6.5,5.5))
        m.draw(show=False)
        buf1 = io.BytesIO()
        pylab.title("Iteration " + str(self.cur_iteration) + ", Fitness function: " + str(m.fitness_func_value) + "\nMean time: " + str(self.mean_time()))
        fig.savefig(buf1, format='png')
        buf1.seek(0)
        fig.clf()

        if (Demographic_model.number_of_populations == 1):
            dadi.Plotting.plot_1d_comp_Poisson(m.sfs, self.data, vmin=1,  show=False)
        elif (Demographic_model.number_of_populations == 2):
            dadi.Plotting.plot_2d_comp_Poisson(m.sfs, self.data, vmin=1, show=False)
        elif (Demographic_model.number_of_populations == 3):
            dadi.Plotting.plot_3d_comp_Poisson(m.sfs, self.data, vmin=1, show=False)
        buf2 = io.BytesIO()
        pylab.savefig(buf2, format='png')
        buf2.seek(0)
        pylab.close('all')
                
        img1 = PIL.Image.open(buf1)
        img2 = PIL.Image.open(buf2)

        weight = img1.size[0] + img2.size[0]
        height = max(img1.size[1], img2.size[1])

        new_img = PIL.Image.new('RGB', (weight, height))

        x_offset = 0
        new_img.paste(img1, (0,0))
        new_img.paste(img2, (img1.size[0], 0))

        if not os.path.exists(draw_pictures_dir):
            os.makedirs(draw_pictures_dir)
        new_img.save(draw_pictures_dir + 'model_' + str(self.cur_iteration) + '.png')

