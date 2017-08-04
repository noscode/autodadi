import random
import numpy as np
import dadi
from matplotlib import pyplot as plt

class Period:
    # sizes of populations are sizes after period
    def __init__(self, time, sizes_of_populations, exponential_growths=None, migration_rates=None, is_first_period=False, is_split_of_population=False):
        self.time = time
        self.sizes_of_populations = sizes_of_populations
        self.number_of_populations = len(sizes_of_populations)
        if exponential_growths is None:
            if not is_first_period and not is_split_of_population:
                self.exponential_growths = [0]*self.number_of_populations
            else:
                self.exponential_growths = None
        else:
            self.exponential_growths = exponential_growths
        self.migration_rates = migration_rates
        self.is_first_period = is_first_period
        self.is_split_of_population = is_split_of_population
        
    def populations(self):
        return xrange(self.number_of_populations)

    def mutate(self, mutation_rate):
        self.time += random.choice([0,1]) * random.choice([-1,1]) * np.random.uniform(0, mutation_rate) * self.time
        for p in self.populations():
            self.sizes_of_populations[p] += random.choice([0,1]) * random.choice([-1,1]) * np.random.uniform(0, mutation_rate) * self.sizes_of_populations[p]
            if not self.is_first_period:
                self.exponential_growths[p] = random.choice([1, 0])
            if self.migration_rates is not None:
                for p2 in self.populations():
                    if self.migration_rates[p][p2] == 0:
                        self.migration_rates[p][p2] = random.random() * 1e-3
                    elif self.migration_rates[p][p2] is not None:
                        self.migration_rates[p][p2] += random.choice([0,1]) * random.choice([-1,1]) * np.random.uniform(0, mutation_rate) * self.migration_rates[p][p2]
class Split(Period):
    def __init__(self, split_procent, population_to_split, sizes_of_populations_before_split):
        new_sizes_of_populations = sizes_of_populations_before_split[:population_to_split]
        new_sizes_of_populations.extend([split_procent* sizes_of_populations_before_split[0], (1-split_procent) * sizes_of_populations_before_split[0]])
        new_sizes_of_populations.extend(sizes_of_populations_before_split[population_to_split+1 :])
        Period.__init__(
                self,
                time=0,
                sizes_of_populations=new_sizes_of_populations,
                is_split_of_population=True)
        self.split_procent = split_procent
        self.population_to_split = population_to_split

    def mutate(self, mutation_rate):
        self.split_procent += random.choice([0,1]) * random.choice([-1,1]) * np.random.uniform(0, mutation_rate) * self.split_procent
        self.split_procent = min(self.split_procent, 0.9)
#            self.update(sum(self.sizes_of_populations[self.population_to_split:self.population_to_split+2]))

    def update(self, new_size_of_population_before_split):
        self.sizes_of_populations[self.population_to_split] = self.split_procent * new_size_of_population_before_split
        self.sizes_of_populations[self.population_to_split+1] = (1-self.split_procent) * new_size_of_population_before_split



class Demographic_model:
    total_time = None
    number_of_populations = None

    
    def __init__(self, number_of_populations, total_time, time_per_generation):
        Demographic_model.number_of_populations = number_of_populations
        Demographic_model.total_time = total_time
        Demographic_model.time_per_generation = time_per_generation
        self.number_of_periods = 0
        self.periods = []
        self.fitness_func_value = None
        self.sfs = None


    def add_period(self, period):
        self.number_of_periods += 1
        self.periods.append(period)

    def add_list_of_periods(self, list_of_periods):
        self.number_of_periods += len(list_of_periods)
        self.periods.extend(list_of_periods)

    def normalize_time(self):
        sum_T = 0.0
        for period in self.periods:
            sum_T += period.time
        for i in xrange(len(self.periods)):
            self.periods[i].time /= sum_T
            self.periods[i].time *= Demographic_model.total_time

    def mutate(self, mutation_rate, mutation_rate_for_period=0.2):
        if np.random.choice([True, False], p=[mutation_rate, 1-mutation_rate]):
            for index in xrange(self.number_of_periods):
                if self.periods[index].is_split_of_population:
                    self.periods[index].update(sum(self.periods[index-1].sizes_of_populations))
                if random.choice([True, False]):
                    self.periods[index].mutate(mutation_rate_for_period)
                    if self.periods[index].is_split_of_population:
                        self.periods[index].update(sum(self.periods[index-1].sizes_of_populations))
        self.normalize_time()
        self.fitness_func_value = None

    def init_simple_model(self, min_N, max_N):
        self.add_period(
                Period(
                    time=0,
                    sizes_of_populations=[float(random.randint(min_N, max_N))],
                    is_first_period=True))
        self.add_period(
                Period(
                    time=random.random(),
                    sizes_of_populations=[float(random.randint(min_N, max_N))],
                    exponential_growths=[random.choice([0,1])]))
        if Demographic_model.number_of_populations > 1:

            split_procent=random.random()
            self.add_period(
                    Split(
                        split_procent,
                        population_to_split=0,
                        sizes_of_populations_before_split=self.periods[-1].sizes_of_populations))
            self.add_period(
                    Period(
                        time=random.random(),
                        sizes_of_populations=[float(random.randint(min_N, max_N)), float(random.randint(min_N, max_N))],
                        exponential_growths=[random.choice([0,1]), random.choice([0,1])],
                        migration_rates=[[None, random.random()*1e-3],[random.random()*1e-3, None]]))
            self.add_period(
                    Period(
                        time=random.random(),
                        sizes_of_populations=[float(random.randint(min_N, max_N)), float(random.randint(min_N, max_N))],
                        exponential_growths=[random.choice([0,1]), random.choice([0,1])],
                        migration_rates=None))
            if Demographic_model.number_of_populations > 2:
                split_procent=random.random()
                self.add_period(
                        Split(
                            split_procent,
                            population_to_split=0,
                            sizes_of_populations_before_split=self.periods[-1].sizes_of_populations))
                self.add_period(
                        Period(
                            time=random.random(),
                            sizes_of_populations=[float(random.randint(min_N, max_N)), float(random.randint(min_N, max_N)), float(random.randint(min_N, max_N))],
                            exponential_growths=[random.choice([0,1]), random.choice([0,1]), random.choice([0,1])],
                            migration_rates=[
                                [None, random.random()*1e-3, random.random()*1e-3],
                                [random.random()*1e-3, None, random.random()*1e-3],
                                [random.random()*1e-3, random.random()*1e-3, None] ] ))
                self.add_period(
                        Period(
                            time=random.random(),
                            sizes_of_populations=[float(random.randint(min_N, max_N)), float(random.randint(min_N, max_N)), float(random.randint(min_N, max_N))],
                            exponential_growths=[random.choice([0,1]), random.choice([0,1]), random.choice([0,1])],
                            migration_rates=None))
        
        self.normalize_time()
    
    def dadi_code(self, theta1, ns, pts):
        xx = dadi.Numerics.default_grid(pts)
        for pos, period in enumerate(self.periods):
            if period.is_first_period:
                phi = dadi.PhiManip.phi_1D(xx, theta0=theta1, nu=period.sizes_of_populations[0])
            elif period.is_split_of_population:
                if period.number_of_populations == 2:
                    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
                else: #if period.number_of_populations == 3:
                    if period.population_to_split == 0:
                        phi = dadi.PhiManip.phi_2D_to_3D_split_1(xx, phi)
                    else: # if period.population_to_split == 1
                        phi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)
            else: # change population size
                growth_funcs = []
                for i in xrange(period.number_of_populations):
                    if period.exponential_growths[i]:
#                        print self.periods[pos-1].sizes_of_populations[i], period.sizes_of_populations[i]
                        growth_funcs.append(_expon_growth_func(self.periods[pos-1].sizes_of_populations[i], period.sizes_of_populations[i], period.time))
                    else:
                        growth_funcs.append(_linear_growth_func(self.periods[pos-1].sizes_of_populations[i], period.sizes_of_populations[i], period.time))
                if period.number_of_populations == 1:
                    phi = dadi.Integration.one_pop(phi, xx, nu=growth_funcs[0], T=period.time, theta0=theta1)
                elif period.number_of_populations == 2:
                    phi = dadi.Integration.two_pops(phi, xx,  T=period.time, nu1=growth_funcs[0], nu2=growth_funcs[1],
                            m12=0 if period.migration_rates == None else period.migration_rates[0][1],
                            m21=0 if period.migration_rates == None else period.migration_rates[1][0], theta0=theta1)
                else:
                    phi = dadi.Integration.three_pops(phi, xx,  T=period.time, nu1=growth_funcs[0], nu2=growth_funcs[1], nu3=growth_funcs[2],
                            m12=0 if period.migration_rates == None else period.migration_rates[0][1],
                            m13=0 if period.migration_rates == None else period.migration_rates[0][2],
                            m21=0 if period.migration_rates == None else period.migration_rates[1][0],
                            m23=0 if period.migration_rates == None else period.migration_rates[1][2], 
                            m31=0 if period.migration_rates == None else period.migration_rates[2][0],
                            m32=0 if period.migration_rates == None else period.migration_rates[2][1], theta0=theta1)
        sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*Demographic_model.number_of_populations)
        return sfs
    
    def dadi_code_to_file(self, filename):
        with open(filename, 'w') as output:
            output.write("import dadi\ndef generated_model(theta1, ns, pts):\n")

            output.write("\txx = dadi.Numerics.default_grid(pts)\n")
            for pos, period in enumerate(self.periods):
                if period.is_first_period:
                    output.write("\tphi = dadi.PhiManip.phi_1D(xx, theta0=theta1, nu=" + str(period.sizes_of_populations[0]) + ")\n")
                elif period.is_split_of_population:
                    if period.number_of_populations == 2:
                        output.write("\tphi = dadi.PhiManip.phi_1D_to_2D(xx, phi)\n")
                    else: #if period.number_of_populations == 3:
                        if period.population_to_split == 0:
                            output.write("\tphi = dadi.PhiManip.phi_2D_to_3D_split_1(xx, phi)\n")
                        else: # if period.population_to_split == 1
                            output.write("\tphi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)\n")
                else: # change population size
                    growth_funcs = []
                    for i in xrange(period.number_of_populations):
                        if period.exponential_growths[i]:
    #                        print self.periods[pos-1].sizes_of_populations[i], period.sizes_of_populations[i]
                            growth_funcs.append(_expon_growth_func_str(self.periods[pos-1].sizes_of_populations[i], period.sizes_of_populations[i], period.time))
                        else:
                            growth_funcs.append(_linear_growth_func_str(self.periods[pos-1].sizes_of_populations[i], period.sizes_of_populations[i], period.time))
                    if period.number_of_populations == 1:
                        output.write("\tphi = dadi.Integration.one_pop(phi, xx, nu=" + growth_funcs[0] + ", T=" + str(period.time) + ", theta0=theta1)\n")
                    elif period.number_of_populations == 2:
                        output.write("\tphi = dadi.Integration.two_pops(phi, xx,  T=" + str(period.time) + ", nu1=" + growth_funcs[0] + ", nu2=" + growth_funcs[1] + ", m12=" + ("0" if period.migration_rates == None else str(period.migration_rates[0][1])) + ", m21=" + ("0" if period.migration_rates == None else str(period.migration_rates[1][0])) + ", theta0=theta1)\n")
                    else:
                        output.write("\tphi = dadi.Integration.two_pops(phi, xx,  T=" + str(period.time) + ", nu1=" + growth_funcs[0] + ", nu2=" + growth_funcs[1] + ", nu3=" + growth_funcs[2] + ", m12=" + "0" if period.migration_rates == None else str(period.migration_rates[0][1]) + "m13=" + "0" if period.migration_rates == None else str(period.migration_rates[0][2]) + ", m21=" + "0" if period.migration_rates == None else str(period.migration_rates[1][0]) + "m23=" + "0" if period.migration_rates == None else str(period.migration_rates[1][2]) + "m31=" + "0" if period.migration_rates == None else str(period.migration_rates[2][0]) + "m32=" + "0" if period.migration_rates == None else str(period.migration_rates[2][1]) + ", theta0=theta1)\n")
            output.write("\tsfs = dadi.Spectrum.from_phi(phi, ns, [xx]*" + str(Demographic_model.number_of_populations) + ")\n\treturn sfs\n")
            

    def __str__(self):
        s = "\n"
        for period in self.periods:
            s +=  "Ns=" + str([int(x) for x in period.sizes_of_populations]) + ", T=" + str(int(period.time)) + ", m=" + str(period.migration_rates) + ", exp=" + str(period.exponential_growths) + "\n"
        return s

    def draw(self, show=True, fig = plt):    

        real_total_time = 2 * Demographic_model.total_time * Demographic_model.time_per_generation
        positions = [[- real_total_time,0]]
        axes = fig.axes()
        axes.set_xlim(positions[0])
        distances_for_split = []
        prev_split = None
        min_dist_split = 1000
        for period in self.periods:
            min_dist_split = max(min_dist_split, max(period.sizes_of_populations))
        max_sizes_per_pops = [[0]*self.periods[-1].number_of_populations]
        for i, period in enumerate(reversed(self.periods)): 
            for p in period.populations():
                max_sizes_per_pops[-1][p] = max(max_sizes_per_pops[-1][p], period.sizes_of_populations[p])
            if period.is_split_of_population:
                if len(distances_for_split) == 0:
                    distances_for_split.append(max_sizes_per_pops[-1][period.population_to_split] + min_dist_split + max_sizes_per_pops[-1][period.population_to_split+1])
                else:
                    distances_for_split.append(max_sizes_per_pops[-1][period.population_to_split] + min_dist_split + max_sizes_per_pops[-1][period.population_to_split+1])
                min_dist_split += (distances_for_split[-1] - max_sizes_per_pops[-1][period.population_to_split]) / 2
                max_sizes_per_pops.append([0]*self.periods[i-1].number_of_populations)

    #    distances_for_split.reverse()
    #    max_sizes_per_pops.reverse()
        for i in xrange(len(distances_for_split)):
            distances_for_split[i] -= max_sizes_per_pops[i][0]

        for i, period in enumerate(self.periods):
            if not period.is_first_period and not period.is_split_of_population:
                real_time_of_period = period.time / Demographic_model.total_time * real_total_time
                for population in period.populations():
                    fig.plot([positions[population][0], positions[population][0] + real_time_of_period], [positions[population][1], positions[population][1]], 'r-')
                    before = self.periods[i-1].sizes_of_populations[population]
                    after  = period.sizes_of_populations[population]
                    x = np.arange(positions[population][0], positions[population][0] + real_time_of_period, 0.2)
                    if period.exponential_growths[population] == 1:
                        fig.plot(x, [positions[population][1] + before * (after / before) ** ((t - positions[population][0]) / real_time_of_period) for t in x], 'b-')
                    else:
                        fig.plot(x, [positions[population][1] + before + (after - before) * ((t - positions[population][0]) / real_time_of_period) for t in x], 'b-')
                    positions[population] = [positions[population][0] + real_time_of_period,  positions[population][1]]
            elif period.is_split_of_population:
                len_of_line = distances_for_split[-1]
                distances_for_split = distances_for_split[:-1]
                positions.insert(period.population_to_split+1, [positions[period.population_to_split][0], positions[period.population_to_split][1] - len_of_line/2])
                positions[period.population_to_split] = [positions[period.population_to_split][0], positions[period.population_to_split][1] + len_of_line/2]
                fig.plot([positions[period.population_to_split][0], positions[period.population_to_split+1][0]], [positions[period.population_to_split][1], positions[period.population_to_split+1][1]], 'r-')

        if show:
            fig.show()


def _linear_growth_func(before, after, time):
    return ( lambda t: before + (after - before) * (t / time) )
def  _expon_growth_func(before, after, time):
    return ( lambda t: before * ((after / before) ** (t / time)) )

def _linear_growth_func_str(before, after, time):
    return "lambda t: " + str(before) + " + " + "(" + str(after) + " - " + str(before) +  ") * (t / " + str(time) +")"
def  _expon_growth_func_str(before, after, time):
    return "lambda t: " + str(before) + " * " + "(" + str(after) + " / " + str(before) +  ") ** (t / " + str(time) +")"


