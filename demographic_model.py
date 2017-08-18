import random
import numpy as np
import dadi
from matplotlib import pyplot as plt
import copy

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

        if self.is_first_period or self.is_split_of_population:
            self.number_of_parameters = 1
        else:
            self.number_of_parameters = 1 + self.number_of_populations * 2 + (0 if self.migration_rates == None else (2 if self.number_of_populations == 2 else 6))
        self.number_of_changes = np.array([1] * self.number_of_parameters)
        
    def populations(self):
        return xrange(self.number_of_populations)

    def mutate_param(self, param_index, mutation_rate, sign):
        if self.is_split_of_population:
            self.mutate(mutation_rate)
        elif self.is_first_period:
            self.sizes_of_populations[0] *= 1 + sign * np.random.uniform(0, mutation_rate)
            self.sizes_of_populations[0] = max(self.sizes_of_populations[0], 1.)
        elif param_index == 0:
            self.time *= 1 + sign * np.random.uniform(0, mutation_rate)
            self.time = max(self.time, 2)
#            self.time = min(self, Period.max_time)
        elif param_index <= self.number_of_populations:
            self.sizes_of_populations[param_index - 1] *= 1 + sign * np.random.uniform(0, mutation_rate)
            self.sizes_of_populations[param_index - 1] = max(self.sizes_of_populations[param_index - 1], 1.)
        elif param_index <= 2 * self.number_of_populations:
            self.exponential_growths[param_index - self.number_of_populations - 1] = 0 if sign == -1 else 1 
        else:
            i = param_index - 2*self.number_of_populations - 1
            x = 0
            y = 0
            for x in xrange(self.number_of_populations):
                for y in xrange(self.number_of_populations):
                    if i == 0 and self.migration_rates[x][y] is not None:
                        break
                    if self.migration_rates[x][y] is not None:
                        i -= 1
                if i==0 and self.migration_rates[x][y] is not None:
                    break
            self.migration_rates[x][y] *= 1 + sign * np.random.uniform(0, mutation_rate)
            if self.migration_rates[x][y] < 1e-7:
                self.migration_rates[x][y] = 0.0
        self.number_of_changes[param_index] += 1

    def mutate(self, mutation_rate):
#        new_period = copy.deepcopy(self)
        self.time += random.choice([-1,1]) * np.random.uniform(0, mutation_rate) * self.time
        if (not self.is_first_period) and (not self.is_split_of_population):
            self.time = max(self.time, 1.)
        for p in self.populations():
            x = random.choice([-1,1]) * np.random.uniform(mutation_rate) * self.sizes_of_populations[p]
#            print x
            self.sizes_of_populations[p] += x
            self.sizes_of_populations[p] = max(self.sizes_of_populations[p], 1.)
            self.sizes_of_populations[p] = min(self.sizes_of_populations[p], 1000000.)
            if not self.is_first_period:
                self.exponential_growths[p] = random.choice([1, 0])
            if self.migration_rates is not None:
                for p2 in self.populations():
                    if self.migration_rates[p][p2] == 0:
                        self.migration_rates[p][p2] = random.random() * 1e-3
                    elif self.migration_rates[p][p2] is not None:
                        self.migration_rates[p][p2] += random.choice([-1,1]) * np.random.uniform(0, mutation_rate) * self.migration_rates[p][p2]
                        self.migration_rates[p][p2] = max(self.migration_rates[p][p2], 0)
#        return new_period

    def __str__(self):
        s = ""
        migr_str = "None" if (self.migration_rates is None) else ""
        if migr_str == "":
            migr_str += "["
            for y in self.migration_rates:
                migr_str += "["
                migr_str += str(y[0] if y[0] is None else "%.2f" % y[0])
                flag = False
                for x in y:
                    if flag:
                        migr_str += "," +  str(x if x is None else "%.2f" % x)
                    else:
                        flag = True
                migr_str += "]"
            migr_str += "]"
        s +=  "Ns=" + str([int(x) for x in self.sizes_of_populations]) + ", T=" + str(int(self.time)) 
        if not(self.is_first_period or self.is_split_of_population):
            s += ", m=" + migr_str + ", exp=" + str(self.exponential_growths)
        return s


    def as_vector(self):
        if self.is_first_period:
            return "[ " + str([int(self.sizes_of_populations[0])]) + " ]"
        if self.is_split_of_population:
            return '[ ' + str("%.2f" % self.split_procent) + ' ]'
        return '[ ' + str(int(self.time)) + ', ' + str([int(x) for x in self.sizes_of_populations]) + ', ' + str(self.exponential_growths) + ('' if self.migration_rates is None else (', ' + str(self.migration_rates))) + ' ]'


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
        self.split_procent = min(self.split_procent, 1 - 1./(self.sizes_of_populations[self.population_to_split] + self.sizes_of_populations[self.population_to_split+1]))
        self.split_procent = max(self.split_procent, 1/(self.sizes_of_populations[self.population_to_split] + self.sizes_of_populations[self.population_to_split+1]))


#            self.update(sum(self.sizes_of_populations[self.population_to_split:self.population_to_split+2]))

    def update(self, new_size_of_population_before_split):
        self.sizes_of_populations[self.population_to_split] = self.split_procent * new_size_of_population_before_split
        self.sizes_of_populations[self.population_to_split+1] = (1-self.split_procent) * new_size_of_population_before_split



class Demographic_model:
    total_time = None
    number_of_populations = None
    
    def __init__(self, number_of_populations=None, total_time=None, time_per_generation=None):
        if number_of_populations is not None:
            Demographic_model.number_of_populations = number_of_populations
        if total_time is not None:
            Demographic_model.total_time = total_time
        if  time_per_generation is not None:
            Demographic_model.time_per_generation = time_per_generation
        self.total_time = Demographic_model.total_time
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
            self.periods[i].time *= self.total_time
            if (not self.periods[i].is_first_period) or (not self.periods[i].is_split_of_population):
                self.periods[i].time = max(self.periods[i].time, 1)
        self.total_time = sum([x.time for x in self.periods])

    def get_total_time(self):
        self.total_time = sum([x.time for x in self.periods])
        return self.total_time

    def mutate(self, mutation_rate):
        index = random.randint(0, self.number_of_periods-1)
#        print index
        self.periods[index].mutate(mutation_rate)
        if self.periods[index].is_split_of_population:
            self.periods[index].update(sum(self.periods[index-1].sizes_of_populations))
        if index != (self.number_of_periods - 1) and self.periods[index+1].is_split_of_population:
            self.periods[index+1].update(sum(self.periods[index].sizes_of_populations))
        self.fitness_func_value = None
#        print self.__str__(end="|")
#        return self

    def mutate_all(self, mutation_rate):
        for index in xrange(self.number_of_periods):
            if random.choice([True, False]):
                self.periods[index].mutate(mutation_rate)
                if self.periods[index].is_split_of_population:
                    self.periods[index].update(sum(self.periods[index-1].sizes_of_populations))
                if index != (self.number_of_periods - 1):
                    if self.periods[index+1].is_split_of_population:
                        self.periods[index+1].update(sum(self.periods[index].sizes_of_populations))  
        self.fitness_func_value = None

    def mutate_clever(self, mutation_rate, index_and_sign=None):
        if not hasattr(self, "params_list"):
            self.params_list = []
            for i in xrange(self.number_of_periods):
                for j in xrange(self.periods[i].number_of_parameters):
                    self.params_list.append((i, j))
            self.number_of_changes = [1] * len(self.params_list)
            self.number_of_changes = np.array(self.number_of_changes, dtype=float)

        if index_and_sign is None:                
    #        self.number_of_changes -= min(self.number_of_changes) - 1
            p = max(self.number_of_changes) + 1 - self.number_of_changes
            p /= sum(p)
            i = np.random.choice(range(len(p)), p=p)
            sign = random.choice([-1, 1])
#            print self.number_of_changes
#            print p
#            print i
        else:
            i, sign = index_and_sign
            
        period_index, param_index = self.params_list[i]
        self.periods[period_index].mutate_param(param_index, mutation_rate, sign=sign)
        if period_index != 0:
            if self.periods[period_index].is_split_of_population:
                self.periods[period_index].update(sum(self.periods[period_index-1].sizes_of_populations))
        if period_index != (self.number_of_periods - 1):
            if self.periods[period_index+1].is_split_of_population:
                self.periods[period_index+1].update(sum(self.periods[period_index].sizes_of_populations))  
        self.number_of_changes[i] += 1
        
        self.fitness_func_value = None
        return i, sign

    def init_simple_model(self, randomized=False):
        min_N = 10
        max_N = random.choice([1000, 10000, 50000, 100000, 200000])#, 1000000])
        
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
        if random.choice([True, False]) or not randomized:
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
                    migration_rates=[[None, random.choice([0,1])* random.random()*1e-3],[random.choice([0,1])*random.random()*1e-3, None]]))
            if random.choice([True, False]) or not randomized:
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
                            population_to_split= 0 if not randomized else random.choice([0,1]),
                            sizes_of_populations_before_split=self.periods[-1].sizes_of_populations))
                self.add_period(
                        Period(
                            time=random.random(),
                            sizes_of_populations=[float(random.randint(min_N, max_N)), float(random.randint(min_N, max_N)), float(random.randint(min_N, max_N))],
                            exponential_growths=[random.choice([0,1]), random.choice([0,1]), random.choice([0,1])],
                            migration_rates=[
                                [None, random.choice([0,1])*random.random()*1e-3, random.choice([0,1])*random.random()*1e-3],
                                [random.choice([0,1])*random.random()*1e-3, None, random.choice([0,1])*random.random()*1e-3],
                                [random.choice([0,1])*random.random()*1e-3, random.choice([0,1])*random.random()*1e-3, None] ] ))
                if random.choice([True, False]) or not randomized:
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

    def moments_code(self, theta1, ns):
        import moments
        pop_to_split_3 = None
        if self.number_of_populations == 3:
            for period in reversed(self.periods):
                if period.is_split_of_population:
                    pop_to_split_3 = period.population_to_split
                    break
        print "ok"
        for pos, period in enumerate(self.periods):
            if period.is_first_period:
                sts = moments.LinearSystem_1D.steady_state_1D(sum(ns), N=period.sizes_of_populations[0], theta=theta1)
                print "fs = moments.LinearSystem_1D.steady_state_1D(sum(ns))"
                fs = moments.Spectrum(sts)
                print "fs = moments.Spectrum(fs)"
                cur_ns = [sum(ns)]
            elif period.is_split_of_population:
                if period.number_of_populations == 2:
                    if pop_to_split_3 is None or pop_to_split_3 == 1:
                        fs = moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))
                        cur_ns = [ns[0], sum(ns[1:])]
                    else:
                        fs = moments.Manips.split_1D_to_2D(fs, ns[0] + ns[2], ns[1])
                        cur_ns = [ns[0] + ns[2], ns[1]]
                else: 
                    if period.population_to_split == 0:
                        fs = moments.Manips.split_2D_to_3D_1(fs, ns[0], ns[2])
                    else:
                        fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])
                    cur_ns = ns
            else: 
                growth_funcs = []
                for i in xrange(period.number_of_populations):
                    if period.exponential_growths[i]:
#                        print self.periods[pos-1].sizes_of_populations[i], period.sizes_of_populations[i]
                        growth_funcs.append(_expon_growth_func(self.periods[pos-1].sizes_of_populations[i], period.sizes_of_populations[i], period.time))
                    else:
                        growth_funcs.append(_linear_growth_func(self.periods[pos-1].sizes_of_populations[i], period.sizes_of_populations[i], period.time))
                list_growth_funcs = lambda t: [ f(t) for f in growth_funcs]
                print "fs.integrate(" + str(np.array(list_growth_funcs(0))) + ", " + str(cur_ns) + ", " + str(period.time) + ", 0.05, m=" + str(period.migration_rates) + ", theta=" + str(theta1) + ")"
                if period.number_of_populations > 1 and period.migration_rates is not None:
                    m=np.array(period.migration_rates, dtype=float)
                    where_are_nans = np.isnan(m)
                    m[where_are_nans] = 0
                    fs.integrate(Npop=list_growth_funcs, tf=period.time, m=m, theta=theta1)
                else:
                    fs.integrate(Npop=list_growth_funcs, tf=period.time, theta=theta1)
        return fs

    def moments_code_to_file(self, filename):
        with open(filename, 'w') as output:
            output.write("import moments\nimport numpy\ndef generated_model(theta1, ns):\n")

            pop_to_split_3 = None
            if self.number_of_populations == 3:
                for period in reversed(self.periods):
                    if period.is_split_of_population:
                        pop_to_split_3 = period.population_to_split
                        break
            for pos, period in enumerate(self.periods):
                if period.is_first_period:
                    output.write("\tsts = moments.LinearSystem_1D.steady_state_1D(sum(ns), N=" + str(period.sizes_of_populations[0]) + ", theta=theta1)\n")
                    output.write("\tfs = moments.Spectrum(sts)\n\n")
                elif period.is_split_of_population:
                    if period.number_of_populations == 2:
                        if pop_to_split_3 is None or pop_to_split_3 == 1:
                            output.write("\tfs = moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))\n\n")
                        else:
                            output.write("\tfs = moments.Manips.split_1D_to_2D(fs, ns[0] + ns[2], ns[1])\n\n")
                    else: 
                        if period.population_to_split == 0:
                            output.write("\tfs = moments.Manips.split_2D_to_3D_1(fs, ns[0], ns[2])\n\n")
                        else:
                            output.write("\tfs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])\n\n")
                else: 
                    growth_funcs = "["
                    for i in xrange(period.number_of_populations):
                        if period.exponential_growths[i]:
                            growth_funcs += _expon_growth_func_str(self.periods[pos-1].sizes_of_populations[i], period.sizes_of_populations[i], period.time) + ", "
                        else:
                            growth_funcs += _linear_growth_func_str(self.periods[pos-1].sizes_of_populations[i], period.sizes_of_populations[i], period.time) + ', '
                    growth_funcs = growth_funcs[:-2]
                    growth_funcs += "]"
                    output.write("\tgrowth_funcs = " + growth_funcs + "\n")
                    output.write("\tlist_growth_funcs = lambda t: [ f(t) for f in growth_funcs]\n")
                    if period.number_of_populations > 1 and period.migration_rates is not None:
                        m=np.array(period.migration_rates, dtype=float)
                        where_are_nans = np.isnan(m)
                        m[where_are_nans] = 0
                        m = m.tolist()
                        output.write("\tm = numpy.array(" + str(m)+")\n")
                        output.write("\tfs.integrate(Npop=list_growth_funcs, tf="+ str(period.time) + ", m=m, theta=theta1)\n\n")
                    else:
                        output.write("\tfs.integrate(Npop=list_growth_funcs, tf=" + str(period.time) + ", theta=theta1)\n\n")
            output.write("\treturn fs")




    
    def dadi_code_to_file(self, filename):
        with open(filename, 'w') as output:
            output.write("import dadi\ndef generated_model(params, ns, pts):\n")
            cur_index = 0
            output.write("\ttheta1 = " + str(self.theta1) + '\n')
            output.write("\txx = dadi.Numerics.default_grid(pts)\n")
            for pos, period in enumerate(self.periods):
                if period.is_first_period:
                    output.write("\tphi = dadi.PhiManip.phi_1D(xx, theta0=theta1, nu=params[" + str(cur_index) + "])\n")
                    cur_index += 1
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
                    output.write("\tT=params[" + str(cur_index) + "]\n")
                    for i in xrange(period.number_of_populations):
                        if period.exponential_growths[i]:
    #                        print self.periods[pos-1].sizes_of_populations[i], period.sizes_of_populations[i]
                            growth_funcs.append(_expon_growth_func_str("params", "after", "T"))
                        else:
                            growth_funcs.append(_linear_growth_func_str("before", "after", "T"))
                    if period.number_of_populations == 1:
                        output.write("\tphi = dadi.Integration.one_pop(phi, xx, nu=" + growth_funcs[0] + ", T=" + str(period.time) + ", theta0=theta1)\n")
                    elif period.number_of_populations == 2:
                        output.write("\tphi = dadi.Integration.two_pops(phi, xx,  T=" + str(period.time) + ", nu1=" + growth_funcs[0] + ", nu2=" + growth_funcs[1] + ", m12=" + ("0" if period.migration_rates == None else str(period.migration_rates[0][1])) + ", m21=" + ("0" if period.migration_rates == None else str(period.migration_rates[1][0])) + ", theta0=theta1)\n")
                    else:
                        output.write("\tphi = dadi.Integration.two_pops(phi, xx,  T=" + str(period.time) + ", nu1=" + growth_funcs[0] + ", nu2=" + growth_funcs[1] + ", nu3=" + growth_funcs[2] + ", m12=" + "0" if period.migration_rates == None else str(period.migration_rates[0][1]) + "m13=" + "0" if period.migration_rates == None else str(period.migration_rates[0][2]) + ", m21=" + "0" if period.migration_rates == None else str(period.migration_rates[1][0]) + "m23=" + "0" if period.migration_rates == None else str(period.migration_rates[1][2]) + "m31=" + "0" if period.migration_rates == None else str(period.migration_rates[2][0]) + "m32=" + "0" if period.migration_rates == None else str(period.migration_rates[2][1]) + ", theta0=theta1)\n")
            output.write("\tsfs = dadi.Spectrum.from_phi(phi, ns, [xx]*" + str(Demographic_model.number_of_populations) + ")\n\treturn sfs\n")
            

    def __str__(self, end="\n"):
        s = end
        for period in self.periods:
            migr_str = "None" if (period.migration_rates is None) else ""
            if migr_str == "":
                migr_str += "["
                for y in period.migration_rates:
                    migr_str += "["
                    migr_str += str(y[0] if y[0] is None else "%.2f" % y[0])
                    flag = False
                    for x in y:
                        if flag:
                            migr_str += "," +  str(x if x is None else "%.2f" % x)
                        else:
                            flag = True
                    migr_str += "]"
                migr_str += "]"
            s +=  "T=" + str(int(period.time)) + ", Ns=" + str([int(x) for x in period.sizes_of_populations]) 
            if not(period.is_first_period or period.is_split_of_population):
                s += ", m=" + migr_str + ", exp=" + str(period.exponential_growths) 
            s += end
        return s

    def as_vector(self):
        s = ''
        for period in self.periods:
            s += period.as_vector()
        return s

    def draw(self, show=True):
        real_total_time = 2 * self.get_total_time() * Demographic_model.time_per_generation
        positions = [[- real_total_time,0]]
        axes = plt.axes()
        axes.set_xlim(positions[0])
        axes.get_yaxis().set_visible(False)
        distances_for_split = []
        prev_split = None
        min_dist_split = 1000
        for period in self.periods:
            min_dist_split = max(min_dist_split, max(period.sizes_of_populations))
        max_sizes_per_pops = [[0]*self.number_of_populations]
        for i, period in reversed(list(enumerate(self.periods))): 
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
                real_time_of_period = period.time / self.get_total_time() * real_total_time
                for population in period.populations():
                    plt.plot([positions[population][0], positions[population][0] + real_time_of_period], [positions[population][1], positions[population][1]], 'r-')
                    before = self.periods[i-1].sizes_of_populations[population]
                    after  = period.sizes_of_populations[population]
                    x = np.arange(positions[population][0], positions[population][0] + real_time_of_period, 0.2)
                    if period.exponential_growths[population] == 1:
                        plt.plot(x, [positions[population][1] + before * (after / before) ** ((t - positions[population][0]) / real_time_of_period) for t in x], 'b-')
                    else:
                        plt.plot(x, [positions[population][1] + before + (after - before) * ((t - positions[population][0]) / real_time_of_period) for t in x], 'b-')
                    positions[population] = [positions[population][0] + real_time_of_period,  positions[population][1]]
            elif period.is_split_of_population:
                len_of_line = distances_for_split[-1]
                distances_for_split = distances_for_split[:-1]
                positions.insert(period.population_to_split+1, [positions[period.population_to_split][0], positions[period.population_to_split][1] - len_of_line/2])
                positions[period.population_to_split] = [positions[period.population_to_split][0], positions[period.population_to_split][1] + len_of_line/2]
                plt.plot([positions[period.population_to_split][0], positions[period.population_to_split+1][0]], [positions[period.population_to_split][1], positions[period.population_to_split+1][1]], 'r-')

        if show:
            plt.show()


def _linear_growth_func(before, after, time):
    return ( lambda t: before + (after - before) * (t / time) )
def  _expon_growth_func(before, after, time):
    return ( lambda t: before * ((after / before) ** (t / time)) )

def _linear_growth_func_str(before, after, time):
    return "lambda t: " + str(before) + " + " + "(" + str(after) + " - " + str(before) +  ") * (t / " + str(time) +")"
def  _expon_growth_func_str(before, after, time):
    return "lambda t: " + str(before) + " * " + "(" + str(after) + " / " + str(before) +  ") ** (t / " + str(time) +")"


