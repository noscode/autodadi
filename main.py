#from window_for_genetic_algorithm import window_for_ga, window_from_file
from genetic_algorithm import GA
#from hill_climbing import HC, HC_multi
import dadi
import sys
from demographic_model import Demographic_model
#import hill_climbing
import pickle

def load_data_for_people():
    data = dadi.Spectrum.from_file('YRI.CEU.CHB.fs')
    data = data.marginalize([2])
    return data

def load_new_data():
    dd = dadi.Misc.make_data_dict("PSKOV.NOVGOROD.YAKUT.txt")
    data = dadi.Spectrum.from_data_dict(dd, pop_ids=['Pskov', 'Novgorod', 'Yakut'], projections=[44, 40, 42])
#    dadi.Plotting.plot_3d_comp_Poisson(data, data, vmin=1)
    data = data.marginalize([0])
    return data

def load_data_for_cheetah():
    dd = dadi.Misc.make_data_dict("cheetah.txt")
    data = dadi.Spectrum.from_data_dict(dd, pop_ids=['Tan', 'Nam'], projections=[6, 8])
    return data
    
def run_genetic_algorithm_for_people(number_of_generations = 100, size_of_population = 30, draw_pictures_dir=False, out_file=None):
    data = load_data_for_people()
    
    number_of_populations = 2
    total_time = 1000
    time_per_generation = 25
    
    theta = 0.37396
    ns = (20,20)
    pts = [40, 50, 60]

    if out_file is not None:
        out_file = open(out_file, 'w')
    
    output_file = open("2d_yri_ceu_models", 'w')
    for i in xrange(10):
        ga_instance = GA(number_of_generations, size_of_population)
        ga_instance.set_params_for_dadi_scene(data, theta, ns, pts)
        ga_instance.init_first_population_of_models(number_of_populations, total_time, time_per_generation)

        if draw_pictures_dir is not None:
            ga_instance.run(draw_pictures_dir + str(i) + "/")
        else:
            ga_instance.run()
        if out_file is not None:
            pickle.dump(ga_instance.models.data, out_file)

def run_genetic_algorithm_for_cheetah(size_of_population = 30, draw_pictures_dir=False, out_file=None):
    data = load_data_for_cheetah()
    
    number_of_populations = 2
    total_time = 22000
    time_per_generation = 3
    
    theta = 0.37396
    ns = (6,8)
    pts = [40, 50, 60]

    if out_file is not None:
        out_file = open(out_file, 'w')
    
    output_file = open("cheetah_models", 'w')
    for i in xrange(10):
        ga_instance = GA(size_of_population)
        ga_instance.set_params_for_dadi_scene(data, theta, ns, pts)
        ga_instance.init_first_population_of_models(number_of_populations, total_time, time_per_generation)

        if draw_pictures_dir is not None:
            ga_instance.run(draw_pictures_dir + str(i) + "/")
        else:
            ga_instance.run()
        if out_file is not None:
            pickle.dump(ga_instance.models.data, out_file)

#def run_hill_climbing_for_people(number_of_threads = 10, display=None, out_file=None):
#    data = load_data_for_people()
#    
#    number_of_populations = 2
#    total_time = 5000
#    time_per_generation = 25
#    
#    theta = 0.37396
#    ns = (20,20)
##    ns = (20,20,20)
##    ns = [44, 40, 42]
##    ns = [40, 42]
#
#    pts = [40, 50, 60]
#    
#    HC.set_params_for_dadi_scene(data, theta, ns, pts)
#    
#    hc_multi = HC_multi(number_of_threads, out_file)
#    hc_multi.init_first_models(number_of_populations, total_time, time_per_generation)
#    
#    
#    if draw_pictures_dir:
#        root = window_for_ga(hc_instance)
#        root.mainloop()
#    else:
#        hc_multi.run()
#
#
#def draw_window_from_file(input_file):
#    data = load_new_data()
#    
#    number_of_populations = 2
#    total_time = 5000
#    time_per_generation = 25
#    
#    theta = 0.37396
##    ns = (20,20)
##    ns = [44, 40, 42]
#    ns = [40, 45]
#    pts = [40, 50, 60]
#
#    root = window_from_file(input_file, data)
#    data = data.marginalize([2])
#    root.mainloop()
    
#load_new_data()
## usual run
if len(sys.argv) == 2 and sys.argv[1] == 'usual':
    run_genetic_algorithm_for_people(out_file="people_models", draw_pictures_dir='people_result_pictures_usual/')

#fast run
elif len(sys.argv) == 1 or sys.argv[1] == 'fast':
    run_genetic_algorithm_for_people(size_of_population = 10, out_file="people_models", draw_pictures_dir='people_result_pictures_fast/')

elif len(sys.argv) == 1 or sys.argv[1] == 'cheetah':
    run_genetic_algorithm_for_cheetah(size_of_population = 10, out_file="cheetah_models", draw_pictures_dir='cheetah_result_pictures/')


#elif len(sys.argv) == 2 and sys.argv[1] == 'from_file':
#    draw_window_from_file("people_models_test")
#
#elif len(sys.argv) == 1 or sys.argv[1] == 'hc':
#    run_hill_climbing_for_people(out_file="people_models", display=False)
#

#def save_picture(m, data):
#        import PIL
#        import io
#        import pylab
#        import PIL.Image
#        import os
#
#                
#        fig = pylab.figure(1, figsize=(6.5,5.5))
#        m.draw(show=False)
#        buf1 = io.BytesIO()
#        fig.savefig(buf1, format='png')
#        buf1.seek(0)
#        fig.clf()
#
#        if (Demographic_model.number_of_populations == 1):
#            dadi.Plotting.plot_1d_comp_Poisson(m.sfs, data, vmin=1,  show=False)
#        elif (Demographic_model.number_of_populations == 2):
#            dadi.Plotting.plot_2d_comp_Poisson(m.sfs, data, vmin=1, show=False)
#        elif (Demographic_model.number_of_populations == 3):
#            dadi.Plotting.plot_3d_comp_Poisson(m.sfs, data, vmin=1, show=False)
#        buf2 = io.BytesIO()
#        pylab.savefig(buf2, format='png')
#        buf2.seek(0)
#        pylab.close('all')
#                
#        img1 = PIL.Image.open(buf1)
#        img2 = PIL.Image.open(buf2)
#
#        weight = img1.size[0] + img2.size[0]
#        height = max(img1.size[1], img2.size[1])
#
#        new_img = PIL.Image.new('RGB', (weight, height))
#
#        x_offset = 0
#        new_img.paste(img1, (0,0))
#        new_img.paste(img2, (img1.size[0], 0))
#        
#        new_img.save('model.png')



#data = dadi.Spectrum.from_file('YRI.CEU.CHB.fs')
#data = data.marginalize([2])
#
#number_of_populations = 2
#total_time = 1000
#time_per_generation = 25
#
#theta = 0.37396
#ns = [20,20]
#import numpy as np
#pts = np.array([40, 50, 60])
#
#from demographic_model import Period, Split
#m = Demographic_model(number_of_populations=number_of_populations, total_time=total_time, time_per_generation=time_per_generation)
#m.add_period(
#    Period(
#        time=0.0,
#        sizes_of_populations=[7752.0],
#        is_first_period=True))
#m.add_period(
#    Period(
#        time=978.0,
#        sizes_of_populations=[81227.0],
#        exponential_growths=[0]))
#m.add_period(
#    Period(
#        time=573.0,
#        sizes_of_populations=[7645.0],
#        exponential_growths=[1]))
#
#m.add_period(
#    Split(
#        split_procent=0.95,
#        population_to_split=0,
#        sizes_of_populations_before_split=m.periods[-1].sizes_of_populations))
#m.add_period(
#    Period(
#        time=409.0,
#        sizes_of_populations=[11544.0, 1444.0],
#        exponential_growths=[1,1],
#        migration_rates=[[None, 43e-5],[81e-5, None]]))
#m.add_period(
#    Period(
#        time=338.0,
#        sizes_of_populations=[19158.0, 20177.0],
#        exponential_growths=[1,0],
#        migration_rates=None))
#
#print m.as_vector()
#print m.get_total_time()
#func_ex = dadi.Numerics.make_extrap_log_func(m.dadi_code)
#m.sfs = func_ex(theta, ns, pts)
#print dadi.Inference.ll(m.sfs, data)
#
#save_picture(m, data)
#
#import moments
#model = m.moments_code(theta, ns)
#print moments.Inference.ll(model, data)
#
#m.moments_code_to_file("bad_model_for_moments.py")
#
#print "ok"
#plot_mod = moments.ModelPlot.generate_model(m.moments_code, theta, ns)
#print "ok"
#moments.ModelPlot.plot_model(plot_mod, save_file='model.png', pop_labels=['YRI', 'CEU'], nref=11293,  gen_time=25 , gen_time_units="KY", reverse_timeline=True)
#
#print m.as_vector()
#m.draw()
