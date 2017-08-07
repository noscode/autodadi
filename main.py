from window_for_genetic_algorithm import window_for_ga, window_from_file
from genetic_algorithm import GA
import dadi
import sys

def load_data_for_people():
    data = dadi.Spectrum.from_file('YRI.CEU.CHB.fs')
    data = data.marginalize([0])
    return data

def run_genetic_algorithm_for_people(number_of_generations = 100, size_of_population = 50, display=False, out_file=None):
    data = load_data_for_people()
    
    number_of_populations = 2
    total_time = 1000
    time_per_generation = 25
    min_N = 10
    max_N = 500000
    
    theta = 0.37396
    ns = (20,20)
    pts = [40, 50, 60]

    ga_instance = GA(number_of_generations, size_of_population, file_to_write_models=out_file)
    ga_instance.set_params_for_dadi_scene(data, theta, ns, pts)
    ga_instance.init_first_population_of_models(number_of_populations, total_time, time_per_generation, min_N, max_N)

    if display:
        root = window_for_ga(ga_instance)
        root.mainloop()
    else:
        ga_instance.run()

def draw_window_from_file(input_file):
    data = load_data_for_people()
    
    number_of_populations = 2
    total_time = 1000
    time_per_generation = 25
    min_N = 10
    max_N = 500000
    
    theta = 0.37396
    ns = (20,20)
    pts = [40, 50, 60]

    root = window_from_file(input_file, data)
    root.mainloop()
    

# usual run
if len(sys.argv) == 1 or sys.argv[1] == 'usual':
    run_genetic_algorithm_for_people(out_file="people_models", display=True)

#fast run
elif len(sys.argv) == 2 and sys.argv[1] == 'fast':
    run_genetic_algorithm_for_people(number_of_generations = 50, size_of_population = 4, out_file="people_models", display=True)

elif len(sys.argv) == 2 and sys.argv[1] == 'from_file':
    draw_window_from_file("people_models")

