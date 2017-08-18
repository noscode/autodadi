import matplotlib
matplotlib.use('TKAgg')
import pylab
import dadi

from genetic_algorithm import GA
from demographic_model import Demographic_model

import Tkinter
from Tkinter import *
import PIL.Image
from PIL import ImageTk
import pickle
import io


class window_for_ga(Tkinter.Tk):
    def __init__(self, ga_instance):
        Tkinter.Tk.__init__(self)
        self.ga_instance = ga_instance
        
        self.info = Tkinter.Label(self, text="", font=("TkDefaultFont", 20))
        self.info.pack(side="top")

        self.cur_model_index = 0
        
        self.time = Tkinter.Label(self, font=("TkDefaultFont", 16))
        self.time.pack(side="top")
 
        self.model_str = Tkinter.Label(self)
        self.model_str.pack(side="top")

        self.image1 = Tkinter.Label(self, border=0)
        self.image1.pack(side="left")
        self.image2 = Tkinter.Label(self, border=0)
        self.image2.pack(side="left")

        self.update()

    def draw_pictures(self, m):
        model = m.sfs
            
        self.info.configure(text="Iteration " + str(self.ga_instance.cur_iteration) + ", Fitness function: " + str(m.fitness_func_value))

        self.model_str.configure(text="MODEL: " + str(self.ga_instance.best_model()))

        fig = pylab.figure(1, figsize=(6.5,5.5))
        m.draw(show=False)
        buf1 = io.BytesIO()
        fig.savefig(buf1, format='png')
        buf1.seek(0)
        fig.clf()

        if (Demographic_model.number_of_populations == 1):
            dadi.Plotting.plot_1d_comp_Poisson(model, self.ga_instance.data, vmin=1,  show=False)
        elif (Demographic_model.number_of_populations == 2):
            dadi.Plotting.plot_2d_comp_Poisson(model, self.ga_instance.data, vmin=1, show=False)
        elif (Demographic_model.number_of_populations == 3):
            dadi.Plotting.plot_3d_comp_Poisson(model, self.ga_instance.data, vmin=1, show=False)
        buf2 = io.BytesIO()
        pylab.savefig(buf2, format='png')
        buf2.seek(0)
        pylab.close('all')
                
        img1 = ImageTk.PhotoImage(PIL.Image.open(buf1))
        img2 = ImageTk.PhotoImage(PIL.Image.open(buf2))

        self.image1.configure(image=img1)
        self.image1.image = img1
        self.image2.configure(image=img2)
        self.image2.image = img2

            
        
    def update(self):
        if not self.ga_instance.is_stoped():
            self.ga_instance.run_one_iteration()
            m = self.ga_instance.best_model()
            self.draw_pictures(m)
            self.cur_model_index = 0
            self.time.configure(text="Mean time: " + str(self.ga_instance.mean_time()))
            self.after(100, self.update)
        else:
            self.next_but = Button(self, text="Next model", command=self.get_next_model)
            self.next_but.pack(side="bottom")

    def get_next_model(self):
        self.cur_model_index += 1
        if self.cur_model_index == len(self.ga_instance.models.data):
            self.cur_model_index = 0
        m = self.ga_instance.models.data[self.cur_model_index]
        self.draw_pictures(m)


class window_from_file(Tkinter.Tk):
    def __init__(self, input_file, data):
        Tkinter.Tk.__init__(self)
        self.input_file = open(input_file)
        self.iteration = 0
        self.data = data
        
        self.info = Tkinter.Label(self, text="", font=("TkDefaultFont", 20))
        self.info.pack(side="top")
         
        self.model_str = Tkinter.Label(self)
        self.model_str.pack(side="top")

        self.image1 = Tkinter.Label(self, border=0)
        self.image1.pack(side="left")
        self.image2 = Tkinter.Label(self, border=0)
        self.image2.pack(side="left")

        Demographic_model.total_time = pickle.load(self.input_file)
        Demographic_model.time_per_generation = pickle.load(self.input_file)
        print len(data.shape)
        Demographic_model.number_of_populations = len(data.shape)

        self.update()
        
    def update(self):
        try:
            m = pickle.load(self.input_file)
            model = m.sfs
            
            self.info.configure(text="Iteration " + str(self.iteration) + ", Fitness function: " + str(m.fitness_func_value))
            self.iteration += 1

            self.model_str.configure(text="MODEL: " + str(m))

            fig = pylab.figure(1, figsize=(6.5,5.5))
            m.draw(show=False)
            fig.savefig("tmp_model.png")
            fig.clf()

            if (Demographic_model.number_of_populations == 1):
                dadi.Plotting.plot_1d_comp_Poisson(model, self.data, show=False)
            elif (Demographic_model.number_of_populations == 2):
                dadi.Plotting.plot_2d_comp_Poisson(model, self.data, show=False)
            elif (Demographic_model.number_of_populations == 3):
                dadi.Plotting.plot_3d_comp_Poisson(model, self.data, show=False)
            pylab.savefig("tmp_dadi.png")
            pylab.close('all')
                    
            img1 = ImageTk.PhotoImage(PIL.Image.open("tmp_model.png"))
            img2 = ImageTk.PhotoImage(PIL.Image.open("tmp_dadi.png"))

            self.image1.configure(image=img1)
            self.image1.image = img1
            self.image2.configure(image=img2)
            self.image2.image = img2

            self.after(1, self.update)
        except (EOFError):
            pass


