# Autodadi
This is first working prototype of future autodadi.

This application implements several basic features. Now it includes testing genetic algorithm on inferring demographic history of two populations Eu and As from the [article](http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000695).

During run it prints current progress in console, moreover shows pictures in a window (I hope so).

# Requirements
```
python 2.X
```
Python modules:
```
dadi
pillow
Tkinter
matplotlib
```


# Run
To run tool for YRI, CEU populations, print:
```
python main.py 
```
or
```
python main.py usual
```
for version with 20 samples in genetic algorithm (in default (fast) version it is 10).

For cheetah (10 samples in genetic algorithm):
```
python main.py cheetah
```

Don't panic, *first iteration is longer than others*, because it creates a lot of models (like Monte Carlo approach).

You can find pictures in \[people/cheetah\]\_result\_pictures\_\[fast, usual\]/ directories

# Warnings
Some warnings may occur:
```
0.0227517207086 10728.7667646
WARNING:Inference:Model is masked in some entries where data is not.
WARNING:Inference:Number of affected entries is 1. Sum of data in those entries is 0.0227517:
```
and:
```
can't invoke "event" command:  application has been destroyed
    while executing
"event generate $w <<ThemeChanged>>"
    (procedure "ttk::ThemeChanged" line 6)
    invoked from within
"ttk::ThemeChanged"
```

It's okay, I know about this :)
