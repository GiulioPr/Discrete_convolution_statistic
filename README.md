# Discrete convolution statistic
This repository contains the Python 3 implementation of the discrete convolution statistic
for hypothesis testing (by G. Prevedello and K.R. Duffy),
and the benchmarking simulations of this statistic against Pearson's chi-squared.

Given *k>1*, *h>0*, and random variables *X<sub>1</sub>, ..., X<sub>k</sub>,
Y<sub>1</sub>, ..., Y<sub>h</sub>*, each having discrete support *{0,...,a}* with possibly different integer *a>0*,
the aim of this statistical procedure is to test the null hypothesis of goodness-of-fit

**H<sub>0</sub>**: *X<sub>1</sub> + ... + X<sub>k</sub> ~ z*,

with *z* probability mass vector of positive entries,
    
and the null hypothesis of equality in distribution

**H<sub>0</sub>**: *X<sub>1</sub> + ... + X<sub>k</sub> ~ Y<sub>1</sub> + ... + Y<sub>h</sub>*.

The statistic of discrete convolution also enables to test the null hypothesis of sub-independence
for *X<sub>1</sub>, ..., X<sub>k</sub>*.

### Code
The function to calculate the discrete convolution statistic is found in code/discrete_convolution_statistics.py,
which is then called in code/Example.ipynb and code/Simulations_run.ipynb notebooks.

The script code/Example.ipynb serves as a minimal example for the application of discrete convolution statistic function.

Considering the hypotheses presented above, the script code/Simulations_run.ipynb
executes Monte Carlo simulations to estimate the proportion of rejections from different tests
under several parametrisations of the random variables, with *k=2*, *h=1*, and

*X<sub>1</sub> ~ x<sub>1</sub> = (1-p, p), X<sub>2</sub> ~ x<sub>2</sub> = (1-q, q)*,

*Y ~ z(r) = (1-r)((1-p)(1-q), 1-pq(1-p)(1-q), pq) + r(1-a, 0, a)*,

with *r* in *[0,1]* and *a = pq + (pq(1-p)(1-q))<sup>1/2</sup>*

The results from these simulations are then stored as csv files csv/Fig1.csv, csv/Fig2.csv and csv/Fig3.csv,
that are then loaded in code/Simulations_plot.ipynb to create the plots stored in the folder figures.
