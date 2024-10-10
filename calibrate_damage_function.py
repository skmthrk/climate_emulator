import os
import sys
import csv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize
import scipy.linalg as la

import matplotlib.patheffects
from matplotlib import font_manager
path_effects = [matplotlib.patheffects.withStroke(linewidth=3.5, foreground="w")]

matplotlib.rc('axes', lw=0.25, edgecolor='k')

# calibrate damage function

data_dir = './data_raw/Barrage2024'
file_name = 'dice2023.csv'

estimates = {}
temps = []
damages = []
weights = []
with open(os.path.join(data_dir, file_name), 'r') as f:
    reader = csv.reader(f, delimiter=',')
    print(reader.__next__())
    for i, line in enumerate(reader):
        study_id, publication_year, temp1920, damage, new, weight = line
        temp = float(temp1920) + 0.4
        damage = -float(damage)/100
        weight = float(weight)
        temps.append(temp)
        damages.append(damage)
        weights.append(weight)
        #if damage >= 0.05:
        #    print(study_id, temp, damage, weight)
        if weight:
            estimates[(study_id, i)] = (temp, damage, weight)

temps = np.array(temps)
damages = np.array(damages)
weights = np.array(weights)

def damage_func(temp, a):
    a1, a2, a3 = a
    return a1*temp + a2*temp**a3

use_weight = True
def Loss(a, estimates=estimates, use_weight=use_weight):
    damages_value = damage_func(temps, a)
    damages_data = damages
    if use_weight:
        errors = (damages_value - damages_data) * (damages_value - damages_data)
        loss = errors.dot(weights)
    else:
        loss = la.norm(damages_value - damages_data)
    #print(loss)
    return loss

a0 = [0.0, 0.003467, 2]
methods = ['Nelder-Mead', 'Powell', 'SLSQP', 'TNC', 'COBYLA', 'BFGS', 'Newton-CG', 'L-BFGS-B']
method = methods[0]
bounds = [(None, None) for i in range(len(a0))]
#bounds[0] = (0, 0)
#bounds[-1] = (2, 2)
tol = 1e-12
maxiter = 100000
res = minimize(fun=Loss, x0=a0, method=method, bounds=bounds, tol=tol, options={'maxiter': maxiter})
print(' method:', method)
print(' message:', res.message)
print(' nit:', res.nit)
print(' status:', res.status)
print(' success:', res.success)
print(' min:', res.fun)
print(' minimizer:', list(res.x))

a = res.x

with open(os.path.join('./output', 'parameter_damage.csv'), 'w') as f:
    f.write(',x\n')
    f.write('a1,{}\n'.format(a[0]))
    f.write('a2,{}\n'.format(a[1]))
    f.write('a3,{}\n'.format(a[2]))

T = np.linspace(0, 10, 100)
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(T, damage_func(T, a), c='k', path_effects=path_effects)
for study_id, t in estimates.items():
    temp, damage, weight = t
    black = np.array((0,0,0))
    white = np.array((1,1,1))
    c = weight*black + (1-weight)*white
    ax.scatter(temp, damage, facecolor=c, zorder=0, marker='o', s=40, edgecolors='k', linewidth=0.5)
ax.set_xlabel("temp")
ax.set_ylabel(f"damage fraction")
ax.set_title(f"f(T)=a1*T + a2*T**a3, a1={a[0]:.4f}, a2={a[1]:.4f}, a3={a[2]:.4f}")
fig.set_tight_layout(True)
fig.savefig("./output/fig_damage_dice2023.svg")
