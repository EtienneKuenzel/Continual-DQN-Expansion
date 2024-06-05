import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import csv
import seaborn as sns
from math import ceil, floor
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
def round_step(n, step):
    if n < step/2:
        return 0
    return floor(n / step) * step
ax = "completions"
plt.rcParams['font.weight'] = 'bold'
filename1 = ax + '2x1024.csv'
filename2 = ax + '4x512.csv'
filename3 = ax + '8x256.csv'

smoothing = 0.95

d = {'networksteps': [], 'Algorithm': [], ax: []}
smooth = 32000
with open(filename1, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    i = 0
    for row in datareader:
        i = i+1
        if i == 1:
            continue
        a = str(row[1])
        if True:
            d.get("networksteps").append(round_step(int(row[0]), smooth))
            d.get(ax).append(float(row[2]))
            d.get("Algorithm").append(a + "1024")
with open(filename2, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    i = 0
    for row in datareader:
        i = i+1
        if i == 1:
            continue
        a = str(row[1])
        if True:
            d.get("networksteps").append(round_step(int(row[0]), smooth))
            d.get(ax).append(float(row[2]))
            d.get("Algorithm").append(a+ "512")
with open(filename3, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    i = 0
    for row in datareader:
        i = i+1
        if i == 1:
            continue
        a = str(row[1])
        if True:
            d.get("networksteps").append(round_step(int(row[0]), smooth))
            d.get(ax).append(float(row[2]))
            d.get("Algorithm").append(a+ "265")

df = pd.DataFrame(data=d)

swarm_plot = sns.lineplot(x="networksteps", y=ax, hue="Algorithm",style ="Algorithm",palette="flare", data=df, zorder=4, errorbar=None)


for x in range(1000000):
    if x == 0:
        continue
    #if 0 == x % 65000and x not in[260000, 520000, 585000, 845000,910000,975000]:
    #   plt.axvline(x=x, color='black',alpha=0.5, zorder=1, linewidth="1", linestyle="dotted")
    if x in [320000, 640000] :
        plt.axvline(x=x, color='black',alpha=0.5, zorder=1, linewidth="1", linestyle="solid")


swarm_plot.grid(visible=None, which='major', axis='x')
swarm_plot.set_ylim(bottom=0, top=1)
swarm_plot.set_xlim(left =0000000, right=1300000)
swarm_plot.legend(loc='upper left')

swarm_plot.tick_params(axis='x', labelsize=14)
swarm_plot.tick_params(axis='y', labelsize=14)
swarm_plot.set_xlabel('Networksteps', fontdict={'size': 18}, fontweight="bold")
swarm_plot.set_ylabel('Completions', fontdict={'size': 18}, fontweight="bold")
plt.tight_layout()
fig = swarm_plot.get_figure()

fig.savefig("customwith_" + ax + ".png")