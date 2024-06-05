import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import csv
import seaborn as sns
from math import ceil
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(6.4+2,4.8)})
def round_step(n, step):
    if n < step/2:
        return 0
    return ceil(n / step) * step

ax = "score"
plt.rcParams['font.weight'] = 'bold'



filename1 = ax + '2x1024.csv'
filename2 = ax + '4x512.csv'
filename3 = ax + '8x256.csv'



d = {'Algorithm': [], ax: [],'Curriculum': [] }
a = 0
with open(filename1, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    i = 0
    for row in datareader:
        i = i+1
        if i == 1:
            continue
        a = str(row[1])
        if 1010000 < int(row[0]):
            d.get(ax).append(float(row[2]))
            d.get("Algorithm").append(a)
            d.get('Curriculum').append("2x1024")
with open(filename3, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    i = 0
    for row in datareader:
        i = i+1
        if i == 1:
            continue
        a = str(row[1])
        if 1001000 < int(row[0]):
            d.get(ax).append(float(row[2]))
            d.get("Algorithm").append(a)
            d.get('Curriculum').append("4x512")
with open(filename3, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    i = 0
    for row in datareader:
        i = i+1
        if i == 1:
            continue
        a = str(row[1])
        if 1001000 < int(row[0]):
            d.get(ax).append(float(row[2]))
            d.get("Algorithm").append(a)
            d.get('Curriculum').append("8x256")











df = pd.DataFrame(data=d)

swarm_plot = sns.boxplot(data=df, x="Algorithm", y=ax, hue="Curriculum", palette="flare", linewidth=2, showmeans=True)
#swarm_plot = sns.swarmplot(data=df, x="Curriculum", y=ax, hue="Algorithm", palette="flare", size=2, dodge=True, linewidth=0.5)
swarm_plot.tick_params(axis='x', labelsize=14)
swarm_plot.tick_params(axis='y', labelsize=14)
swarm_plot.legend(loc='upper left')
swarm_plot.set_xlabel('CDE Expansion', fontdict={'size': 18}, fontweight="bold")
swarm_plot.set_ylabel('Scores', fontdict={'size': 18}, fontweight="bold")
swarm_plot.set_ylim(bottom=-1, top=0)
#plt.yticks([0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 7/7])
plt.tight_layout()
fig = swarm_plot.get_figure()
#print(str(round(np.mean([ax_value for algorithm, ax_value, curr in zip(d['Algorithm'], d[ax], d['Curriculum']) if algorithm == "DDDQN + EWC + RPAU" and curr == "Custom + Rehearsal"]),3)) + " \pm " + str(round(np.std([ax_value for algorithm, ax_value, curr in zip(d['Algorithm'], d[ax], d['Curriculum']) if algorithm == "DDDQN + EWC + RPAU" and curr == "Custom + Rehearsal"]),3)))

# Perform t-test
algorithm1_data = df[df['Algorithm'] == "DDDQN + RPAU"]
# Replace "Another Algorithm" with the name of your second algorithm if needed
algorithm2_data = df[df['Algorithm'] == "DDDQN"]

t_stat, p_value = stats.ttest_ind(algorithm1_data[ax], algorithm2_data[ax], equal_var=False)

print("T-test results:")
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Adding mean and standard deviation to the plot
mean_value = np.mean(algorithm1_data[ax])
std_value = np.std(algorithm1_data[ax])
print(f"Mean ± Std of rpau: {mean_value:.3f} ± {std_value:.3f}")
mean_value = np.mean(algorithm2_data[ax])
std_value = np.std(algorithm2_data[ax])
print(f"Mean ± Std os vanilla: {mean_value:.3f} ± {std_value:.3f}")
fig = swarm_plot.get_figure()
fig.savefig("out.png")