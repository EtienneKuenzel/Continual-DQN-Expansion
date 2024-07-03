import pandas as pd
import csv
from math import floor
import seaborn as sns
from argparse import ArgumentParser

import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
def round_step(n, step):
    if n < step/2:
        return 0
    return floor(n / step) * step


def process_file(filename, typeof):
    smooth = 80000
    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)  # Skip the header row
        for row in datareader:
            print(float(row[0]))
            d.get("networksteps").append(round_step(int(row[0]), smooth))
            d.get(typeof).append(float(row[3]))
            d.get("Environment").append(str(row[4]))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", help="insert file path",default='eval.csv', type=str)
    parser.add_argument("--type", help="completions, score",default="completions", type=str)
    parser_args = parser.parse_args()

    d = {'networksteps': [], 'Environment': [], parser_args.type: []}
    process_file(parser_args.file, parser_args.type)
    df = pd.DataFrame(data=d)


    swarm_plot = sns.lineplot(x="networksteps", y=parser_args.type, hue="Environment",style ="Environment",palette="flare", data=df, zorder=4, errorbar=None)
    plt.rcParams['font.weight'] = 'bold'


    for x in range(1000000):
        if x == 0:
            continue
        f = [80000]*12
        f = [value * (index + 1) for index, value in enumerate(f)]
        if x in f:
           plt.axvline(x=x, color='black',alpha=0.5, zorder=1, linewidth="1", linestyle="dotted")
        if x in [320000, 640000, 1000000] :
            plt.axvline(x=x, color='black',alpha=0.5, zorder=1, linewidth="1", linestyle="solid")


    swarm_plot.grid(visible=None, which='major', axis='x')
    swarm_plot.set_xlim(left=0000000, right=1300000)

    if parser_args.type =="completions":
        swarm_plot.legend(loc='upper left')
        swarm_plot.set_ylim(bottom=0, top=1)
    if parser_args.type =="score":
        swarm_plot.legend(loc='lower left')
        swarm_plot.set_ylim(bottom=-1, top=0)

    swarm_plot.tick_params(axis='x', labelsize=14)
    swarm_plot.tick_params(axis='y', labelsize=14)
    swarm_plot.set_xlabel('Networksteps', fontdict={'size': 18}, fontweight="bold")
    swarm_plot.set_ylabel(parser_args.type, fontdict={'size': 18}, fontweight="bold")
    plt.tight_layout()
    fig = swarm_plot.get_figure()

    fig.savefig(parser_args.file[:-4] + "_training.png")