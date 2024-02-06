import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, MaxNLocator)

csvfilename = '../../data/ftg/processed_28-01-2024_14h17m58s.csv'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
BENCHMARKS = ['koza1', 'koza2', 'koza3', 'nguyen3',
              'nguyen4', 'nguyen5', 'nguyen6', 'nguyen7', 'nguyen8']
INSTANCENUM = 100


class Stats:
    def __init__(self):
        self.mean = []
        self.sd = []
        self.sem = []
        self.q1 = []
        self.median = []
        self.q3 = []


def build_chart(x, stats, stats_r):
    names = ['Mean c.n. $\pm$ std', '1Q c.n.', 'Median c.n.', '3Q c.n.']
    cmap=plt.cm.jet_r
    fig, ax = plt.subplots(1, 1)
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(which='both', linestyle='--', linewidth='0.5')
    legends = []
    mean = np.array(stats.mean)
    err = np.array(stats.sd)
    li, = ax.plot(x, mean, label=names[0], c=cmap(0))
    ax.fill_between(x, mean-err, mean+err, alpha=0.2, color=cmap(0))
    # err = np.array(stats.sem)
    # ax.fill_between(x, mean-err, mean+err, alpha=0.2, color=cmap(2))
    # legends.append(li)
    cnt = 3
    for prop in [stats.q1, stats.median, stats.q3]:
        li, = ax.plot(x, prop, label=names[cnt-2], c=cmap(cnt/6))
        legends.append(li)
        cnt += 1
    # li = ax.plot(x, stats_r.mean, label='Mean Retries', c=cmap(0.99))
    # legends.append(li)
    handles, labels = ax.get_legend_handles_labels()
    # leg = fig.legend(handles, labels, loc='upper center', ncol=3, prop={'size': 15}, bbox_to_anchor=(0.5, 0.96))
    leg = fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98))
    # for legobj in leg.legend_handles:
        # legobj.set_linewidth(2.)
    ax.set_ylabel('Condition number of $\mathbf{G}$')

    fig.savefig('ftg-cond.png')
    plt.close()


def build_scatter(x, stats, stats_r):
    names = ['Mean c.n. $\pm$ std', '1Q c.n.', 'Median c.n.', '3Q c.n.']
    cmap=plt.cm.jet_r
    fig, ax = plt.subplots(1, 1)
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(which='both', linestyle='--', linewidth='0.5')
    mean = np.array(stats.mean)
    err = np.array(stats.sd)
    ax.scatter(x, mean, color=cmap(0))
    ax.plot(x, mean, label=names[0], c=cmap(0))
    ax.fill_between(x, mean-err, mean+err, alpha=0.2, color=cmap(0))
    cnt = 3
    for prop in [stats.q1, stats.median, stats.q3]:
        ax.scatter(x, prop, color=cmap(cnt/6))
        ax.plot(x, prop, label=names[cnt-2], c=cmap(cnt/6))
        cnt += 1
    ax.legend(loc='lower center', ncols=2)
    ax.set_ylabel('Condition number of $\mathbf{G}$')
    fig.savefig('ftg-cond.pdf')
    plt.close()


def component_mean_err(data):
    stats = Stats()
    i = 0
    while True:
        row = [l[i] for l in data if i < len(l)]
        if len(row) == 0:
            break
        mean = np.mean(row)
        sd = np.std(row)
        sem = sd / np.sqrt(len(row))
        q1 = np.percentile(row, 25)
        median = np.percentile(row, 50)
        q3 = np.percentile(row, 75)

        stats.mean.append(mean)
        stats.sd.append(sd)
        stats.sem.append(sem)
        stats.q1.append(q1)
        stats.median.append(median)
        stats.q3.append(q3)
        i += 1
    return stats


df = pd.read_csv(csvfilename, low_memory=False)

conds = []
retries = []
mp = {}
for benchmark in BENCHMARKS:
    for instance in range(INSTANCENUM):
        df1 = df.loc[(df['alg'] == 'ftg') & (df['benchmark'] == benchmark) & (df['instance'] == instance)]
        conds.append(df1['cond'].values)
        retries.append(df1['retries'].values)
        reason = df1['term_reason'].values[0]
        if not reason in mp:
            mp[reason] = 1
        else:
            mp[reason] += 1
stats=component_mean_err(conds)
print(stats.mean)
print(stats.median)
stats_r = component_mean_err(retries)
build_scatter(np.array([i+1 for i in range(20)]),stats, stats_r)
print(mp)

