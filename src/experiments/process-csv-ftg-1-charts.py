import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, MaxNLocator)
import numpy as np


csvfilename = './processed_21-01-2024_19h48m11s.csv'

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

K = 3
BUDGET = 10**5
ALG = 'one-plus-lambda'

fig, axs = plt.subplots(1, 1)
df = pd.read_csv(csvfilename, low_memory=False)
# benchmarks = ['koza1','koza2','koza3','nguyen3','nguyen4','nguyen5','nguyen6','nguyen7','nguyen8']
benchmarks = ['koza1']
for benchmark in benchmarks:
    df1 = df.loc[(df['alg']==ALG) & (df['benchmark'] == benchmark)]
    s = df1[f'evals_to_tol_{K}']
    evals_to_tol = s[s.apply(lambda x: x <= BUDGET)].values
    mean = np.mean(evals_to_tol)
    sd = np.std(evals_to_tol)
    sem = sd / np.sqrt(len(evals_to_tol))
    q1 = np.percentile(evals_to_tol, 25)
    median = np.percentile(evals_to_tol, 50)
    q3 = np.percentile(evals_to_tol, 75)
    sr = len(evals_to_tol) / float(df1['numruns'].values[0])
    print('tolerance, mean, sd, sem, q1, median, q3, sr)')
    print(f'{10**(-K):.10f} & {mean:.3f} & {sd:.3f} & {sem:.3f} & {q1:.3f} & {median:.3f} & {q3:.3f} & {sr:.3f}')
    y = np.zeros(BUDGET, dtype=int)
    for evals in evals_to_tol:
        y[evals] += 1
    done = 0
    for i in range(BUDGET):
        done += y[i]
        y[i] = done
    x = np.linspace(0, BUDGET, BUDGET, dtype=int)
    axs.plot(x, y, linewidth=1, label=benchmark)
    # axs.hlines(y=100, xmin=x[-1], xmax=10**5, linewidth=2)

axs.set_xscale('log')
axs.yaxis.set_major_locator(MultipleLocator(10))
axs.yaxis.set_minor_locator(MultipleLocator(5))
axs.tick_params(axis='x', which='major', labelsize=10, pad=0)
axs.tick_params(axis='y', which='major', labelsize=10, pad=0)
axs.grid(which='both', linestyle='--', linewidth='0.5')
axs.set_xlabel('Number of dataset traverses')
axs.set_ylabel('Success rate in percents')
handles, labels = axs.get_legend_handles_labels()
leg = fig.legend(handles, labels, loc='lower right',
                 ncol=3, prop={'size': 10}, bbox_to_anchor=(0.9, 0.15))
for legobj in leg.legend_handles:
    legobj.set_linewidth(2.)

fig.savefig('success-rate-ftg-1.pdf')
plt.close()
