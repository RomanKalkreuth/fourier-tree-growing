import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, MaxNLocator)
import numpy as np


csvfilename = '../../data/ftg/processed_ftg_21-01-2024_06h38m33s.csv'

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

EPS = 1e-8
fig, axs = plt.subplots(1, 1)
df = pd.read_csv(csvfilename, low_memory=False)
benchmarks = ['koza1','koza2','koza3','nguyen3','nguyen4','nguyen5','nguyen6','nguyen7','nguyen8']
for benchmark in benchmarks:
    df1 = df.loc[(df['alg']=='ftg') & (df['benchmark'] == benchmark)]
    y = df1['numruns'].values - df1['cnt'].values
    x = df1['eval_number'].values
    loss = df1['av_loss'].values
    y = y[100:]
    x = x[100:]
    loss = loss[100:]

    if x[-1] < 10**5 and loss[-1] <= EPS:
        x = np.append(x, x[-1]+1)
        y = np.append(y, 100)


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

fig.savefig('success-rate-ftg.pdf')
plt.close()
