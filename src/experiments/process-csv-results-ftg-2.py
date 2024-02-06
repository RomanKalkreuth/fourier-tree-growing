import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, MaxNLocator)

csvfilename = '../../data/ftg/processed_28-01-2024_13h02m35s.csv'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
BENCHMARKS = ['koza1', 'koza2', 'koza3', 'nguyen3',
              'nguyen4', 'nguyen5', 'nguyen6', 'nguyen7', 'nguyen8']

df = pd.read_csv(csvfilename, low_memory=False)

fig, ax = plt.subplots(1, 1)
ax.set_yscale('log')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.grid(which='both', linestyle='--', linewidth='0.5')
legends = []
for benchmark in BENCHMARKS:
    df1 = df.loc[(df['alg'] == 'ftg') & (df['benchmark'] == benchmark)]
    x = df1['gen'].values
    y = df1['av_cond'].values
    err = df1['std_cond'].values
    li, = ax.plot(x, y, label=benchmark)
    ax.fill_between(x, y-err, y+err, alpha=0.2)
    legends.append(li)

handles, labels = ax.get_legend_handles_labels()
leg = fig.legend(handles, labels, loc='upper center', ncol=3,
                 prop={'size': 15}, bbox_to_anchor=(0.5, 0.96))
# for legobj in leg.legend_handles:
    # legobj.set_linewidth(2.)

fig.savefig('ftg-cond.png')
plt.close()
