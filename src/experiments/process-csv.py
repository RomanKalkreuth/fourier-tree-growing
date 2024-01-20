import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, MaxNLocator)


csvfilename = 'processed_19-01-2024_19h27m49s.csv'


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# Read the CSV file into a DataFrame
df = pd.read_csv(csvfilename, low_memory=False)
algs = ['canonical-ea', 'one-plus-lambda']
colors = ['blue', 'green']
fig, axs = plt.subplots(4, 4, figsize=(15, 10))
legends = []
row = 0
col_labels = ['min loss', 'max span', 'mean tree depth', 'mean tree size']
for lambda_ in [500]:
    for degree in [10, 100]:
        for constant in ['1', 'koza-erc']:
            col = 0
            for mean_name, err_name in [('av_min_loss', 'std_min_loss'), ('av_max_span', 'std_max_span'), ('av_av_tdepth', 'std_av_tdepth'), ('av_av_tsize', 'std_av_tsize')]:
                ax = axs[row, col]

                ax.tick_params(axis='x', which='major', labelsize=10, pad=0)
                ax.tick_params(axis='y', which='major', labelsize=10, pad=0)
                ax.grid(which='both', linestyle='--', linewidth='0.05')
                if row == 3:
                    ax.set_xlabel(col_labels[col], loc='center', size=15)
                if col == 0:
                    if constant == 'koza-erc':
                        c='(-1,1)'
                    else:
                        c='{1}'
                    ax.set_ylabel(f'$d={degree}, c={c}$', loc='center', size=10)
                for i in range(len(algs)):
                    df1 = df.loc[(df['alg'] == algs[i]) & (df['lambda_'] == lambda_) & (
                        df['degree'] == degree) & (df['constant'] == constant)]
                    if mean_name == 'av_min_loss':
                        ax.set_yscale('log')
                    x = df1['gen_number'].values
                    y = df1[mean_name].values
                    err = df1[err_name].values
                    li, = ax.plot(x, y, c=colors[i], label=algs[i])
                    ax.fill_between(x, y - err, y + err,
                                    facecolor=colors[i], alpha=0.20)
                    legends.append(li)
                col += 1
            if row == 0:
                handles, labels = ax.get_legend_handles_labels()
                leg = fig.legend(handles, labels, loc='upper center',
                                 ncol=len(algs), prop={'size': 15}, bbox_to_anchor=(0.5, 0.95))
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(2.)
            row += 1
fig.savefig('large-scale-poly.pdf')
plt.close()
