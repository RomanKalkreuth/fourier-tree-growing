import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, MaxNLocator)


csvfilename = '../../data/large-scale-poly/processed_21-01-2024_11h17m13s.csv'


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# Read the CSV file into a DataFrame
df = pd.read_csv(csvfilename, low_memory=False)
algs = ['canonical-ea', 'one-plus-lambda']
names = ['Canonical-GP', r'$(1 + \lambda)$-GP']
colors = ['blue', 'green']
fig, axs = plt.subplots(4, 4, figsize=(15, 10))
legends = []
col = 0
col_labels = ['min loss', 'max span', 'mean tree depth', 'mean tree size']
for lambda_ in [500]:
    for degree in [10, 100]:
        for constant in ['1', 'none']:
            row = 0
            for mean_name, err_name in [('av_min_loss', 'std_min_loss'), ('av_max_span', 'std_max_span'), ('av_av_tdepth', 'std_av_tdepth'), ('av_av_tsize', 'std_av_tsize')]:
                ax = axs[row, col]

                ax.tick_params(axis='x', which='major', labelsize=10, pad=0)
                ax.tick_params(axis='y', which='major', labelsize=10, pad=0)
                ax.grid(which='both', linestyle='--', linewidth='0.05')
                if col == 0:
                    ax.set_ylabel(col_labels[row], loc='center', size=13)
                if row == 0:
                    if constant == 'koza-erc':
                        c = '(-1,1)'
                    elif constant == '1':
                        c = '\\{1\\}'
                    elif constant == 'none':
                        c = '\\varnothing'
                    ax.set_title(f'$k={degree}, c={c}$', loc='center', size=15)
                for i in range(len(algs)):
                    df1 = df.loc[(df['alg'] == algs[i]) & (df['lambda_'] == lambda_) & (
                        df['degree'] == degree) & (df['constant'] == constant)]
                    # if mean_name == 'av_min_loss':
                    ax.set_yscale('log')
                    x = df1['gen_number'].values
                    y = df1[mean_name].values
                    err = df1[err_name].values
                    li, = ax.plot(x, y, c=colors[i], label=names[i])
                    ax.fill_between(x, y - err, y + err,
                                    facecolor=colors[i], alpha=0.20)
                    legends.append(li)
                row += 1
            if row == 4:
                events_sign = []
                for i in range(len(algs)):
                    df1 = df.loc[(df['alg'] == algs[i]) & (df['lambda_'] == lambda_) & (df['degree'] == degree) & (df['constant'] == constant)]
                    av = float(df1['av_cnt_event'].values[0])
                    std = float(df1['std_cnt_event'].values[0])
                    sign = f'${av:.2f} \pm {std:.2f}$'
                    events_sign.append(sign)
                plt.figtext(0.18 + 0.2 * col, 0.07, events_sign[0], horizontalalignment='left', fontsize=15, color=colors[0])
                plt.figtext(0.18 + 0.2 * col, 0.04, events_sign[1], horizontalalignment='left', fontsize=15, color=colors[1])
                # sign = '; '.join(events_sign)
                # ax.set_xlabel(sign, loc='center', size=10, color='blue')
                handles, labels = ax.get_legend_handles_labels()
                leg = fig.legend(handles, labels, loc='upper center',
                                 ncol=len(algs), prop={'size': 15}, bbox_to_anchor=(0.5, 0.96))
                for legobj in leg.legend_handles:
                    legobj.set_linewidth(2.)
            col += 1
fig.savefig('large-scale-poly.pdf')
plt.close()
