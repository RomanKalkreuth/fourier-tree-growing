import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, MaxNLocator)
import numpy as np


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    cmap=plt.cm.jet
    cmap.set_bad('k')
    im = ax.imshow(data, cmap=cmap)

    # Create colorbar
    v = np.linspace(0, 100, 100, endpoint=True)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
                   # labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             # rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


# csvfilename = '../../data/ftg/processed_21-01-2024_23h12m15s.csv'
csvfilename = '../../data/ftg/processed_28-01-2024_16h04m22s.csv'

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

MAXK = 8
K = 8
BUDGET = 10**5
ALG = 'ftg'
LAMBDA = 1

fig, axs = plt.subplots(1, 1)
df = pd.read_csv(csvfilename, low_memory=False)
benchmarks = ['koza1','koza2','koza3','nguyen3','nguyen4','nguyen5','nguyen6','nguyen7','nguyen8']
benchmark = 'koza1'
algs = ['one-plus-one', 'one-plus-lambda', 'canonical-ea', 'ftg']
template0 = r'\multirow{36}{*}{\rotatebox{90}{$\sum\limits_{\BF{x} \in \X} \br{F(\BF{x}) - \Fest(\BF{x})}^2 < #TOL#$}}'
template1 = r'& \multirow{ 4}{*}{#BN#}'
template2 = r'& #ALG# & #MEAN# & #SD# & #SEM# & #Q1# & #MED#  & #Q3# & #SR#  \\'
template3 = r'& & #ALG# & #MEAN# & #SD# & #SEM# & #Q1# & #MED#  & #Q3# & #SR#  \\'
template4 = r'\cline{2-10}'
for K in range(2, -1, -1):
    if K == 0:
        print(template0.replace('#TOL#', '1'))
    else:
        print(template0.replace('#TOL#', r'10^{-' + str(K) + r'}'))
    for benchmark in benchmarks:
        print(template1.replace('#BN#', benchmark))
        cnt = 0
        for ALG in algs:
            if ALG == 'one-plus-one':
                ALG = 'one-plus-lambda'
                alg_name = r'$(1+1)$-GP'
                LAMBDA = 1
            elif ALG == 'ftg':
                alg_name = r'\textbf{FTG}'
                LAMBDA = 1
            elif ALG == 'one-plus-lambda':
                LAMBDA = 500
                alg_name = r'($1+\lambda$)-GP'
            elif ALG == 'canonical-ea':
                LAMBDA = 500
                alg_name = r'Canonical-GP'
            df1 = df.loc[(df['alg']==ALG) & (df['benchmark'] == benchmark)& (df['lambda_']==LAMBDA)]
            s = df1[f'evals_to_tol_{K}']
            evals_to_tol = s[s.apply(lambda x: x <= BUDGET)].values
            t5 = r'$#N#$'
            t6 = r'$\mathbf{#N#}$'
            tt = t5
            if ALG == 'ftg':
                tt = t6
            if cnt == 0:
                template = template2
            else:
                template = template3
            template = template.replace('#ALG#', alg_name)
            if len(evals_to_tol) != 0:
                mean = np.mean(evals_to_tol)
                template = template.replace('#MEAN#', tt.replace('#N#', f'{mean:.3f}'))
                sd = np.std(evals_to_tol)
                template = template.replace('#SD#', tt.replace('#N#', f'{sd:.3f}'))
                sem = sd / np.sqrt(len(evals_to_tol))
                template = template.replace('#SEM#', tt.replace('#N#', f'{sem:.3f}'))
                q1 = np.percentile(evals_to_tol, 25)
                template = template.replace('#Q1#', tt.replace('#N#', f'{q1:.3f}'))
                median = np.percentile(evals_to_tol, 50)
                template = template.replace('#MED#', tt.replace('#N#', f'{median:.3f}'))
                q3 = np.percentile(evals_to_tol, 75)
                template = template.replace('#Q3#', tt.replace('#N#', f'{q3:.3f}'))
                sr = len(evals_to_tol) / float(df1['numruns'].values[0]) * 100
                template = template.replace('#SR#', tt.replace('#N#', f'{int(sr)}'))
            else:
                mean = float("inf")
                template = template.replace('#MEAN#', r'$\infty$')
                sd = 0
                template = template.replace('#SD#', r'$0$')
                sem = 0
                template = template.replace('#SEM#', r'$0$')
                q1 = float("inf")
                template = template.replace('#Q1#', r'$\infty$')
                median = float("inf")
                template = template.replace('#MED#', r'$\infty$')
                q3 = float("inf")
                template = template.replace('#Q3#', r'$\infty$')
                sr = 0
                template = template.replace('#SR#', r'$0$')
            # print(f'{benchmark} & {ALG} & {10**(-K):.10f} & {mean:.3f} & {sd:.3f} & {sem:.3f} & {q1:.3f} & {median:.3f} & {q3:.3f} & {sr:.3f}')
            print(template)
            cnt += 1
        if benchmark != 'nguyen8':
            print(template4)
        else:
            print('\n\hline\n')

        # y = np.zeros(BUDGET, dtype=int)
        # for evals in evals_to_tol:
            # y[evals] += 1
        # done = 0
        # for i in range(BUDGET):
            # done += y[i]
            # y[i] = done
        # x = np.linspace(0, BUDGET, BUDGET, dtype=int)
        # axs.plot(x, y, linewidth=1, label=benchmark)

# axs.set_xscale('log')
# axs.yaxis.set_major_locator(MultipleLocator(10))
# axs.yaxis.set_minor_locator(MultipleLocator(5))
# axs.tick_params(axis='x', which='major', labelsize=10, pad=0)
# axs.tick_params(axis='y', which='major', labelsize=10, pad=0)
# axs.grid(which='both', linestyle='--', linewidth='0.5')
# axs.set_xlabel('Number of dataset traverses')
# axs.set_ylabel('Success rate in percents')
# handles, labels = axs.get_legend_handles_labels()
# leg = fig.legend(handles, labels, loc='lower right',
                 # ncol=3, prop={'size': 10}, bbox_to_anchor=(0.9, 0.15))
# for legobj in leg.legend_handles:
    # legobj.set_linewidth(2.)

# fig.savefig('success-rate-ftg-1.pdf')
# plt.close()
