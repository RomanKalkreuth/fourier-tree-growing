import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

BENCHMARKS = ['koza1','koza2','koza3','nguyen3','nguyen4','nguyen5','nguyen6','nguyen7','nguyen8']
ALGS = ['ftg', 'one-plus-one', 'one-plus-lambda', 'canonical-ea']
MAXK = 8
BUDGET = 10**5


def extract_data(csvfilename):
    df = pd.read_csv(csvfilename, low_memory=False)
    algs_tol_data_sr, algs_tol_data_av_fe = [], []
    for alg in ALGS:
        data_sr = np.zeros((len(BENCHMARKS), MAXK + 1))
        data_av_fe = np.zeros((len(BENCHMARKS), MAXK + 1))
        algs_tol_data_sr.append(data_sr)
        algs_tol_data_av_fe.append(data_av_fe)
        for i_bench in range(len(BENCHMARKS)):
            for k in range(MAXK + 1):
                if alg == 'one-plus-one':
                    alg = 'one-plus-lambda'
                    lambda_ = 1
                elif alg == 'ftg':
                    lambda_ = 1
                else:
                    lambda_ = 500
                benchmark = BENCHMARKS[i_bench]
                df1 = df.loc[(df['alg']==alg) & (df['benchmark'] == BENCHMARKS[i_bench])& (df['lambda_']==lambda_)]
                s = df1[f'evals_to_tol_{k}']
                evals_to_tol = s[s.apply(lambda x: x <= BUDGET)].values
                if len(evals_to_tol) != 0:
                    mean = np.mean(evals_to_tol)
                    sd = np.std(evals_to_tol)
                    sem = sd / np.sqrt(len(evals_to_tol))
                    q1 = np.percentile(evals_to_tol, 25)
                    median = np.percentile(evals_to_tol, 50)
                    q3 = np.percentile(evals_to_tol, 75)
                    sr = len(evals_to_tol) / float(df1['numruns'].values[0]) * 100
                else:
                    mean = float("inf")
                    sd = 0
                    sem = 0
                    q1 = float("inf")
                    median = float("inf")
                    q3 = float("inf")
                    sr = 0
                data_sr[i_bench][k] = sr
                data_av_fe[i_bench][k] = median
    return algs_tol_data_sr, algs_tol_data_av_fe


sns.set(font_scale=1.2)
def build_heatmap(data, vmin, vmax, name, cmap='jet'):
    xlabels = ['$10^{' + str(-k) + '}$' for k in range(MAXK+1)]
    fig, ax = plt.subplots(1, 1)
    hm = sns.heatmap(data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidth=1, yticklabels=BENCHMARKS, xticklabels=xlabels, annot_kws={"fontsize":10})
    fig.colorbar(hm.collections[0], ax=ax, location='right')
    fig.savefig(f'{name}.pdf')
    plt.close()


def to_comparison_value(values):
    a = values[0]
    b = max(values[1:])
    v = int((a+1)-(b+1))
    print(v)
    return v


def to_comparison_value_2(values):
    a = values[0]
    b = min(values[1:])
    v = (a)/(b)
    print(v)
    return v


def combine_algs_results(data, to_value):
    ans = np.zeros((len(BENCHMARKS), MAXK + 1))
    for i in range(len(BENCHMARKS)):
        for k in range(MAXK + 1):
            ans[i][k] = to_value([d[i][k] for d in data])
    return ans


# d1, d2 = extract_data('../../data/ftg/processed_21-01-2024_23h12m15s.csv')
d1, d2 = extract_data('../../data/ftg/processed_28-01-2024_16h04m22s.csv')
combinted_sr = combine_algs_results(d1, to_comparison_value)
cmapr = sns.color_palette("RdGy", as_cmap=True)
cmap = sns.color_palette("RdGy", as_cmap=True).reversed()
build_heatmap(combinted_sr, vmin=0, vmax=100, name='heatmap-combined-sr', cmap=cmap)
cr2 = combine_algs_results(d2, to_comparison_value_2)
build_heatmap(cr2, vmin=0, vmax=1., name='heatmap-combined-av-med', cmap=cmapr)


