# @author Kirill Antonov

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import csv


def extract_bestsofar(everyeval_path, budget):
    if not os.path.exists(everyeval_path):
        return
    with open(everyeval_path, 'r') as f:
        r = csv.reader(f, delimiter=' ')
        next(r, None)
        global_min = float("inf")
        x = []
        y = []
        cnt = 0
        for row in r:
            if row[1] == 'raw_y':
                continue
            v = float(row[1])
            if v < global_min:
                x.append(cnt)
                y.append(v)
                global_min = v
            else:
                x.append(cnt)
                y.append(global_min)
            cnt += 1
        if len(x) == 0 or x[len(x) - 1] < 0.1*budget:
            return
        while x[len(x) - 1] > budget:
            x.pop()
            y.pop()
        while x[len(x) - 1] < budget:
            x.append(len(x))
            y.append(y[len(y) - 1])
        assert len(x) == len(y)
        print(len(x))
        return np.array(x), np.array(y)


def process_data_folder(data_folder, nruns, budget, fname, dim):
    good_x = []
    ys = []
    for i in range(nruns):
        path = data_folder + \
            f'/everyeval-{i}/data_f25_{fname}/IOHprofiler_f25_DIM{dim}.dat'
        print(path)
        res = extract_bestsofar(path, budget)
        if res:
            good_x, y = res
            ys.append(y)

    Y = np.array(ys)
    y1 = []
    err = []
    for i in range(len(Y[0])):
        y1.append(np.nanmean(Y[:, i]))
        err.append(np.nanstd(Y[:, i]))

    y1 = np.array(y1)
    err = np.array(err)
    return good_x, y1, err


def build_averaged_convergence2(fname, nruns=10, budget=10000):
    data_folders = [f'gp-umda-exp-{fname}', f'gp-rs-exp-{fname}']
    colors = mpl.cm.jet(np.linspace(0, 1, len(data_folders)))
    names = ['UMDA, 10 runs', 'RS, 10 runs']
    pdf_name = f'averaged-convergence-22-11-23-{fname}.pdf'

    cnt = len(data_folders)

    x = []
    ys = []
    errs = []
    for data_folder in data_folders:
        x, y, err = process_data_folder(data_folder, nruns, budget, fname, 1)
        x = x[0:]
        y = y[0:]
        err = err[0:]
        ys.append(y)
        errs.append(err)

    fig = plt.figure()
    plt.rcParams.update({'font.size': 10})
    ax = plt.gca()
    ax.set_yscale('log')
    lis = []
    for i in range(cnt):
        y, err, cl = ys[i], errs[i], colors[i % len(colors)]
        li, = plt.plot(x, y, c=cl)
        lis.append(li)
        ax.fill_between(x, y - err, y + err, facecolor=cl, alpha=0.20)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_minor_formatter().set_scientific(False)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    yticks = [0.1, 0.5, 1, 1.5, 5.0, 10.0]
    ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(yticks))
    plt.yticks(yticks, fontsize=8)
    plt.legend(lis, names)
    plt.grid(linewidth=0.4)
    fig.text(0.5, 0.02, 'Number of $f$ evaluations', ha='center')
    fig.text(0.06, 0.52, 'Best-so-far $f(x)$',
             va='center', rotation='vertical')
    fig.savefig(pdf_name)
    plt.close()


build_averaged_convergence2('koza2')
