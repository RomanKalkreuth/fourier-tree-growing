import os
import numpy as np
import datetime
import argparse

CS_DTYPE = int
def parse_poly_str(str_poly):
    ds, cs = [], []
    sign = 1
    for s in str_poly.split('*'):
        for t in s.split(' '):
            if t == '-':
                sign = -1
            elif t == '+':
                sign = 1
            elif t[0] == 'x':
                d = t.split('^')[1]
                ds.append(int(d))
            else:
                cs.append(CS_DTYPE(t)*sign)
                sign = 1
    if len(ds) == 0 and len(cs) == 1:
        ds.append(0)
    return ds, cs


def parse_line(line):
    parts = line.split(',')
    if len(parts) < 5 or len(parts[-1].strip()) == 0:
        return None, None, None, None, None, None
    for part in parts:
        if ':' in part:
            parts2 = part.split(':')
            word = parts2[0].strip()
            if word == 'gen':
                gen = int(parts2[1])
            elif word == 'depth':
                depth = int(parts2[1])
            elif word == 'size':
                size = int(parts2[1])
            elif word == 'loss':
                loss = float(parts2[1])
        else:
            ds, cs = parse_poly_str(part.strip())
    return gen, depth, size, loss, ds, cs


class GenInfo:
    def __init__(self):
        self.gen_number = 0
        self.gen_max_span = 0
        self.gen_min_loss = float("inf")
        self.gen_sum_td = 0
        self.gen_sum_ts = 0
        self.gen_offspring_cnt = 0

    def update(self, gen, depth, size, loss, ds, cs):
        self.gen_number = gen
        self.gen_max_span = max(self.gen_max_span, len(ds))
        self.gen_min_loss = min(self.gen_min_loss, loss)
        self.gen_sum_td += depth
        self.gen_sum_ts += size
        self.gen_offspring_cnt += 1

    def reset(self):
        self.gen_max_span = 0
        self.gen_min_loss = float("inf")
        self.gen_sum_td = 0
        self.gen_sum_ts = 0
        self.gen_offspring_cnt = 0


class FileInfo:
    def __init__(self):
        self.gen_number = []
        self.gen_min_loss = []
        self.gen_max_span = []
        self.gen_av_td = []
        self.gen_av_ts = []

    def update(self, gi):
        self.gen_number.append(gi.gen_number)
        self.gen_min_loss.append(gi.gen_min_loss)
        self.gen_max_span.append(gi.gen_max_span)
        self.gen_av_td.append(gi.gen_sum_td / gi.gen_offspring_cnt)
        self.gen_av_ts.append(gi.gen_sum_ts / gi.gen_offspring_cnt)


class DirInfo:
    def __init__(self):
        self.alg = None
        self.lambda_ = None
        self.degree = None
        self.constant = None
        self.numruns = 0
        self.min_losses_per_gen = []
        self.max_spans_per_gen = []
        self.av_tdepths_per_gen = []
        self.av_tsize_per_gen = []

    def update(self, fi):
        self.numruns += 1
        self.min_losses_per_gen.append(fi.gen_min_loss)
        self.max_spans_per_gen.append(fi.gen_max_span)
        self.av_tdepths_per_gen.append(fi.gen_av_td)
        self.av_tsize_per_gen.append(fi.gen_av_ts)


def parse_file(filename):
    minloss = float("inf")
    prvgen = -1
    gi = GenInfo()
    fi = FileInfo()
    with open(filename, 'r') as file:
        for line in file:
            if line.split(' ')[0] == 'Target':
                continue
            gen, depth, size, loss, ds, cs = parse_line(line)
            if gen == None:
                break
            if prvgen != gen and prvgen != -1:
                fi.update(gi)
                gi.reset()
            gi.update(gen, depth, size, loss, ds, cs)
            prvgen = gen
        fi.update(gi)
    return fi


def component_mean_err(data):
    means = []
    stds = []
    cnts = []
    i = 0
    while True:
        row = [l[i] for l in data if i < len(l)]
        if len(row) == 0:
            break
        mean = np.mean(row)
        std = np.std(row)
        means.append(mean)
        stds.append(std)
        cnts.append(len(row))
        i += 1
    return means, stds, cnts


def parse_setup_subdir_name(name, di):
    terms = name.split('_')
    di.alg = terms[0].lstrip('A-')
    di.lambda_ = int(terms[1].lstrip('L'))
    di.degree = int(terms[2].lstrip('D'))
    di.constant = terms[3].lstrip('C-')
    global CS_DTYPE
    if di.constant == 'koza-erc':
        CS_DTYPE = float
    else:
        CS_DTYPE = int


def parse_setup_directory(directory):
    di = DirInfo()
    last_subdirectory = os.path.basename(directory)
    parse_setup_subdir_name(last_subdirectory, di)
    for filename in os.listdir(directory):
        fullfilename = os.path.join(directory, filename)
        if os.path.isfile(fullfilename) and filename.startswith('run'):
            fi = parse_file(fullfilename)
            di.update(fi)
    # av, err = component_mean_err(di.min_losses_per_gen)
    # return av, err
    return di


def parse_exp_directory(directory, csvfilename):
    with open(csvfilename, 'w') as csvfile:
        print('alg,lambda_,degree,constant,numruns,gen_number,cnt,av_min_loss,std_min_loss,av_max_span,std_max_span,av_av_tdepth,std_av_tdepth,av_av_tsize,std_av_tsize',file=csvfile)
        for subdirectory in os.listdir(directory):
            sd = os.path.join(directory, subdirectory)
            if os.path.isdir(sd) and subdirectory.startswith('A-'):
                di = parse_setup_directory(sd)
                av_min_loss, std_min_loss, cnt = component_mean_err(di.min_losses_per_gen)
                av_max_span, std_max_span, cnt = component_mean_err(di.max_spans_per_gen)
                av_av_tdepth, std_av_tdepth, cnt = component_mean_err(di.av_tdepths_per_gen)
                av_av_tsize, std_av_tsize, cnt = component_mean_err(di.av_tsize_per_gen)
                for i in range(len(av_min_loss)):
                    print(di.alg, di.lambda_, di.degree, di.constant, di.numruns, i, cnt[i], av_min_loss[i], std_min_loss[i], av_max_span[i], std_max_span[i], av_av_tdepth[i], std_av_tdepth[i], av_av_tsize[i], std_av_tsize[i], sep=',', file=csvfile)


now = datetime.datetime.now()
timestamp = now.strftime('%d-%m-%Y_%Hh%Mm%Ss')
csvfilename = f'processed_{timestamp}.csv'
parser = argparse.ArgumentParser()
parser.add_argument('--dirs', type=str, nargs='+', help='list of exp directories')
args = parser.parse_args()
for expdir in args.dirs:
    parse_exp_directory(expdir, csvfilename)

