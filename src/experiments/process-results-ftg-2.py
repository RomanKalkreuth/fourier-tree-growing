import os
import numpy as np
import datetime
import argparse

class LineInfo:
    def __init__(self):
        self.gen = None
        self.depth = None
        self.size = None
        self.loss = None
        self.num_evals = None
        self.cond = None
        self.retries = None


class GenInfo:
    def __init__(self):
        self.gen_number = 0
        self.cond = 0
        self.retries = 0

    def update(self, li):
        self.gen_number = li.gen
        self.cond = li.cond
        self.retries = li.retries

    def reset(self):
        self.gen_number = 0
        self.cond = 0
        self.retries = 0


class FileInfo:
    def __init__(self):
        self.instance_number = 0
        self.conds = []
        self.retries = []
        self.termination_reason = None

    def update(self, gi):
        self.conds.append(gi.cond)
        self.retries.append(gi.retries)


class DirInfo:
    def __init__(self):
        self.alg = None
        self.benchmark = None
        self.constant = None
        self.lambda_ = None
        self.numruns = 0
        self.instance_number = []
        self.conds = []
        self.retries = []
        self.termination_reasons = []

    def update(self, fi):
        self.numruns += 1
        self.instance_number.append(fi.instance_number)
        self.conds.append(fi.conds)
        self.retries.append(fi.retries)
        self.termination_reasons.append(str(fi.termination_reason))


def parse_line(line):
    li = LineInfo()
    parts = line.split(',')
    for part in parts:
        if ':' in part:
            parts2 = part.split(':')
            word = parts2[0].strip()
            if word == 'gen':
                li.gen = int(parts2[1])
            elif word == 'depth':
                li.depth = int(parts2[1])
            elif word == 'size':
                li.size = int(parts2[1])
            elif word == 'loss':
                li.loss = float(parts2[1])
            elif word == 'num_evals':
                li.num_evals = int(parts2[1])
            elif word == 'cond':
                li.cond = float(parts2[1])
            elif word == 'retries':
                li.retries = int(parts2[1])
    return li


def parse_file(filename):
    minloss = float("inf")
    prvgen = -1
    gi = GenInfo()
    fi = FileInfo()
    fi.instance_number = int(os.path.basename(filename).lstrip('run-'))
    with open(filename, 'r') as file:
        for line in file:
            w1 = line.split(' ')[0]
            if w1 == 'Target':
                continue
            if w1 == 'termination:':
                fi.termination_reason = line.lstrip('termination:').strip()
                continue
            li  = parse_line(line)
            if li.gen == None:
                break
            if prvgen != li.gen and prvgen != -1:
                fi.update(gi)
                gi.reset()
            gi.update(li)
            prvgen = li.gen
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
    di.benchmark = terms[2].lstrip('B-')
    di.constant = terms[3].lstrip('C-')


def parse_setup_directory(directory):
    di = DirInfo()
    last_subdirectory = os.path.basename(directory)
    parse_setup_subdir_name(last_subdirectory, di)
    for filename in os.listdir(directory):
        fullfilename = os.path.join(directory, filename)
        if os.path.isfile(fullfilename) and filename.startswith('run'):
            fi = parse_file(fullfilename)
            di.update(fi)
    return di


def parse_exp_directory(directory, csvfilename):
    with open(csvfilename, 'a') as csvfile:
        for subdirectory in os.listdir(directory):
            sd = os.path.join(directory, subdirectory)
            if os.path.isdir(sd) and subdirectory.startswith('A-'):
                di = parse_setup_directory(sd)
                print(di.alg)
                av_conds, std_conds, cnt = component_mean_err(di.conds)
                av_retries, std_retries, cnt = component_mean_err(di.retries)
                reasons = '\"' + ",".join(di.termination_reasons) + '\"'
                for i in range(len(av_conds)):
                    print(f'{di.alg},{di.benchmark},{di.constant},{di.lambda_},{di.numruns},{cnt[i]},{i},{av_conds[i]:.3f},{std_conds[i]:.3f},{av_retries[i]},{std_retries[i]},{reasons}', file=csvfile)


now = datetime.datetime.now()
timestamp = now.strftime('%d-%m-%Y_%Hh%Mm%Ss')
csvfilename = f'processed_{timestamp}.csv'
print('Logging to', csvfilename)
parser = argparse.ArgumentParser()
parser.add_argument('--dirs', type=str, nargs='+', help='list of exp directories')
args = parser.parse_args()
with open(csvfilename, 'w') as csvfile:
    print('alg,benchmark,constant,lambda_,numruns,cnt,gen,av_cond,std_cond,av_retries,std_retries,term_reasons',file=csvfile)
for expdir in args.dirs:
    print(expdir)
    parse_exp_directory(expdir, csvfilename)

