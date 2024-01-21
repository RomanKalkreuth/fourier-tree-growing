import os
import numpy as np
import datetime
import argparse


class GenInfo:
    def __init__(self):
        self.gen_number = 0
        self.min_loss = float("inf")
        self.num_evals = 0

    def update(self, gen, depth, size, loss, num_evals):
        self.gen_number = gen
        self.min_loss = min(loss, self.min_loss)
        self.num_evals = num_evals

    def reset(self):
        self.gen_number = 0


class FileInfo:
    def __init__(self):
        self.instance_number = 0
        self.cnt_tol = [10**9] * 10

    def update(self, gi):
        for k in range(len(self.cnt_tol)):
            eps = 10**(-k)
            if gi.min_loss <= eps:
                self.cnt_tol[k] = min(self.cnt_tol[k], gi.num_evals)


class DirInfo:
    def __init__(self):
        self.alg = None
        self.benchmark = None
        self.constant = None
        self.lambda_ = None
        self.numruns = 0
        self.instance_number = []
        self.instance_cnt_tol = []

    def update(self, fi):
        self.numruns += 1
        self.instance_number.append(fi.instance_number)
        self.instance_cnt_tol.append(fi.cnt_tol)


def parse_line(line):
    parts = line.split(',')
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
            elif word == 'num_evals':
                num_evals = int(parts2[1])
    return gen, depth, size, loss, num_evals


def parse_file(filename):
    minloss = float("inf")
    prvgen = -1
    gi = GenInfo()
    fi = FileInfo()
    fi.instance_number = int(os.path.basename(filename).lstrip('run-'))
    with open(filename, 'r') as file:
        for line in file:
            if line.split(' ')[0] == 'Target':
                continue
            gen, depth, size, loss, num_evals = parse_line(line)
            if gen == None:
                break
            if prvgen != gen and prvgen != -1:
                fi.update(gi)
                gi.reset()
            gi.update(gen, depth, size, loss, num_evals)
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
                for i in range(di.numruns):
                    print(di.alg, di.benchmark, di.constant, di.lambda_, di.numruns,di.instance_number[i], sep=',', end='', file=csvfile)
                    for j in range(len(di.instance_cnt_tol[i])):
                        print(',', di.instance_cnt_tol[i][j], sep='', end='', file=csvfile)
                    print(file=csvfile)


now = datetime.datetime.now()
timestamp = now.strftime('%d-%m-%Y_%Hh%Mm%Ss')
csvfilename = f'processed_{timestamp}.csv'
print('Logging to', csvfilename)
parser = argparse.ArgumentParser()
parser.add_argument('--dirs', type=str, nargs='+', help='list of exp directories')
args = parser.parse_args()
with open(csvfilename, 'w') as csvfile:
    print('alg,benchmark,constant,lambda_,numruns,instance,evals_to_tol_0,evals_to_tol_1,evals_to_tol_2,evals_to_tol_3,evals_to_tol_4,evals_to_tol_5,evals_to_tol_6,evals_to_tol_7,evals_to_tol_8,evals_to_tol_9',file=csvfile)
for expdir in args.dirs:
    print(expdir)
    parse_exp_directory(expdir, csvfilename)

