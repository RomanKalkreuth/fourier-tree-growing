import os
import numpy as np
import datetime
import argparse


class GenInfo:
    def __init__(self):
        self.gen_number = 0
        self.loss = 0
        self.num_evals = 0
        self.depth = 0
        self.size = 0

    def update(self, gen, depth, size, loss, num_evals):
        self.gen_number = gen
        self.depth = depth
        self.size = size
        self.loss = loss
        self.num_evals = num_evals

    def reset(self):
        self.gen_number = 0
        self.loss = 0
        self.num_evals = 0
        self.depth = 0
        self.size = 0


class FileInfo:
    def __init__(self):
        self.depth = []
        self.size = []
        self.loss = []
        self.last_evals = 0
        self.last_loss = float("inf")
        self.last_size = float("inf")
        self.last_depth = float("inf")

    def update(self, gi):
        if self.last_evals != 0:
            for i in range(self.last_evals + 1, gi.num_evals + 1):
                self.depth.append(self.last_depth)
                self.size.append(self.last_size)
                self.loss.append(self.last_loss)
        self.last_evals = gi.num_evals
        self.last_loss = gi.loss
        self.last_size = gi.size
        self.last_depth = gi.depth


class DirInfo:
    def __init__(self):
        self.alg = None
        self.benchmark = None
        self.constant = None
        self.numruns = 0
        self.losses_per_eval = []
        self.size_per_eval = []
        self.depth_per_eval = []

    def update(self, fi):
        self.numruns += 1
        fi.loss.append(fi.last_loss)
        fi.size.append(fi.last_size)
        fi.depth.append(fi.last_depth)
        self.losses_per_eval.append(fi.loss)
        self.size_per_eval.append(fi.size)
        self.depth_per_eval.append(fi.depth)


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
                av_loss, std_loss, cnt = component_mean_err(di.losses_per_eval)
                av_size, std_size, cnt = component_mean_err(di.size_per_eval)
                av_depth, std_depth, cnt = component_mean_err(di.depth_per_eval)
                for i in range(len(av_loss)):
                    print(di.alg, di.benchmark, di.constant, di.numruns, i + 3, cnt[i], av_loss[i], std_loss[i], av_size[i], std_size[i], av_depth[i], std_depth[i], sep=',', file=csvfile)


now = datetime.datetime.now()
timestamp = now.strftime('%d-%m-%Y_%Hh%Mm%Ss')
csvfilename = f'processed_{timestamp}.csv'
parser = argparse.ArgumentParser()
parser.add_argument('--dirs', type=str, nargs='+', help='list of exp directories')
args = parser.parse_args()
with open(csvfilename, 'w') as csvfile:
    print('alg,benchmark,constant,numruns,eval_number,cnt,av_loss,std_loss,av_size,std_size,av_depth,std_depth',file=csvfile)
for expdir in args.dirs:
    print(expdir)
    parse_exp_directory(expdir, csvfilename)

