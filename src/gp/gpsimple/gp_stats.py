import csv
import time


def write_results_to_csv(results, file_name=None, dir_path=None):
    if file_name is None:
        file_name = "experiment_" + str(time.time()) + ".csv"
    if dir_path is None:
        path = "results/" + file_name
    else:
        path = dir_path + file_name

    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(results)
