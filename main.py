import os
import sys
import pandas as pd


# -------------------------------------------DATA CLASS--------------------------------------------
class Data:
    results_df = None
    ground_truth_df = None

    def __init__(self, res_file, gnd_file):
        self.read_csv_files(res_file, gnd_file)

    def read_csv_files(self, res_file, gnd_file):
        self.results_df = pd.read_csv(res_file)
        self.ground_truth_df = pd.read_csv(gnd_file)


# ---------------------------------------------UTILS-----------------------------------------------
def create_human_readable_inputs(data):
    if os.stat("human_readable_input/results.txt").st_size == 0:
        with open('human_readable_input/results.txt', 'w+') as writer:
            writer.write(data.train_res_df.to_string())

    if os.stat("human_readable_input/ground-truth.txt").st_size == 0:
        with open('human_readable_input/train_ground-truth.txt', 'w+') as writer:
            writer.write(data.train_gnd_df.to_string())


if __name__ == "__main__":
    # When running from the command-line
    # _, algo_results_file, ground_truth_file = sys.argv

    results_file = 'input/train_algo-results.csv'
    ground_truth_file = 'input/train_ground-truth.csv'

    data_obj = Data(results_file, ground_truth_file)
    create_human_readable_inputs(data_obj)

