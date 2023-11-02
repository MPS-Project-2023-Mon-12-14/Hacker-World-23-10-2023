import os
import sys
from enum import Enum
import pandas as pd


# ----------------------------------------------DATA-----------------------------------------------
# Class for all types of data: input CSVs, best trees generated, etc.
# Best trees generated - may be a priority queue (ordered by tree's score)
class Data:
    results_df = None
    ground_truth_df = None

    def __init__(self, res_file, gnd_file):
        self.read_csv_files(res_file, gnd_file)

    def read_csv_files(self, res_file, gnd_file):
        self.results_df = pd.read_csv(res_file)
        self.ground_truth_df = pd.read_csv(gnd_file)


# ---------------------------------------------NODE------------------------------------------------
# Class for tree node that stores: value, tree_level, node_index, assigned_op, is_leaf
# <value>: threshold - leaves; computed result - other nodes
# <tree_level>: the level in the tree
# <node_index>: the node's index on the specified level
# <assigned_op>: None - leaves; randomly chosen op - other nodes
# <is_leaf>: boolean flag that checks if the node is a flag or not

# TODO: implement class Node
class Node:
    pass


# ---------------------------------------------TREE------------------------------------------------
class Tree:
    def __init__(self):
        self.generate_tree()

    # TODO: Randomly generate number of levels, number of nodes per level, operations for nodes, etc..
    def generate_tree(self):
        pass


# -------------------------------------------OPERATIONS--------------------------------------------
# Class for all mathematical operations that tree nodes can be randomly chosen from
class Operations(Enum):
    # TODO: add all possible mathematical operations - examples in the project's text
    MIN = 1
    MAX = 2

    def __init__(self):
        super().__init__()

    # TODO: add methods that can handle 2 or more inputs for each operation defined above


# ---------------------------------------------UTILS-----------------------------------------------
def create_human_readable_inputs(data):
    if os.stat("human_readable_input/results.txt").st_size == 0:
        with open('human_readable_input/results.txt', 'w+') as writer:
            writer.write(data.results_df.to_string())

    if os.stat("human_readable_input/ground-truth.txt").st_size == 0:
        with open('human_readable_input/train_ground-truth.txt', 'w+') as writer:
            writer.write(data.ground_truth_df.to_string())


if __name__ == "__main__":
    # When running from the command-line
    # _, results_file, ground_truth_file = sys.argv

    results_file = 'input/train_algo-results.csv'
    ground_truth_file = 'input/train_ground-truth.csv'

    data_obj = Data(results_file, ground_truth_file)
    create_human_readable_inputs(data_obj)
