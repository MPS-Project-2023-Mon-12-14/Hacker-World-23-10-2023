import functools
import operator
import os
import random
import sys
from enum import Enum
import pandas as pd

# ---------------------------------------------GLOBALS---------------------------------------------
NUMBER_OF_TREES_TO_GENERATE = 10
NUMBER_OF_TREES_TO_STORE = 5

RESULTS_FILE = 'input/train_algo-results.csv'
GROUND_TRUTH_FILE = 'input/train_ground-truth.csv'


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


# -------------------------------------------OPERATIONS--------------------------------------------
# Class for all mathematical operations that tree nodes can be randomly chosen from
class Operations(Enum):
    MIN = 'min'
    MAX = 'max'
    ARITHMETIC_MEAN = 'mean'
    GEOMETRIC_MEAN = 'geomean'
    WEIGHTED_MEAN = 'weightedmean'
    MEDIAN = 'median'
    IF_ELSE = 'ifelse'
    # CUSTOM_ARITHMETIC = 'custom_arithmetic'


# ---------------------------------------------NODE------------------------------------------------
# Class for tree node that stores: value, children_list, assigned_op, is_leaf
# <value>: threshold - leaves; computed result - other nodes
# <operation>: None - leaves; randomly chosen op - other nodes
# <children>: the node's children list

class Node:
    def __init__(self, value=None, operation=None, children=None):
        self.value = value
        self.operation = operation
        self.children = children or []


# ---------------------------------------------TREE------------------------------------------------
class Tree:
    def __init__(self, thresholds):
        self.max_levels = random.randint(3, 7)
        self.root = self._random_tree(self.max_levels, thresholds)
        self._evaluate_tree(self.root)

    def _random_tree(self, max_levels, thresholds):
        if max_levels == 0:
            value = random.choice(thresholds.iloc[0])
            return Node(value=value)

        operation = random.choice(list(Operations))
        num_children = random.randint(2, 10)
        children = [self._random_tree(max_levels - 1, thresholds) for _ in range(num_children)]

        return Node(operation=operation, children=children)

    def _evaluate_tree(self, node):
        if not node.children:
            return node.value

        child_values = [self._evaluate_tree(child) for child in node.children]

        if node.operation == Operations.MIN:
            node.value = min(child_values)
        elif node.operation == Operations.MAX:
            node.value = max(child_values)
        elif node.operation == Operations.ARITHMETIC_MEAN:
            node.value = sum(child_values) / len(child_values)
        elif node.operation == Operations.GEOMETRIC_MEAN:
            node.value = (functools.reduce(operator.mul, child_values)) ** (1 / len(child_values))
        elif node.operation == Operations.WEIGHTED_MEAN:
            weights = random.choices(range(2, 11), k=len(child_values))
            node.value = sum(val * weight for val, weight in zip(child_values, weights)) / sum(weights)
        elif node.operation == Operations.MEDIAN:
            node.value = sorted(child_values)[len(child_values) // 2]
        elif node.operation == Operations.IF_ELSE:
            # Determine the number of conditions based on the assumption that
            # child values are organized in pairs
            num_conditions = len(child_values) // 2

            # Check if there is at least one condition-outcome pair
            if num_conditions > 0:

                # Pair conditions and outcomes for iteration using zip
                condition_outcome_pairs = zip(child_values[:num_conditions], child_values[num_conditions:])

                # Iterate through the condition-outcome pairs and assign outcome
                # to node value if condition is True
                for condition, outcome in condition_outcome_pairs:
                    if condition:
                        node.value = outcome
                        break

        # elif node.operation == 'custom_arithmetic':
        #     available_operations = ['x^2', '|x-0.5|', 'x+0.5', '2x', '0.5x', 'x*y', 'x+y', '|x-y|', 'x^y']
        #     operation = random.choice(available_operations)
        #     node.value = self._apply_custom_arithmetic(operation, child_values)

        return node.value

    def print_tree(self):
        self._print_tree(self.root)

    def _print_tree(self, node, level=0):
        if not node.children:
            print(" " * (4 * level) + str(node.value))
        else:
            print(" " * (4 * level) + str(node.operation))
            for child in node.children:
                self._print_tree(child, level + 1)


# ---------------------------------------------UTILS-----------------------------------------------
def create_human_readable_inputs(data):
    if os.stat("human_readable_input/results.txt").st_size == 0:
        with open('human_readable_input/results.txt', 'w+') as writer:
            writer.write(data.results_df.to_string())

    if os.stat("human_readable_input/ground-truth.txt").st_size == 0:
        with open('human_readable_input/train_ground-truth.txt', 'w+') as writer:
            writer.write(data.ground_truth_df.to_string())


# ---------------------------------------------MAIN------------------------------------------------
def main():
    # When running from the command-line
    # _, results_file, ground_truth_file = sys.argv

    data = Data(RESULTS_FILE, GROUND_TRUTH_FILE)
    create_human_readable_inputs(data)

    top_trees = []
    for _ in range(NUMBER_OF_TREES_TO_GENERATE):
        tree = Tree(data.results_df)
        if not top_trees or (tree.root.value is not None and
                             (t.root.value is None or tree.root.value >= t.root.value for t in top_trees)):
            top_trees.append(tree)
            if len(top_trees) > NUMBER_OF_TREES_TO_STORE:
                top_trees.remove(min(top_trees, key=lambda t: t.root.value))

    for i, tree in enumerate(top_trees):
        print(f"Tree {i + 1}:")
        tree.print_tree()


if __name__ == "__main__":
    main()
