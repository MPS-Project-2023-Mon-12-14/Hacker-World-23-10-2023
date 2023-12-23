# main.py
"""
This is a module of the application that concerns itself with Data, Operations,
Node, Tree and Utils classes. It also contains the main function that runs the
application.
:const NUMBER_OF_TREES_TO_GENERATE: number of trees to generate
:const NUMBER_OF_TREES_TO_STORE: number of trees to store
:const RESULTS_FILE: path to the results CSV file
:const GROUND_TRUTH_FILE: path to the ground truth CSV file
:cls Data: class for all types of data: input CSVs, best trees generated, etc.
:cls Operations: class for all mathematical operations that tree nodes can be
                    randomly chosen from
:cls Node: class for tree node that stores value, operation and children list
:cls Tree: class for tree that stores root node and max_levels
:cls Utils: class for utility functions
:func main: main function that runs the application
"""

import functools
import operator
import os
import random
import string
import time
from enum import Enum
from threading import Thread
from typing import ForwardRef, List, Union
import re

import numpy as np
import pandas as pd

NUMBER_OF_TREES_TO_GENERATE = 2
LOCAL_FOLDER = "local/train"


class Data:
    """
    Class for all types of data: input CSVs, best trees generated, etc. Best
    trees generated may be a priority queue (ordered by tree's score).
    :attr results_df: str representing the path to the results CSV file
    :attr ground_truth_df: str representing the path to the ground truth CSV
                            file
    :meth __init__(res_file, gnd_file): constructor for Data class
    :meth read_csv_files: reads the CSV files and stores them inside the class
                            attributes
    """

    def __init__(self, res_file: str, gnd_file: str) -> None:
        """
        Constructor for Data class.
        :param res_file: str representing the path to the results CSV file
        :param gnd_file: str representing the path to the ground truth CSV file
        :return: None
        """
        self.results_df = None
        self.ground_truth_df = None
        self.read_csv_files(res_file, gnd_file)

    def read_csv_files(self, res_file: str, gnd_file: str) -> None:
        """
        Reads the CSV files and stores them inside the class attributes.
        :param res_file: str representing the path to the results CSV file
        :param gnd_file: str representing the path to the ground truth CSV file
        :return: None
        """
        self.results_df = pd.read_csv(res_file)
        self.ground_truth_df = pd.read_csv(gnd_file)


class Operations(Enum):
    """
    Class for all mathematical operations that tree nodes can be randomly chosen
    from.
    :const MIN: str representing the minimum operation
    :const MAX: str representing the maximum operation
    :const ARITHMETIC_MEAN: str representing the arithmetic mean operation
    :const GEOMETRIC_MEAN: str representing the geometric mean operation
    :const WEIGHTED_MEAN: str representing the weighted mean operation
    :const MEDIAN: str representing the median operation
    :const IF_ELSE: str representing the if-else operation
    """

    MIN = "min"
    MAX = "max"
    ARITHMETIC_MEAN = "mean"
    GEOMETRIC_MEAN = "geomean"
    WEIGHTED_MEAN = "weightedmean"
    MEDIAN = "median"
    IF_ELSE = "ifelse"
    # self.CUSTOM_ARITHMETIC = 'custom_arithmetic'


class Node:
    """
    Class for tree node that stores value, operation and children list.
    :attr value: Union[None, np.float64] representing the threshold, for leaves,
                    or the computed result, for other nodes
    :attr operation: Union[None, Operations] containing None, for leaves, or a
                     randomly chosen operation, for other nodes
    :attr children: Union[None, List['Node']] representing the node's children
                    list
    :meth __init__(value, operation, children): constructor for Node class
    """

    def __init__(
            self,
            pos: Union[None, np.int_] = None,
            value: Union[None, np.float64] = None,
            operation: Union[None, Operations] = None,
            children: Union[None, List[ForwardRef("Node")]] = None,
    ) -> None:
        """
        :attr value: Union[None, np.float64] representing the threshold, for
                        leaves, or the computed result, for other nodes
        :attr operation: Union[None, Operations] containing None, for leaves,
                            or a randomly chosen operation, for other nodes
        :attr children: Union[None, List['Node']] representing the node's
                        children list
        """
        self.pos = pos
        self.value = value
        self.operation = operation
        self.children = children or []


class Tree(Thread):
    """
    Class for tree that stores root node and max_levels.
    :attr root: Node representing the root node of the tree
    :attr max_levels: int representing the maximum number of levels that the
                        tree can have
    :meth __init__(thresholds): constructor for Tree class
    :meth _random_tree(max_levels, thresholds): generate a random tree
    :meth _evaluate_tree(node): evaluate the tree
    :meth print_tree(): print the tree
    :meth _print_tree(node, level): print the tree recursively
    """

    def __init__(self, identity: int, name: string) -> None:
        """
        Constructor for Tree class.
        :param identity : identity of tree
        :param name : name of tree
        :return: None
        """
        Thread.__init__(self)
        self.identity = identity
        self.name = name
        self.max_levels = 2
        self.root = None

    def run(self) -> None:
        self.root = self._random_tree(self.max_levels)

    def _random_tree(self, max_levels: int) -> Node:
        """
        Generate a random tree.
        :param max_levels: int representing the maximum number of levels that
                            the tree can have
        :param thresholds: pd.DataFrame object containing the thresholds
        :return: Node object representing the root node of the tree
        """
        if max_levels == 0:
            pos = random.randint(3, 11)
            return Node(pos=pos)

        operation = random.choice(list(Operations))
        num_children = 2
        children = [self._random_tree(max_levels - 1) for _ in range(num_children)]
        return Node(operation=operation, children=children)

    def evaluate_tree(self, node: Node, row: int, thresholds: pd.DataFrame) -> np.float64:
        """
        Evaluate the tree.
        :param node: Node object representing the root node of the tree
        :param row : current row of Excel file
        :param thresholds : current Excel file
        :return: np.float64 representing the computed result
        """
        if not node.children:
            node.value = thresholds.iloc[:, node.pos][row]
            return node.value

        child_values = [self.evaluate_tree(child, row, thresholds) for child in node.children]

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

    def print_tree(self) -> None:
        """
        Print the tree.
        :return: None
        """
        self._print_tree(self.root)

    def _print_tree(self, node: Node, level: int = 0) -> None:
        """
        Print the tree recursively.
        :param node: Node object representing the root node of the tree that is
                        being printed recursively
        :param level: int representing the current level of the tree
        :return: None
        """
        if not node.children:
            logfile = open(self.name + '.txt', 'a+')
            print(" " * (4 * level) + "LEAF.Pos." + str(node.pos), file=logfile)
            print(" " * (4 * level) + "LEAF.Pos." + str(node.pos))
            logfile.close()
        else:
            if level == 0:
                logfile = open(self.name + '.txt', 'a+')
                print(" " * (4 * level) + "ROOT." + str(node.operation), file=logfile)
                print(" " * (4 * level) + "ROOT." + str(node.operation))
                logfile.close()
            else:
                logfile = open(self.name + '.txt', 'a+')
                print(" " * (4 * level) + "NODE." + str(node.operation), file=logfile)
                print(" " * (4 * level) + "NODE." + str(node.operation))
                logfile.close()
            for child in node.children:
                self._print_tree(child, level + 1)


class Utils:
    """
    Class for utility functions.
    :meth create_human_readable_inputs(data): create human readable inputs
    """

    @staticmethod
    def create_human_readable_inputs(data: Data) -> None:
        """
        Create human readable inputs.
        :param data: Data object containing the results and ground truth CSVs
        :return: None
        """
        if os.stat("human_readable_input/results.txt").st_size == 0:
            with open("human_readable_input/results.txt", "w+", encoding="utf-8") as writer:
                writer.write(data.results_df.to_string())

        if os.stat("human_readable_input/ground-truth.txt").st_size == 0:
            with open("human_readable_input/train_ground-truth.txt", "w+", encoding="utf-8") as writer:
                writer.write(data.ground_truth_df.to_string())

    @staticmethod
    def caculate_F_measure(data: Data, threshold: np.float64) -> float:
        pixel = round(threshold * 255)
        f_measures = data.ground_truth_df[['Var' + str(pixel)]]
        length = len(f_measures['Var' + str(pixel)])
        sum = 0
        for i in range(0, length):
            sum += f_measures['Var' + str(pixel)][i]
        return sum / length

    @staticmethod
    def get_f_measure(tree: Tree, data: Data) -> float:
        length = len(data.results_df)
        sum = 0
        for i in range(0, length):
            threshold_value = tree.evaluate_tree(tree.root, i, data.ground_truth_df)
            pixel_value = round(threshold_value * 255)
            f_measures = data.ground_truth_df[['Var' + str(pixel_value)]]
            sum += f_measures['Var' + str(pixel_value)][i]
        return sum / length


def side_thread(tree: Tree, index_parent, index, arr):
    dir_list = os.listdir(LOCAL_FOLDER)
    number_file = len(dir_list)

    if index == 0:
        start = round(index * number_file / 4)
    else:
        start = round(index * number_file / 4) + 1
    end = round(min((index + 1) * number_file / 4, number_file))

    for i in range(start, end):
        excel = "local/train/" + dir_list[i]
        result = pd.read_csv(excel)
        length = len(result)
        pixel_value = result.iloc[:, 0]
        pixel_class = result.iloc[:, 1]
        for j in range(0, length):
            threshold_value = tree.evaluate_tree(tree.root, j, result)
            if threshold_value < pixel_value[j] and pixel_class[j] == 0:
                arr[0] = arr[0] + 1
            elif threshold_value < pixel_value[j] and pixel_class[j] == 1:
                arr[1] = arr[1] + 1
            elif threshold_value > pixel_value[j] and pixel_class[j] == 1:
                arr[2] = arr[2] + 1
            elif threshold_value > pixel_value[j] and pixel_class[j] == 0:
                arr[3] = arr[3] + 1


def thread_calculate_F_measure(tree: Tree, id: int, f_measure):
    list_of_thread = []
    list_of_arr = []
    for i in range(4):
        arr = [0 for i in range(4)]
        list_of_arr.append(arr)

    for i in range(4):
        t0 = Thread(target=side_thread, args=(tree, id, i, list_of_arr[i]))
        list_of_thread.append(t0)

    for i in range(4):
        list_of_thread[i].start()

    for i in range(4):
        list_of_thread[i].join()

    count_TP = 0
    count_FP = 0
    count_TN = 0
    count_FN = 0
    for i in range(4):
        count_TP = count_TP + list_of_arr[i][0]
        count_FP = count_FP + list_of_arr[i][1]
        count_TN = count_TN + list_of_arr[i][2]
        count_FN = count_FN + list_of_arr[i][3]
    f_measure[id] = count_TP / (count_TP + 0.5 * (count_FP + count_FN))


def parse_file(file_path: str) -> Tree:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    tree = Tree(identity=1, name="ExampleTree")
    stack = []

    for line in lines:
        indent_level = len(re.match(r'\s*', line).group())
        while len(stack) > indent_level:
            stack.pop()

        line = line.strip()

        if line.startswith('ROOT'):
            operation_str = line.split('.')[2]
            operation = Operations[operation_str]
            tree.root = Node(operation=operation)
            stack.append(tree.root)

        elif line.startswith('NODE'):
            operation_str = line.split('.')[2]
            operation = Operations[operation_str]
            new_node = Node(operation=operation)
            current_node = stack[-1]
            current_node.children.append(new_node)
            stack.append(new_node)

        elif line.startswith('LEAF'):
            pos = int(line.split('.')[2])
            new_node = Node(pos=pos)
            current_node = stack[-1]
            current_node.children.append(new_node)

    return tree


def main() -> None:
    """
    Main function that runs the application.
    :return: None
    """
    print("Parsing a tree as an example:")
    
    # Example of filename and usage:
    file_path = 'suise.txt'
    parsed_tree = parse_file(file_path)
    parsed_tree.print_tree()


if __name__ == "__main__":
    main()
