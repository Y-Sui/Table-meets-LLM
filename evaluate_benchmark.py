import argparse
import json
import csv
import os.path
import logging
import re

import numpy as np
import pandas as pd

from itertools import groupby
from operator import itemgetter

from config import MODELS, DATASETS, TASKS, LINEARIZE
from sklearn.metrics import accuracy_score

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def exact_match_score(ground_truth, predictions):
    return accuracy_score(ground_truth, predictions)

def flatten_list(list):
    return [item for sublist in list for item in sublist]


class BenchmarkEvaluator:
    def __init__(self, args, line=None):
        self.experiment_path = os.path.join("./exps", args.log_file_path)
        self.log_file_path = args.log_file_path
        self.tasks = TASKS[args.task_group] # task list
        if args.linearize_list is None:
            self.linearizes = LINEARIZE
        else:
            self.linearizes = args.linearize_list
        self.models = args.model
        self.task_group = args.task_group
        self.modes = {} # modes
        self.exact_matching_score = []

        # dump one sample
        self.sample = []

        for task in self.tasks:
            task_mode = []
            for linearize in self.linearizes:
                mode_dict = {
                        "mode": linearize,
                        "start": [],
                        "end": []
                }
                if task != "table_partition":
                    txt_file_name = f"{self.task_group}_{task}"
                else:
                    txt_file_name = "table_partition"
                with open(os.path.join(self.experiment_path, txt_file_name, f"unified_{task}.txt"), "r") as txt_file:
                    task_spans = txt_file.readlines()
                    for line in range(len(task_spans)):
                        mode = task_spans[line].split(",")[0].replace(f"{task}", "").replace(".jsonl", "").replace(f"{''.join(self.models)}", "")
                        row_index_number = task_spans[line].split(": ")[-1].replace("[", "").replace("]", "").replace("\n", "").split(",")  # list
                        if self.task_group == "table":
                            if mode.__contains__(linearize):
                                mode_dict["start"].append(int(row_index_number[0]))
                                mode_dict["end"].append(int(row_index_number[1]))
                        else:
                            mode_dict["start"].append(int(row_index_number[0]))
                            mode_dict["end"].append(int(row_index_number[1]))
                    task_mode.append(mode_dict)
            self.modes[task] = task_mode

        # for task in ["row_retrieval"]:
        for task in self.tasks:
            for model in self.models:
                for mode in self.modes[task]:
                    ground_list, predict_list = self.retrieve_pair(task, model, mode)
                    score = exact_match_score(ground_list, predict_list)
                    self.exact_matching_score.append(score)
                    logging.info(f"{task}_{model}_{mode['mode']}: {score}")
        self.save_scores_to_csv()
        # self.save_sample_x_to_txt()

    def retrieve_pair(self, task_name:str, model_name:str, mode:dict):
        ground_truth_list, predict_list = [], []
        if task_name != "table_partition":
            txt_file_name = f"{self.task_group}_{task_name}"
        else:
            txt_file_name = "table_partition"
        log_files = os.listdir(os.path.join(self.experiment_path, txt_file_name))
        for log_file in log_files:
            if log_file.__contains__(model_name):
                target_log_file_path = log_file
                continue
        ground_truth_path = os.path.join(self.experiment_path, txt_file_name, f"unified_{task_name}.jsonl")
        predict_value_path = os.path.join(self.experiment_path, txt_file_name, target_log_file_path, "output_dataset", f"unified_{task_name}_samples.0.jsonl")
        with open(ground_truth_path, "r") as json_file:
            for idx in range(len(mode["start"])):
                for line in json_file.readlines()[mode["start"][idx]: mode["end"][idx]]:
                    obj = json.loads(line)
                    # dump one sample
                    self.sample.append(f"{mode['mode']}\n" + obj["prompt"] + "\n")
                    ground_truth_list.append(obj["completion"])
        with open(predict_value_path, "r") as json_file:
            for idx in range(len(mode["start"])):
                for line in json_file.readlines()[mode["start"][idx]: mode["end"][idx]]:
                    obj = json.loads(line)
                    if task_name == "cell_lookup":
                        try:
                            pattern = r"(\d+ \| \d+)+"
                            predict_list.append(re.search(pattern, obj['samples'][0].replace("|", " | ")).group())
                        except:
                            predict_list.append(obj['samples'][0].split(".")[0].split("\n")[-1].replace("the answer is: ", ""))
                    elif task_name == "cell_lookup_pos":
                        predict_list.append(obj['samples'][0].replace("\n", "").replace("the answer is: ", "").replace("The answer is: ", "").replace("the answer is:", "").replace("The answer is:", "").replace('"', '').replace(".", ""))
                    elif task_name == "column_retrieval":
                        predict_list.append(obj['samples'][0].replace("\nthe answer is:\n", "").replace("\nThe answer is: ", "").replace("\nThe answer is:\n", "").replace("\nthe answer is: ", "").replace('"', '').replace(".", "").replace("\n", ""))
                    elif task_name == "row_retrieval":
                        predict_list.append(obj['samples'][0].replace("\n", "").replace("The answer is: ", "").replace("The answer is:", "").replace("\nthe answer is: ", ""))
                    elif task_name == "size_detection":
                        try:
                            pattern = r"(\d+ \| \d+)+"
                            predict_list.append(re.search(pattern, obj['samples'][0].replace("|", " | ")).group())
                        except:
                            predict_list.append(obj['samples'][0].replace("\nthe answer is: ", "").replace("\nthe answer is:\n", "").replace("\nThe answer is:", "").replace("\nThe answer is:\n", "").replace("\n", ""))
                    elif task_name == "merged_cell_detection":
                        try:
                            pattern = r"(\d+ \| \d+)+"
                            predict_list.append(re.search(pattern, obj['samples'][0].replace("|", " | ")).group())
                        except:
                            predict_list.append(obj['samples'][0].replace("\nthe answer is: ", "").replace("\nthe answer is:\n", "").replace("\nThe answer is:", "").replace("\nThe answer is:\n", "").replace("\n", "").replace("|", " | "))
                    elif task_name == "table_partition":
                        predict_list.append(obj['samples'][0].replace("Answer: ", "").replace("<tr>", "").replace("</tr>", "").replace("\nThe answer is:\n", "").replace("\n", "").replace("</tr>", ""))
                    else:
                        predict_list.append(obj['samples'][0])

        if task_name == "row_retrieval":
            for i in range(len(predict_list)):
                if predict_list[i].__contains__(ground_truth_list[i]):
                    predict_list[i] = ground_truth_list[i]
        if task_name == "table_partition":
            for i in range(len(ground_truth_list)):
                ground_split = ground_truth_list[i].split("|")
                ground_a, ground_b = ground_split[0], ground_split[1]
                try:
                    pred_split = predict_list[i].split("]|[")
                    pred_a, pred_b = pred_split[0], pred_split[1]
                except:
                    pred_split = predict_list[i]
                    pred_a, pred_b = pred_split, pred_split
                if pred_a.__contains__(ground_a) or pred_b.__contains__(ground_b):
                    predict_list[i] = ground_truth_list[i]

        return ground_truth_list, predict_list

    def save_scores_to_csv(self):
        os.makedirs(f"./output/evaluation", exist_ok=True)
        df = pd.DataFrame()
        with open(f"./output/evaluation/{self.log_file_path}_evaluation.csv", "w", newline='') as csv_file:
            # score_matrix = np.reshape(self.exact_matching_score, (1, len(self.models), len(list(self.modes.values())[0])))
            score_matrix = np.reshape(self.exact_matching_score, (len(self.tasks), len(self.models), len(list(self.modes.values())[0])))
            for i in range(len(self.tasks)):
                for j in range(len(self.models)):
                    for z in range(len(self.modes[self.tasks[i]])):
                        new_df = pd.DataFrame({
                            "task": [self.tasks[i]], "model": [self.models[j]], "mode": [self.modes[self.tasks[i]][z]['mode']], "score": [score_matrix[i, j, z]]
                        })
                        df = pd.concat([df, new_df])
        df.to_csv(f"./output/evaluation/{self.log_file_path}_evaluation.csv", index=False)

    def save_sample_x_to_txt(self):
        with open(f"./output/evaluation/{self.log_file_path}_sample.txt", 'w', encoding="utf-8") as f:
            for item in self.sample:
                f.write(str(item) + '\n')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_group", default="table", type=str, help="Please give the task name")
    parser.add_argument("--log_file_path", default="table_benchmarks_20230119_1_shot_log", type=str, help="Please indicate the log file path")
    parser.add_argument("--model", default=["text003"], nargs="+", help="Please give the model results you want to evaluate")
    parser.add_argument("--linearize_list", nargs="+")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    BenchmarkEvaluator(args)


if __name__ == "__main__":
    main()