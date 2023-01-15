import argparse
import json
import csv
import os.path
from config import MODELS, DATASETS, TASKS
from sklearn.metrics import accuracy_score


class BenchmarkEvaluator:
    def __init__(self, args):
        self.experiment_path = os.path.join("./exps", args.log_file_path)
        self.log_file_path = args.log_file_path
        self.tasks = TASKS[args.task_group] # task list
        self.models = args.model
        self.task_group = args.task_group
        self.exact_matching_score = []
        for task in self.tasks:
            for model in self.models:
                ground_list, predict_list = self.retrieve_pair(task, model)
                self.exact_matching_score.append(self.exact_match_score(ground_list, predict_list))
        self.save_scores_to_csv()

    def retrieve_pair(self, task_name:str, model_name:str):
        ground_truth_list, predict_list = [], []
        log_files = os.listdir(os.path.join(self.experiment_path, f"{self.task_group}_{task_name}"))
        for log_file in log_files:
            if log_file.__contains__(model_name):
                target_log_file_path = log_file
                continue
        ground_truth_path = os.path.join(self.experiment_path, f"{self.task_group}_{task_name}", f"unified_{task_name}.jsonl")
        predict_value_path = os.path.join(self.experiment_path, f"{self.task_group}_{task_name}", target_log_file_path, "output_dataset", f"unified_{task_name}_samples.0.jsonl")
        with open(ground_truth_path, "r") as json_file:
            for json_line in json_file:
                ground_truth_list.append(json.loads(json_line)["completion"])
        with open(predict_value_path, "r") as json_file:
            for json_line in json_file:
                sample = json.loads(json_line)['samples'][0].split(".")[0]
                predict_list.append(sample)
        return ground_truth_list, predict_list

    def exact_match_score(self, ground_truth, predictions):
        return accuracy_score(ground_truth, predictions)

    def save_scores_to_csv(self):
        os.makedirs(f"./output/evaluation", exist_ok=True)
        with open(f"./output/evaluation/{self.log_file_path}_evaluation.csv", "w", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([f"task name", "model", "score"])
            for i in range(len(self.tasks)):
                for j in range(len(self.models)):
                    if i == 0:
                        writer.writerow([self.tasks[i], self.models[j], self.exact_matching_score[(i+1)*j]])
                    else:
                        writer.writerow([self.tasks[i], self.models[j], self.exact_matching_score[i*j]])

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_group", default="table", type=str, help="Please give the task name")
    parser.add_argument("--log_file_path", default="table_benchmarks_20230114_log", type=str, help="Please indicate the log file path")
    parser.add_argument("--model", default=["chat002", "text003"], nargs="+", help="Please give the model results you want to evaluate")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    BenchmarkEvaluator(args)


if __name__ == "__main__":
    main()