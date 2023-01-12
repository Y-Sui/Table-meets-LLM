import argparse
from config import DATASETS
from datasets import load_dataset


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_group", type=str, default="group-1", choices=["group-1", "group-2", "group-3"])
    parser.add_argument("--data_format", type=str, choices=["table", "database", "online_form", "knowledge_graph"], default="table")
    parser.add_argument("--task", type=str, default="cell_lookup")
    args = parser.parse_args()
    args.datasets = DATASETS[args.data_format] # retrieve the associated datasets
    return args

def load_dataset_script():
    dataset_cosql = load_dataset("../scripts/cosql.py")


def main():
    args = get_arguments()

if __name__ == "__main__":
    main()

