import json
import argparse
import os


def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=["totto", "feverous", "tabfact", "sqa", "hybridqa"], nargs="+",
                        help="Please specifiy the task name.")
    parser.add_argument("--task", default=["cell_lookup", "column_retrieval", "row_retrieval", "scope_detection", "cell_lookup_pos", "span_detection"], nargs="+",
                        help="Please specifiy the task name.")
    # parser.add_argument("--structured_type", default="table", help="Please specify the type of the structured data.", type=str, choices=DATASETS.keys())
    parser.add_argument("--objective", default=["zero"], nargs="+",
                        help="Please specify the parsing objective.")  # choices = ['zero', 'heur_{idx}', 'linear_{idx}']
    parser.add_argument("--split", default=["validation"], nargs="+",
                        help="Please specify which split you want to generate/parse.")  # choices = ['train', 'validation', 'test']
    parser.add_argument("--unified", default=False, action="store_true",
                        help="generate the unified file for babel input")
    parser.add_argument("--unified_file_output", default="./exps/downstream_tasks_20230113_log/", type=str)
    args = parser.parse_args()
    return args

def unified_save_jsonl(args):
    file_list = os.listdir("./table")
    os.makedirs("./table/unified/", exist_ok=True)
    for task in args.task:
        unified_list = []
        split = 0
        with open(f"./table/unified/unified_{task}.txt", "a", encoding="utf-8") as log_f:
            for file in file_list:
                if "_".join(file.split("_")[1:]).split(".")[0] == task:
                    with open(f"./table/{file}", "r") as f:
                        start = split
                        end = split + len(f.readlines())
                        span_log = [start, end]
                        log_f.write(f"{file}, Row: {span_log}\n")
                        split = end
                    with open(f"./table/{file}", "r") as f:
                        for line in f.readlines():
                            unified_list.append(json.loads(line))
        with open(f"./table/unified/unified_{task}.jsonl", "w", encoding='utf-8') as out_f:
            for content in unified_list:
                out_f.write(json.dumps(content) + "\n")

def main():
    args = get_arguments()
    unified_save_jsonl(args)

if __name__ == "__main__":
    main()