import json
import argparse
import os

TABLE_TASKS = ["cell_lookup", "column_retrieval", "row_retrieval", "scope_detection", "cell_lookup_pos", "span_detection"]
FORM_TASKS = ["block_dependency", "block_traversal"]

def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=["cell_lookup", "column_retrieval", "row_retrieval", "scope_detection", "cell_lookup_pos", "span_detection", ""], nargs="+",
                        help="Please specifiy the task name.")
    parser.add_argument("--type", default="table")
    args = parser.parse_args()
    return args

def unified_save_jsonl(args):
    if args.type == "table":
        tasks = TABLE_TASKS
        path = "./table"
    elif args.type == "form":
        tasks = FORM_TASKS
        path = "./form"
    file_list = os.listdir(path)
    os.makedirs(f"{path}/unified/", exist_ok=True)
    for task in tasks:
        unified_list = []
        split = 0
        with open(f"{path}/unified/unified_{task}.txt", "a", encoding="utf-8") as log_f:
            for file in file_list:
                if task == "cell_lookup":
                    if "_".join(file.split("_")[1:]).split(".")[0].__contains__(task) and "_".join(file.split("_")[1:]).split(".")[0].__contains__("cell_lookup_pos") is False:
                        with open(f"{path}/{file}", "r") as f:
                            start = split
                            end = split + len(f.readlines())
                            span_log = [start, end]
                            log_f.write(f"{file}, Row: {span_log}\n")
                            split = end
                        with open(f"{path}/{file}", "r") as f:
                            for line in f.readlines():
                                unified_list.append(json.loads(line))
                else:
                    if "_".join(file.split("_")[1:]).split(".")[0].__contains__(task):
                        with open(f"{path}/{file}", "r") as f:
                            start = split
                            end = split + len(f.readlines())
                            span_log = [start, end]
                            log_f.write(f"{file}, Row: {span_log}\n")
                            split = end
                        with open(f"{path}/{file}", "r") as f:
                            for line in f.readlines():
                                unified_list.append(json.loads(line))
        with open(f"{path}/unified/unified_{task}.jsonl", "w", encoding='utf-8') as out_f:
            for content in unified_list:
                out_f.write(json.dumps(content) + "\n")

def main():
    args = get_arguments()
    unified_save_jsonl(args)

if __name__ == "__main__":
    main()