import argparse
import json
import jsonlines
import os
from config import get_requests


def read_jsonl(path):
    rets = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            rets.append(obj)
    return rets


def read_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_jsonl(path, content_list: list):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "validation.jsonl"), "w") as outfile:
        for content in content_list:
            outfile.write(json.dumps(content) + "\n")

def get_task_span(file_path):
    task_file, row_index = [], []
    with open(file_path, "r") as txt_file:
        task_spans = txt_file.readlines() # ../generated/dart/heur_7\validation.jsonl, Row: [31076, 33844]
    for i in range(len(task_spans)):
        if args.job == "zero-shot":
            if not task_spans[i].split(",")[0].__contains__("zero"):
                continue
        elif args.job == "linear":
            if not task_spans[i].split(",")[0].__contains__("linear"):
                continue
        task_file.append(task_spans[i].split(",")[0])
        row_index_number = task_spans[i].split(": ")[-1].replace("[", "").replace("]", "").replace("\n", "").split(",") # list
        row_index.append({"start": int(row_index_number[0]), "end": int(row_index_number[1])})
    return task_file, row_index

def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", default="./exps/downstream_tasks_20230120_self_augmented_p2_log", help="Directory of samples")
    parser.add_argument("--to_dir", default="./exps/downstream_tasks_20230120_self_augmented_p2_revision_log", help="Output directory")
    parser.add_argument("--heur", default="heur_8")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()

    src_dir = args.src_dir
    heur_idx = args.heur
    references = read_jsonl(os.path.join(src_dir, heur_idx, "validation.jsonl"))
    task_file, row_index = get_task_span(os.path.join(src_dir, heur_idx, "validation.txt"))
    model_path = os.listdir(os.path.join(src_dir, heur_idx))
    for i in model_path:
        if i.__contains__("text003"):
            info_path = i
    new_content_info = []
    additional_info = read_jsonl(os.path.join(src_dir, heur_idx, info_path, "output_dataset", "validation_samples.0.jsonl"))
    for task_idx in range(len(task_file)):
        task_name = task_file[task_idx]
        start_idx, end_idx = row_index[task_idx]["start"], row_index[task_idx]["end"]
        ref = references[start_idx: end_idx]
        info = additional_info[start_idx: end_idx]
        for i in range(len(ref)):
            example = ref[i]["example"]
            prompt = ref[i]["prompt"]
            completion = ref[i]["completion"]
            add_info = info[i]['samples'][0]
            prefix, suffix = prompt.split("<request>")
            new_prompt = example + prefix + "<information>\n" + add_info + "\n<request>\n" + get_requests(task_name) + suffix
            new_content_info.append({"prompt": new_prompt, "completion": completion})

    save_jsonl(os.path.join(src_dir, heur_idx), new_content_info)

    # unified_dict = {"content": [], "task": [], "objective": []}

    # # loop for different downstream tasks: totto ...
    # for task, objs in split_info.items():
    #     # loop for different objectives: heur_1 ...
    #     for obj, line_span in objs.items():
    #         start_idx, end_idx = line_span
    #         task_results = sample_results[start_idx: end_idx]
    #         ref = references[one_shot_split_info[task][0]: one_shot_split_info[task][1]]
    #         new_split_info = []
    #         for i in range(len(task_results)):
    #             prompt = ref[i]["prompt"]
    #             sample = task_results[i]['samples'][0]  # only has 1 sample
    #             prefix, suffix = prompt.split('<request>')
    #             new_prompt = prefix + "<information>\n" + sample + "\n<request>\n" + get_requests(task) + suffix
    #             completion = ref[i]["completion"]
    #             new_split_info.append({"prompt": new_prompt, "completion": completion})
    #         unified_dict["content"].append(new_split_info)
    #         unified_dict["task"].append(task)
    #         unified_dict["objective"].append(obj)
    #
    # save_unified_jsonl(args.to_dir, unified_dict)

