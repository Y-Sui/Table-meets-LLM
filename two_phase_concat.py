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


def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", default="./exps/downstream_tasks_20230119_self_augmented_log/main/text003_760816f6-3bd1-4eae-917c-9e933973a6a8/output_dataset/validation_samples.0.jsonl",
                        help="Directory of samples")
    parser.add_argument("--split_dir", default="./exps/downstream_tasks_20230119_self_augmented_log/main/validation.json",
                        help="Directory of split info")
    parser.add_argument("--ref_dir", default="exps/downstream_tasks_20230119_manual_log/0_1_2_3/validation.jsonl",
                        help="Directory of completion containing downstream groundtruth")
    parser.add_argument("--to_dir", default="./exps/downstream_tasks_20230120_self_augmented_p2_log",
                        help="Output directory")
    return parser.parse_args()

def save_unified_jsonl(output_path: str, unified_dict: dict):
    os.makedirs(output_path, exist_ok=True)
    content_list = []
    split = 0
    with open(f"{output_path}/validation.txt", "a", encoding="utf-8") as log_f:
        for index, value in enumerate(unified_dict["content"]):
            info = unified_dict["task"][index] + "|" + unified_dict["objective"][index]
            start = split
            end = split + len(value)
            span_log = [start, end]
            log_f.write(f"{info}, Row: {span_log}\n")
            split = end
            content_list.append(value)
    with open(f"{output_path}/validation.jsonl", "w") as outfile:
        for content in content_list:
            for ele in content:
                outfile.write(json.dumps(ele) + "\n")

if __name__ == '__main__':
    args = get_arguments()

    one_shot_split_info = read_json("./exps/downstream_tasks_20230119_self_augmented_log/main/1-shot.json")
    split_info = read_json(args.split_dir)
    sample_results = read_jsonl(args.src_dir)
    references = read_jsonl(args.ref_dir)

    unified_dict = {"content": [], "task": [], "objective": []}

    # loop for different downstream tasks: totto ...
    for task, objs in split_info.items():
        # loop for different objectives: heur_1 ...
        for obj, line_span in objs.items():
            start_idx, end_idx = line_span
            task_results = sample_results[start_idx: end_idx]
            ref = references[one_shot_split_info[task][0]: one_shot_split_info[task][1]]
            new_split_info = []
            for i in range(len(task_results)):
                prompt = ref[i]["prompt"]
                sample = task_results[i]['samples'][0]  # only has 1 sample
                prefix, suffix = prompt.split('<request>')
                new_prompt = prefix + "<information>\n" + sample + "\n<request>\n" + get_requests(task) + suffix
                completion = ref[i]["completion"]
                new_split_info.append({"prompt": new_prompt, "completion": completion})
            unified_dict["content"].append(new_split_info)
            unified_dict["task"].append(task)
            unified_dict["objective"].append(obj)

    save_unified_jsonl(args.to_dir, unified_dict)

