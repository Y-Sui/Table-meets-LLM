import argparse
import json
import jsonlines
import os
from main.config import get_requests


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
        task_spans = (
            txt_file.readlines()
        )  # ../generated/dart/heur_7\validation.jsonl, Row: [31076, 33844]
    for i in range(len(task_spans)):
        task_file.append(task_spans[i].split(",")[0].split("|")[0])
        row_index_number = (
            task_spans[i]
            .split(": ")[-1]
            .replace("[", "")
            .replace("]", "")
            .replace("\n", "")
            .split(",")
        )  # list
        row_index.append(
            {"start": int(row_index_number[0]), "end": int(row_index_number[1])}
        )
    return task_file, row_index


def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        default="./exps/downstream_tasks_20230120_self_augmented_p2_log",
        help="Directory of samples",
    )
    parser.add_argument(
        "--to_dir",
        default="./exps/downstream_tasks_20230120_self_augmented_p2_revision_log",
        help="Output directory",
    )
    # parser.add_argument("--heur", default=["heur_0", "heur_1", "heur_2", "heur_3", "heur_4", "heur_5", "heur_6", "heur_8", "heur_9", "heur_10"])
    parser.add_argument("--heur", default=["heur_8", "heur_9", "heur_10"])
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    src_dir = args.src_dir

    for heur_idx in args.heur:
        os.makedirs(os.path.join(args.to_dir, heur_idx), exist_ok=True)

        references = read_jsonl(os.path.join(src_dir, heur_idx, "validation.jsonl"))
        task_file, row_index = get_task_span(
            os.path.join(src_dir, heur_idx, "validation.txt")
        )
        model_path = os.listdir(os.path.join(src_dir, heur_idx))
        for i in model_path:
            if i.__contains__("text003"):
                info_path = i
        new_content_info = []
        additional_info = read_jsonl(
            os.path.join(
                src_dir,
                heur_idx,
                info_path,
                "output_dataset",
                "validation_samples.0.jsonl",
            )
        )
        for task_idx in range(len(task_file)):
            task_name = task_file[task_idx]
            start_idx, end_idx = (
                row_index[task_idx]["start"],
                row_index[task_idx]["end"],
            )
            ref = references[start_idx:end_idx]
            info = additional_info[start_idx:end_idx]
            for i in range(len(ref)):
                example = ref[i]["example"]
                prompt = ref[i]["prompt"]
                completion = ref[i]["completion"]
                add_info = info[i]['samples'][0]
                prefix, suffix = prompt.split("<request>")
                new_prompt = (
                    example
                    + prefix
                    + "<information>"
                    + add_info
                    + "\n<request>\n"
                    + get_requests(task_name)
                )
                new_content_info.append(
                    {"prompt": new_prompt, "completion": completion}
                )

        save_jsonl(os.path.join(args.to_dir, heur_idx), new_content_info)
