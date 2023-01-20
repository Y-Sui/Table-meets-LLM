import argparse
import json
import jsonlines
from config import get_requests


def read_jsonl(path):
    rets = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            rets.append(obj)

    return rets


def read_json(path: str):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", default=None, required=True,
                        help="Directory of samples")
    parser.add_argument("--split_dir", default=None, required=True,
                        help="Directory of split info")
    parser.add_argument("--ref_dir", default=None, required=True,
                        help="Directory of completion containing downstream groundtruth")
    parser.add_argument("--to_dir", default=None, required=True,
                        help="Output directory")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    split_info = read_json(args.split_dir)
    sample_results = read_jsonl(args.src_dir)
    references = read_jsonl(args.ref_dir)

    outputs = []
    new_split_info = []

    current_idx = 0
    # loop for different downstream tasks: totto ...
    for task, objs in split_info.items():
        # loop for different objectives: heur_1 ...
        for obj, line_span in objs.items():
            start_idx, end_idx = line_span
            task_results = sample_results[start_idx: end_idx]
            ref = references[start_idx:end_idx]

            for task_result in task_results:
                prompt = str(task_result['prompt'])
                sample = task_result['samples'][0]  # only has 1 sample

                prefix, suffix = prompt.split('<request>')
                new_prompt = prefix + "<information>\n" + sample + get_requests(task)

                # TODO: split <statement> or <question> from suffix and add it to new_prompt

                print(suffix)
                break
            break
        break
