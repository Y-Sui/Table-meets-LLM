import argparse
import json
import logging
import os
import sys
import numpy as np
from tqdm import tqdm
from typing import List

sys.path.insert(0, "utils")

from datasets import load_dataset
from utils import get_unique_items, load_json, FormLinearize, StructuredDataLinearize
from main.config import DATASETS, get_heuristics, get_requests, get_end_prompt
from transformers import GPT2TokenizerFast

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)


class BabelConvertor:
    def __init__(self):
        self.form_linearizer = None
        self.prompt_input = None
        self.split = None
        self.objective = None
        self.instruct = None
        self.data_type = None
        self.task = None
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.linearizer = StructuredDataLinearize()
        self.linearizer.end_prompt = ""

    def fit_heuristics_constraints(self, sequence, max_token_length=4000):
        tokenized_sequence = self.tokenizer(sequence).input_ids
        if len(tokenized_sequence) < max_token_length:  # the max seq_length of GPT-3
            return True
        else:
            return False

    def get_one_shot_example(self):
        training_set = self.dataset['train']
        idx = np.random.randint(0, len(training_set))
        return training_set[idx]

    def set_split_obj(
        self,
        task: str,
        structured_type: str,
        split: str,
        objective: str,
        instruction: str,
        linearize_func: str,
        use_partition_mark: bool,
        use_format_explanation: bool,
        heuristic: str,
    ):
        self.prompt_input = []  # refresh when init set_split_obj
        self.split = split
        self.objective = objective
        self.task = task
        self.data_type = structured_type
        self.instruct = instruction
        self.linearize_func = linearize_func
        self.use_partition_mark = use_partition_mark
        self.use_format_explanation = use_format_explanation
        self.heuristic = heuristic
        # try:
        #     self.dataset = load_dataset(f"./scripts/dataset_collection/{task}.py", ignore_verifications=True)
        # except:
        #     if self.task.__contains__("multi"):
        #         huggingface_hub = "multi_woz_22"
        #     elif self.task == "sqa":
        #         huggingface_hub = "msr_sqa"
        #     elif self.task == "dart":
        #         huggingface_hub = "dart"
        #     elif self.task.__contains__("formlm"):
        #         huggingface_hub = "./scripts/formlm_dataset_OOF/dev_selected_data0426.json"
        #     else:
        #         huggingface_hub = ""
        #     # load formlm dataset from local
        #     if self.task.__contains__("formlm"):
        #         validation_data = load_json(huggingface_hub)["data"]
        #         self.form_linearizer = FormLinearize()
        #         self.dataset = []
        #         for data in validation_data:
        #             self.dataset.append(data)
        #     else:
        #         self.dataset = load_dataset(huggingface_hub, ignore_verifications=True)
        self.dataset = load_dataset(f"./scripts/dataset_collection/{task}.py")
        self.flag = 0  # no heuristics generation (zero-shot)
        self.request = get_requests(self.task)
        self.end_prompt = "The answer is \n"
        if objective.__contains__("heur"):
            self.request = get_heuristics(self.data_type)[objective]
            self.end_prompt = "The structural information is \n"
            self.flag = 1

    def retrieve_sample_list(self):
        dict = {
            "feverous": self.retrieve_feverous,
            "hybridqa": self.retrieve_hybridqa,
            "totto": self.retrieve_totto,
            "tabfact": self.retrieve_tabfact,
            "sqa": self.retrieve_sqa,
        }
        return dict[self.task]()

    def retrieve_feverous(self):
        def to_linearized_data(_example):
            data = {
                "title": "",
                "context": _example["context"],
                "table": {
                    "header": _example['table']['header'][0],
                    "rows": _example['table']['rows'][0],
                    "caption": "",
                },
            }
            ret = self.linearizer.retrieve_linear_function(
                self.linearize_func,
                self.use_partition_mark,
                self.use_format_explanation,
                False,
                data,
            )
            return ret

        oneshot_pool = []

        if 'oneshot' in self.objective:
            while len(oneshot_pool) < 512:
                oneshot_example = self.get_one_shot_example()
                try:
                    oneshot_prompt = to_linearized_data(oneshot_example)
                except:
                    continue
                label = "0" if oneshot_example["label"] == "REFUTES" else "1"
                oneshot_prompt = (
                    "<example>\n"
                    + oneshot_prompt
                    + "<statement>\n"
                    + oneshot_example["statement"]
                    + "\n"
                    + self.end_prompt
                    + label
                    + "\n</example>"
                )
                if self.fit_heuristics_constraints(
                    oneshot_prompt, max_token_length=1024
                ):
                    oneshot_pool.append(oneshot_prompt)

                    if len(oneshot_pool) % 32 == 0:
                        print('-', end="", flush=True)

        if self.heuristic != None:
            self.request = get_heuristics(self.data_type, context_type="statement")[
                self.heuristic
            ]
            self.end_prompt = get_end_prompt()[self.heuristic]

        for example in self.dataset[self.split]:
            oneshot_prompt = ""
            if self.split != 'train' and 'oneshot' in self.objective:
                idx = np.random.randint(0, len(oneshot_pool))
                oneshot_prompt = oneshot_pool[idx] + "\n"

            content = {}
            # Scrape the desired content from the example
            label = example["label"]
            statement = example["statement"] + "\n"
            # header = "|".join(example["table"]["header"][0]) + "\n"
            # cells = []
            # for i in range(len(example["table"]["rows"][0])):
            #     cells.append("|".join(example["table"]["rows"][0][i]) + "\n")
            # cells = "".join(cells) + "\n"
            # table_info = header + cells

            try:
                table_info = to_linearized_data(example)
            except:
                continue
            content["prompt"] = (
                self.instruct
                + table_info
                + "<request>\n"
                + self.request
                + "<statement>\n"
                + statement
                + self.end_prompt
            )
            content["completion"] = "0" if label == "REFUTES" else "1"
            if (
                self.fit_heuristics_constraints(
                    "".join(content.values()), 4000 - 1024 - 500
                )
                is False
            ):
                continue

            # content["prompt"] = oneshot_prompt + content["prompt"]
            content["example"] = oneshot_prompt
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_hybridqa(self):
        def to_linearized_data(_example):
            data = {
                "title": "",
                "context": [_example["context"], _example["passage"]],
                "table": {
                    "header": _example['table']['header'],
                    "rows": _example['table']['rows'],
                    "caption": "",
                },
            }
            ret = self.linearizer.retrieve_linear_function(
                self.linearize_func,
                self.use_partition_mark,
                self.use_format_explanation,
                False,
                data,
            )
            return ret

        oneshot_pool = []

        if 'oneshot' in self.objective:
            while len(oneshot_pool) < 512:  # this is a bit slow
                oneshot_example = self.get_one_shot_example()
                if oneshot_example["table"] is None:
                    continue

                oneshot_prompt = to_linearized_data(oneshot_example)
                oneshot_prompt = (
                    "<example>\n"
                    + oneshot_prompt
                    + "<question>\n"
                    + oneshot_example["question"]
                    + "\n"
                    + self.end_prompt
                    + oneshot_example["answer_text"]
                    + "\n</example>"
                )
                if self.fit_heuristics_constraints(
                    oneshot_prompt, max_token_length=1024
                ):
                    oneshot_pool.append(oneshot_prompt)

                    if len(oneshot_pool) % 32 == 0:
                        print('-', end="", flush=True)

        if self.heuristic != None:
            self.request = get_heuristics(self.data_type, context_type="question")[
                self.heuristic
            ]
            self.end_prompt = get_end_prompt()[self.heuristic]

        for example in tqdm(self.dataset[self.split], leave=False):
            oneshot_prompt = ""
            if self.split != 'train' and 'oneshot' in self.objective:
                idx = np.random.randint(0, len(oneshot_pool))
                oneshot_prompt = oneshot_pool[idx] + "\n"

            content = {}
            table = example["table"]
            if table is None:
                continue

            table_info = to_linearized_data(example)
            label = example["answer_text"]
            content["prompt"] = (
                self.instruct
                + table_info
                + "<request>\n"
                + self.request
                + "<question>\n"
                + example["question"]
                + "\n"
                + self.end_prompt
            )

            content["completion"] = label
            if (
                self.fit_heuristics_constraints(
                    "".join(content.values()), 4000 - 1024 - 500
                )
                is False
            ):
                continue

            # content["prompt"] = oneshot_prompt + content["prompt"]
            content["example"] = oneshot_prompt
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_sqa(self):
        def to_linearized_data(_example):
            data = {
                "title": "",
                "context": "",
                "table": {
                    "header": _example['table_header'],
                    "rows": _example['table_data'],
                    "caption": "",
                },
            }
            ret = self.linearizer.retrieve_linear_function(
                self.linearize_func,
                self.use_partition_mark,
                self.use_format_explanation,
                False,
                data,
            )
            return ret

        oneshot_pool = []

        if 'oneshot' in self.objective:
            while len(oneshot_pool) < 512:
                oneshot_example = self.get_one_shot_example()

                oneshot_prompt = to_linearized_data(oneshot_example)
                oneshot_prompt = (
                    "<example>\n"
                    + oneshot_prompt
                    + "<question>\n"
                    + oneshot_example["question"]
                    + "\n"
                    + self.end_prompt
                    + "|".join(oneshot_example["answer_text"])
                    + "\n</example>"
                )
                if self.fit_heuristics_constraints(
                    oneshot_prompt, max_token_length=1024
                ):
                    oneshot_pool.append(oneshot_prompt)

                    # if len(oneshot_pool) % 32 == 0:
                    #     print('-', end="", flush=True)

        if self.heuristic != None:
            self.request = get_heuristics(self.data_type, context_type="question")[
                self.heuristic
            ]
            self.end_prompt = get_end_prompt()[self.heuristic]

        for example in tqdm(self.dataset[self.split], leave=False):
            oneshot_prompt = ""
            if self.split != 'train' and 'oneshot' in self.objective:
                idx = np.random.randint(0, len(oneshot_pool))
                oneshot_prompt = oneshot_pool[idx] + "\n"

            content = {}
            # id = example["id"]
            question = example["question"] + "\n"
            answer = "|".join(example["answer_text"])

            table_info = to_linearized_data(example)
            content["prompt"] = (
                self.instruct
                + "\n"
                + table_info
                + "<request>\n"
                + self.request
                + "<question>\n"
                + question
                + self.end_prompt
            )
            content["completion"] = answer
            if (
                self.fit_heuristics_constraints(
                    "".join(content.values()), 4000 - 1024 - 500
                )
                is False
            ):
                continue

            # content["prompt"] = oneshot_prompt + content["prompt"]
            content["example"] = oneshot_prompt
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_tabfact(self):
        def to_linearized_data(_example):
            data = {
                "title": "",
                "context": "",
                "table": {
                    "header": _example['table']['header'],
                    "rows": _example['table']['rows'],
                    "caption": _example['table']['caption'],
                },
            }
            ret = self.linearizer.retrieve_linear_function(
                self.linearize_func,
                self.use_partition_mark,
                self.use_format_explanation,
                False,
                data,
            )
            return ret

        oneshot_pool = []
        if 'oneshot' in self.objective:
            while len(oneshot_pool) < 512:
                oneshot_example = self.get_one_shot_example()
                if oneshot_example['table'] is None:
                    continue
                oneshot_prompt = to_linearized_data(oneshot_example)
                oneshot_prompt = (
                    "<example>\n"
                    + oneshot_prompt
                    + "<statement>\n"
                    + oneshot_example["statement"]
                    + "\n"
                    + self.end_prompt
                    + str(oneshot_example['label'])
                    + "\n</example>"
                )
                if self.fit_heuristics_constraints(
                    oneshot_prompt, max_token_length=1024
                ):
                    oneshot_pool.append(oneshot_prompt)

                    if len(oneshot_pool) % 32 == 0:
                        print('-', end="", flush=True)

        # must after oneshot sampling
        if self.heuristic != None:
            self.request = get_heuristics(self.data_type, context_type="statement")[
                self.heuristic
            ]
            self.end_prompt = get_end_prompt()[self.heuristic]

        for example in tqdm(self.dataset[self.split], leave=False):
            oneshot_prompt = ""
            if self.split != 'train' and 'oneshot' in self.objective:
                idx = np.random.randint(0, len(oneshot_pool))
                oneshot_prompt = oneshot_pool[idx] + "\n"

            content = {}
            input = example["table"]
            statement = example["statement"] + "\n"
            if input is None:
                continue

            table_info = to_linearized_data(example)
            label = example["label"]
            content["prompt"] = (
                self.instruct
                + "\n"
                + table_info
                + "<request>\n"
                + self.request
                + "<statement>\n"
                + statement
                + "\n"
                + self.end_prompt
            )
            content["completion"] = str(label)

            if (
                self.fit_heuristics_constraints(
                    "".join(content.values()), 4000 - 1024 - 500
                )
                is False
            ):
                continue

            # content["prompt"] = oneshot_prompt + content["prompt"]
            content["example"] = oneshot_prompt
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_totto(self):
        def to_linearized_data(_example):
            highlight_idx = _example["highlighted_cells"]
            table = _example["table"]
            highlight_header, highlight_info = [], []

            table_rows = []
            for r_idx in range(len(table)):
                row = []
                for c_idx in range(len(table[r_idx])):
                    # TODO: here a bug occurs that header and table_info don't have same number of columns
                    # Investigation: it seems some table are illegal
                    row.extend(
                        [table[r_idx][c_idx]["value"]]
                        * table[r_idx][c_idx]["column_span"]
                    )

                table_rows.append(row)

            parsed_header = list(x.replace("|", "") for x in table_rows[0])
            try:
                for h_idx in highlight_idx:
                    highlight_info.append(
                        str(h_idx)
                        + ": "
                        + str(parsed_header[h_idx[1]])
                        + "|"
                        + table[h_idx[0]][h_idx[1]]["value"]
                        + "\n"
                    )
            except:
                for h_idx in highlight_idx:
                    highlight_info.append(
                        str(h_idx)
                        + ": "
                        + "-"
                        + "|"
                        + table[h_idx[0]][h_idx[1]]["value"]
                        + "\n"
                    )

            data = {
                "title": _example['table_page_title'],
                "context": "",
                "table": {
                    "header": table_rows[0],
                    "rows": table_rows[1:],
                    "caption": _example['table_section_title'],
                },
            }

            ret = self.linearizer.retrieve_linear_function(
                self.linearize_func,
                self.use_partition_mark,
                self.use_format_explanation,
                False,
                data,
            )

            if len(highlight_info) > 0:
                return ret + "<Highlighted>\n" + "".join(highlight_info)
            else:
                return ret

        oneshot_pool = []
        if 'oneshot' in self.objective:
            while len(oneshot_pool) < 512:
                oneshot_example = self.get_one_shot_example()

                # oneshot_prompt = to_linearized_data(oneshot_example)

                try:
                    oneshot_prompt = to_linearized_data(oneshot_example)
                except:
                    continue

                oneshot_prompt = (
                    "<example>\n"
                    + oneshot_prompt
                    + "\nThe natural language description for each highlighted part of the table:\n"
                    + "\n".join(oneshot_example["final_sentences"])
                    + "\n</example>"
                )
                if self.fit_heuristics_constraints(
                    oneshot_prompt, max_token_length=1024
                ):
                    oneshot_pool.append(oneshot_prompt)

                    if len(oneshot_pool) % 32 == 0:
                        print('-', end="", flush=True)

        if self.heuristic != None:
            self.request = get_heuristics(
                self.data_type, context_type="highlighted part"
            )[self.heuristic]
            self.end_prompt = get_end_prompt()[self.heuristic]
        else:
            self.end_prompt = ""

        for example in tqdm(self.dataset[self.split], leave=False):
            oneshot_prompt = ""
            if self.split != 'train' and 'oneshot' in self.objective:
                idx = np.random.randint(0, len(oneshot_pool))
                oneshot_prompt = oneshot_pool[idx] + "\n"

            content = {}
            # highlight_idx = example["highlighted_cells"]
            final_questions = example["final_sentences"]
            # table_page_title = example["table_page_title"]
            # table_section_title = example["table_section_title"]
            # tables = example["table"]
            # header_info, table_info = [], []
            # highlight_header, highlight_info = [], []
            # for r_idx in range(len(tables)):
            #     # if r_idx != 0:
            #     #     table_info.append("\n")
            #     row = []
            #     for c_idx in range(len(tables[r_idx])):
            #         if r_idx == 0:
            #             if tables[r_idx][c_idx]["column_span"] == 1:
            #                 header_info.append(tables[r_idx][c_idx]["value"])
            #             else:
            #                 for time in range(tables[r_idx][c_idx]["column_span"]):
            #                     header_info.append(tables[r_idx][c_idx]["value"])
            #         else:
            #             row.append(tables[r_idx][c_idx]["value"])
            #
            #     if r_idx != 0:
            #         table_info.append(row)
            #
            # parsed_header = list(x.replace("|", "") for x in header_info)
            # try:
            #     for h_idx in highlight_idx:
            #         highlight_info.append(
            #             str(h_idx) + ": " + str(parsed_header[h_idx[1]]) + "|" + tables[h_idx[0]][h_idx[1]][
            #                 "value"] + "\n")
            # except:
            #     for h_idx in highlight_idx:
            #         highlight_info.append(str(h_idx) + ": " + "-" + "|" + tables[h_idx[0]][h_idx[1]]["value"] + "\n")
            # table_info = "<Page>\n" + table_page_title + "\n" + "<Table>\n" + table_section_title + "\n" + "<Cells>\n" + "".join(
            #     header_info) + "\n" + "".join(table_info) + "\n" + "<Highlighted>\n" + "".join(highlight_info) + "\n"
            try:
                table_info = to_linearized_data(example)
            except:
                continue

            content["prompt"] = (
                self.instruct
                + table_info
                + "\n<request>\n"
                + self.request
                + self.end_prompt
            )
            content["completion"] = "\n".join(final_questions)
            if (
                self.fit_heuristics_constraints(
                    "".join(content.values()), 4000 - 1024 - 500
                )
                is False
            ):
                continue

            # content["prompt"] = oneshot_prompt + content["prompt"]
            content["example"] = oneshot_prompt
            self.prompt_input.append(content)
        return self.prompt_input


def get_keys(dict, value):
    return [k for k, v in dict.items() if value in v]


def save_raw_jsonl(task: str, split_set: str, mode: str):
    try:
        dataset = load_dataset(
            f"./scripts/dataset_collection/{task}.py", ignore_verifications=True
        )
    except FileNotFoundError:
        if task.__contains__("multi"):
            huggingface_hub = "multi_woz_22"
        elif task == "sqa":
            huggingface_hub = "msr_sqa"
        elif task == "dart":
            huggingface_hub = "dart"
        elif task.__contains__("formlm"):
            huggingface_hub = "./scripts/formlm_dataset_OOF/dev_selected_data0426.json"
        else:
            huggingface_hub = ""

        if task.__contains__("formlm"):
            validation_data = load_json(huggingface_hub)["data"]
            dataset = []
            for data in validation_data:
                dataset.append(data)
        else:
            dataset = load_dataset(huggingface_hub, ignore_verifications=True)

    os.makedirs(f"./generated/{task}/raw/", exist_ok=True)
    with open(f"./generated/{task}/raw/{split_set}_{mode}.jsonl", "w") as outfile:
        if task.__contains__("formlm") is False:
            for example in dataset[split_set]:
                outfile.write(json.dumps(example) + "\n")
        else:
            for example in dataset:
                outfile.write(json.dumps(example) + "\n")


def save_jsonl(
    objective: str, task: str, split_set: str, content_list: list, mode: str
):
    os.makedirs(f"./generated/{task}/{objective}/", exist_ok=True)
    with open(
        f"./generated/{task}/{objective}/{split_set}_{mode}.jsonl", "w"
    ) as outfile:
        for content in content_list:
            outfile.write(json.dumps(content) + "\n")


def save_unified_jsonl(output_path: str, unified_dict: dict, mode: str):
    os.makedirs(output_path, exist_ok=True)
    content_list = []
    split = 0
    with open(f"{output_path}/validation_{mode}.txt", "a", encoding="utf-8") as log_f:
        for index, value in enumerate(unified_dict["content"]):
            info = (
                unified_dict["task"][index]
                + "|"
                + unified_dict["choices"][index]
                + "|"
                + unified_dict["objective"][index]
            )
            start = split
            end = split + len(value)
            span_log = [start, end]
            log_f.write(f"{info}, Row: {span_log}\n")
            split = end
            content_list.append(value)
    with open(f"{output_path}/validation_{mode}.jsonl", "w") as outfile:
        for content in content_list:
            for ele in content:
                outfile.write(json.dumps(ele) + "\n")


def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default=["formlm_opt", "formlm_qa", "formlm_block_type"],
        nargs="+",
        help="Please specifiy the task name.",
    )
    # parser.add_argument("--structured_type", default="table", help="Please specify the type of the structured data.", type=str, choices=DATASETS.keys())
    parser.add_argument(
        "--objective",
        default=["zero"],
        nargs="+",
        help="Please specify the parsing objective.",
    )  # choices = ['zero', 'heur_{idx}', 'linear_{idx}', oneshot]
    parser.add_argument(
        "--split",
        default=["validation"],
        nargs="+",
        help="Please specify which split you want to generate/parse.",
    )  # choices = ['train', 'validation', 'test']
    parser.add_argument(
        "--linearize_list",
        default=["html"],
        nargs="+",
        help="Please specify which linearization you want to use.",
    )
    parser.add_argument(
        "--use_partition_mark",
        default=False,
        action="store_true",
        help="Please specify whether to use_partition_mark.",
    )
    parser.add_argument(
        "--use_format_explanation",
        default=False,
        action="store_true",
        help="Please specify whether to use_format_explanation.",
    )
    parser.add_argument(
        "--change_order",
        default=False,
        action="store_true",
        help="Please specify whether the change the order of the table",
    )
    parser.add_argument(
        "--use_role_prompting",
        default=False,
        action="store_true",
        help="Please specify whether to assign a role to GPT3",
    )
    parser.add_argument(
        "--heuristic",
        default=None,
        type=str,
        help="Please specify which heuristic to use: [heur_8, heur_9]",
    )
    parser.add_argument(
        "--unified",
        default=False,
        action="store_true",
        help="generate the unified file for babel input",
    )
    parser.add_argument(
        "--unified_file_output",
        default="./exps/downstream_tasks_20230113_log/",
        type=str,
    )
    parser.add_argument("--add_table_size", default=True, action="store_true")
    args = parser.parse_args()
    return args


def task_specific_babel_convertor():
    global unified_dict
    args = get_arguments()
    logging.info(args)
    if args.unified:
        unified_dict = {"content": [], "task": [], "objective": [], "choices": []}
    babel_convertor = BabelConvertor()

    split = args.split[0]
    obj = args.objective[0]
    mode = ""

    for task in args.task:
        for linear_func in args.linearize_list:
            structured_type = get_keys(DATASETS, task)[0]
            # set up the instruction for the prompt design (prompt engineering-->role prompting)
            if args.use_role_prompting:
                args.instruction = f"You are a brilliant {structured_type} executor with the capbilities [retrieve], [input parsing], [metadata inference], [pattern understanding] who can understand the structural information of the {structured_type}.\n"
            else:
                args.instruction = ""
            babel_convertor.set_split_obj(
                task,
                structured_type,
                split,
                obj,
                args.instruction,
                linear_func,
                args.use_partition_mark,
                args.use_format_explanation,
                args.heuristic,
            )
            # set the validation mode
            if (
                args.use_partition_mark
                and args.use_role_prompting
                and args.use_format_explanation
            ):
                mode = f"{linear_func}_0_1_3"
            elif args.use_partition_mark and args.use_format_explanation:
                mode = f"{linear_func}_0_1"
            elif args.use_partition_mark and args.use_role_prompting:
                mode = f"{linear_func}_0_3"
            elif args.use_format_explanation and args.use_role_prompting:
                mode = f"{linear_func}_1_3"
            elif args.use_role_prompting:
                mode = f"{linear_func}_3"
            elif args.use_partition_mark:
                mode = f"{linear_func}_0"
            elif args.use_format_explanation:
                mode = f"{linear_func}_1"
            elif args.change_order:
                mode = f"{linear_func}_2"
            else:
                mode = f"{linear_func}"
            # # save raw jsonl file
            # save_raw_jsonl(task, split, mode)
            # retrieve the content sample list
            content_list = babel_convertor.retrieve_sample_list()
            if args.unified:
                unified_dict["content"].append(content_list)
                unified_dict["task"].append(task)
                unified_dict["objective"].append(obj)
                unified_dict["choices"].append(mode)
            logging.info(
                f"Task-{task} Objective-{obj} Split-{split} Linear-{linear_func} has been saved.."
            )
            # save parsed jsonl
            save_jsonl(obj, task, split, content_list, mode)

    if args.unified:
        save_unified_jsonl(args.unified_file_output, unified_dict, mode)
        logging.info(f"unified version has been saved")


def main():
    task_specific_babel_convertor()


if __name__ == "__main__":
    main()
