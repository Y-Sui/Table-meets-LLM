import argparse
import json
import logging
import os
import random
import sys
from random import sample

import numpy as np
from transformers import GPT2TokenizerFast

from unified_babel_convertor import BabelConvertor

sys.path.insert(0, "utils")

from utils import FormLinearize, StructuredDataLinearize
from config import DATASETS

# from tprompt.dte.embedding import DTEEmbedding
# from tprompt.dte.generator import generate_embeddings
# from tprompt.dte.download import download_dte
# from tprompt.dte.retriever import retrieve

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class DteFewShotGenerator:
    def __init__(self, top_K):
        # model_path = download_dte(use_cached=True)
        # self.dte = DTEEmbedding(os.path.join(model_path, "model.config"))
        self.train_split = {}
        self.validation_split = {}
        self.topK = top_K

    def generate_samples_embedding(self, Q, A):
        return generate_embeddings(self.dte, Q, A)

    def retrieve_few_shot(self):
        random_sample_pair = random.sample(list(enumerate(self.train_split["Q"])), k=150)
        train_split_q, train_split_a = [], []
        for idx, val in random_sample_pair:
            train_split_q.append(val)
            train_split_a.append(self.train_split["A"][idx])
        training_split_embedding = self.generate_samples_embedding(train_split_q, train_split_a)
        validation_split_embedding = self.generate_samples_embedding(self.validation_split["Q"],
                                                                     self.validation_split["A"])
        return retrieve(validation_split_embedding, training_split_embedding, batch_size=16, topK=self.topK,
                        instruct="", prompt_delimiter="input: ", completion_delimiter="output: ")

    def generate_few_shot_examples(self, babel_format_input: list):
        train_Q, train_A, validation_Q, validation_A = [], [], [], []
        for line in babel_format_input:
            train_Q.append(line["prompt"])
            train_A.append(line["completion"])
        for line in babel_format_input:
            validation_Q.append(line["prompt"])
            validation_A.append(line["completion"])
        self.train_split = {"Q": train_Q, "A": train_A}
        self.validation_split = {"Q": validation_Q, "A": validation_A}
        few_shot_examples = self.retrieve_few_shot()
        return few_shot_examples


class BabelBenchmarkGenerator:
    def __init__(self):
        self.babel_convertor = BabelConvertor()
        self.dte_few_shot_generator = DteFewShotGenerator(top_K=2)  # 2 few-shot
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.linearize = StructuredDataLinearize()

    def fit_heuristics_constraints(self, sequence):
        tokenized_sequence = self.tokenizer(sequence).input_ids
        if len(tokenized_sequence) < 3500:  # the max seq_length of GPT-3
            return True
        else:
            return False


def get_keys(dict, value):
    return [k for k, v in dict.items() if value in v]


def retrieve_unique_cell_element(lst: list):
    unique_elements = set([num for row in lst for num in row])
    random_element = random.choice(list(unique_elements))
    for row_index, row in enumerate(lst):
        for col_index, element in enumerate(row):
            if element == random_element:
                return row_index, col_index, random_element


def retrieve_unique_row_element(lst: list):
    random_row_column = random.choice(lst)
    for row_index, row in enumerate(lst):
        if row == random_row_column:
            return row_index, row


def retrieve_unique_column_span(lst: list):
    pair = []
    for j, cell in enumerate(lst[0]):  # only consider the header (field)
        if cell['is_header'] is True and cell['column_span'] > 1:
            pair.append((j, cell['value']))
    return pair if pair != [] else 'None'


def retrieve_swap_tables(lst: list, start_index: int, end_index: int):
    lst[:, [start_index, end_index]] = lst[:, [end_index, start_index]]
    return lst


class DataRetrievalGenerator(BabelBenchmarkGenerator):
    def __init__(self):
        super(DataRetrievalGenerator, self).__init__()
        self.table_datasets_list = DATASETS["table"]
        self.form_datasets_list = DATASETS["form"]
        self.split = "validation"


class InputPartitionGenerator(BabelBenchmarkGenerator):
    def __init__(self):
        super(InputPartitionGenerator, self).__init__()


def save_table_jsonl(dataset_name, task, mode, content_list: list):
    """
    save as jsonl file
    """
    os.makedirs(f"./generated/benchmark/table/", exist_ok=True)
    with open(f"./generated/benchmark/table/{dataset_name}_{task}_{mode}.jsonl", "w") as outfile:
        for content in content_list:
            outfile.write(json.dumps(content) + "\n")


def save_form_jsonl(dataset_name, task, mode, content_list: list):
    """
    save as jsonl file
    """
    os.makedirs(f"./generated/benchmark/form/", exist_ok=True)
    with open(f"./generated/benchmark/form/{dataset_name}_{task}_{mode}.jsonl", "w") as outfile:
        for content in content_list:
            outfile.write(json.dumps(content) + "\n")


class TableDataRetrievalGenerator(DataRetrievalGenerator):
    def __init__(self, args):
        super(TableDataRetrievalGenerator, self).__init__()
        if args.dataset is not None:
            self.table_datasets_list = args.dataset
        self.linearize_list = args.linearize_list
        self.task = None
        self.request = None
        self.objective = args.objective  # ["zero-shot", "1-shot", "few-shot"]

        # linearize
        self.use_partition_mark = args.use_partition_mark  # mark as 0
        self.use_format_explanation = args.use_format_explanation  # mark as 1
        self.change_order = args.change_order  # mark as 2
        self.use_role_prompting = args.use_role_prompting  # mark as 3

        # random sample
        self.random_samples = 300

        self.structured_type = "table"
        self.instruction = f"You are a brilliant {self.structured_type} executor with the capabilities of {self.structured_type} partation, {self.structured_type} parsing, {self.structured_type} table search/retrieval, and table operation/manipulation. You can solve any tasks related to {self.structured_type}.\n"
        self.end_prompt = "The answer is "
        for objective in self.objective:  # ["zero-shot", "1-shot", "few-shot"]
            for table_dataset in self.table_datasets_list:  # ["tabfact", "sqa", "hybridqa", "feverous", "totto"]
                self.babel_convertor.set_split_obj(table_dataset, self.structured_type, self.split, objective, self.instruction)
                self.dataset = self.babel_convertor.dataset
                self.dataset_name = table_dataset
                self.specific_objective = objective
                # benchmark tasks
                self.cell_lookup_pair = []
                self.cell_lookup_pos_pair = []
                self.row_pair = []
                self.column_pair = []
                self.size_pair = []
                self.column_span_pair = []
                self.table_partition = []
                self.table_transpose = []
                self.column_swap = []
                for linearize in self.linearize_list:
                    self.linearize_function = linearize
                    # for saving as jsonl
                    if self.use_partition_mark and self.use_role_prompting and self.use_format_explanation:
                        self.mode = f"{self.linearize_function}_0_1_3"
                    elif self.use_partition_mark and self.use_format_explanation:
                        self.mode = f"{self.linearize_function}_0_1"
                    elif self.use_role_prompting:
                        self.mode = f"{self.linearize_function}_3"
                    elif self.use_partition_mark:
                        self.mode = f"{self.linearize_function}_0"
                    elif self.use_format_explanation:
                        self.mode = f"{self.linearize_function}_1"
                    elif self.change_order:
                        self.mode = f"{self.linearize_function}_2"
                    else:
                        self.mode = f"{self.linearize_function}"
                    self.retrieve_sample_list()

    def retrieve_sample_list(self):
        dict = {
            "tabfact": self.retrieval_tabfact_info,
            "sqa": self.retrieval_sqa_info,
            "feverous": self.retrieval_feverous_info,
            "hybridqa": self.retrieval_hybridqa_info,
            "totto": self.retrieval_totoo_info,
        }
        return dict[self.dataset_name]()

    def retrieve_sample_0(self, dict):
        return "Example_0:\n\n" + dict["prompt"] + "\n" + dict["completion"] + "\n"

    def append_sample(self, dict, example):
        dict['prompt'] = example + dict['prompt']
        return dict

    def random_sampling(self, list):
        if len(list) > self.random_samples:
            return sample(list, self.random_samples)
        else:
            return sample(list, len(list))

    def retrieval_tabfact_info(self):
        for idx, example in enumerate(self.dataset[self.split]):

            # input = example["table"]
            # cells = []
            # header = "|".join(input["header"]) + "\n"
            # for i in range(len(input["rows"])):
            #     cells.append("|".join(input["rows"][i]) + "\n")
            # schema_knowledge = "<header>\n" + header + "<cells>\n" + "".join(cells) + self.end_prompt

            structured_data_dict = {
                "title": "",
                "context": "",
                "table": {
                    "header": example['table']['header'],
                    "rows": example['table']['rows'],
                    "caption": example['table']['caption']
                }
            }
            try:
                schema_knowledge = self.linearize.retrieve_linear_function(self.linearize_function, self.use_partition_mark, self.use_format_explanation, self.change_order, structured_data_dict)
            except:
                continue
            if self.specific_objective == "1-shot":
                if idx == 0:
                    cell_lookup_generation_sample_0 = self.retrieve_sample_0(self.cell_lookup_generation(example['table']['rows'], schema_knowledge))
                    cell_lookup_pos_generation_sample_0 = self.retrieve_sample_0(self.cell_lookup_pos_generation(example['table']['rows'], schema_knowledge))
                    row_pair_generation_sample_0 = self.retrieve_sample_0(self.table_row_retrieval_generation(example['table']['rows'], schema_knowledge))
                    column_pair_generation_sample_0 = self.retrieve_sample_0(self.table_column_retrieval_generation(example['table']['header'], schema_knowledge))
                    size_pair_generation_sample_0 = self.retrieve_sample_0(self.table_size_detection_generation(example["table"]['rows'], schema_knowledge))
                    table_partition_pair_generation_sample_0 = self.retrieve_sample_0(self.table_partition_generation(example["table"]["rows"], schema_knowledge))

                cell_lookup_generation_sample = self.append_sample(self.cell_lookup_generation(example['table']['rows'], schema_knowledge), cell_lookup_generation_sample_0)
                cell_lookup_pos_generation_sample = self.append_sample(self.cell_lookup_pos_generation(example['table']['rows'], schema_knowledge), cell_lookup_pos_generation_sample_0)
                row_pair_generation_sample = self.append_sample(self.table_row_retrieval_generation(example['table']['rows'], schema_knowledge), row_pair_generation_sample_0)
                column_pair_generation_sample = self.append_sample(self.table_column_retrieval_generation(example['table']['header'], schema_knowledge), column_pair_generation_sample_0)
                size_pair_generation_sample = self.append_sample(self.table_size_detection_generation(example["table"]['rows'], schema_knowledge), size_pair_generation_sample_0)
                table_partition_pair_generation_sample = self.append_sample(self.table_partition_generation(example["table"]["rows"], schema_knowledge), table_partition_pair_generation_sample_0)

                # 1-shot
                if self.fit_heuristics_constraints(cell_lookup_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pair.append(cell_lookup_generation_sample)
                if self.fit_heuristics_constraints(cell_lookup_pos_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pos_pair.append(cell_lookup_pos_generation_sample)
                if self.fit_heuristics_constraints(row_pair_generation_sample['prompt']) is False:
                    continue
                self.row_pair.append(row_pair_generation_sample)
                if self.fit_heuristics_constraints(column_pair_generation_sample['prompt']) is False:
                    continue
                self.column_pair.append(column_pair_generation_sample)
                if self.fit_heuristics_constraints(size_pair_generation_sample['prompt']) is False:
                    continue
                self.size_pair.append(size_pair_generation_sample)
                if self.fit_heuristics_constraints(table_partition_pair_generation_sample["prompt"]) is False:
                    continue
                self.table_partition.append(table_partition_pair_generation_sample)

            elif self.specific_objective == "zero-shot":
                # zero-shot
                cell_lookup_generation_sample = self.cell_lookup_generation(example['table']['rows'], schema_knowledge)
                cell_lookup_pos_generation_sample = self.cell_lookup_pos_generation(example['table']['rows'], schema_knowledge)
                row_pair_generation_sample = self.table_row_retrieval_generation(example['table']['rows'], schema_knowledge)
                column_pair_generation_sample = self.table_column_retrieval_generation(example['table']['header'], schema_knowledge)
                size_pair_generation_sample = self.table_size_detection_generation(example["table"]["rows"], schema_knowledge)
                table_partition_pair_generation_sample = self.table_partition_generation(example["table"]["rows"], schema_knowledge)

                if self.fit_heuristics_constraints(cell_lookup_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pair.append(cell_lookup_generation_sample)
                if self.fit_heuristics_constraints(cell_lookup_pos_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pos_pair.append(cell_lookup_pos_generation_sample)
                if self.fit_heuristics_constraints(row_pair_generation_sample['prompt']) is False:
                    continue
                self.row_pair.append(row_pair_generation_sample)
                if self.fit_heuristics_constraints(column_pair_generation_sample['prompt']) is False:
                    continue
                self.column_pair.append(column_pair_generation_sample)
                if self.fit_heuristics_constraints(size_pair_generation_sample['prompt']) is False:
                    continue
                self.size_pair.append(size_pair_generation_sample)
                if self.fit_heuristics_constraints(table_partition_pair_generation_sample["prompt"]) is False:
                    continue
                self.table_partition.append(table_partition_pair_generation_sample)

        if self.specific_objective == "few-shot":
            # few-shot
            self.cell_lookup_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.cell_lookup_pos_pair)
            self.cell_lookup_pos_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.cell_lookup_pos_pair)
            self.row_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.row_pair)
            self.column_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.column_pair)
            self.size_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.size_pair)
            self.table_partition = self.dte_few_shot_generator.generate_few_shot_examples(self.table_partition)

        # save as jsonl (tabfact)
        logging.info(f"{self.dataset_name} tasks {self.specific_objective} datasets have been generated..")
        save_table_jsonl(self.dataset_name, "cell_lookup", self.mode, self.random_sampling(self.cell_lookup_pair))
        save_table_jsonl(self.dataset_name, "cell_lookup_pos", self.mode, self.random_sampling(self.cell_lookup_pos_pair))
        save_table_jsonl(self.dataset_name, "row_retrieval", self.mode, self.random_sampling(self.row_pair))
        save_table_jsonl(self.dataset_name, "column_retrieval", self.mode, self.random_sampling(self.column_pair))
        save_table_jsonl(self.dataset_name, "size_detection", self.mode, self.random_sampling(self.size_pair))
        save_table_jsonl(self.dataset_name, "table_partition", self.mode, self.random_sampling(self.table_partition))

    def retrieval_sqa_info(self):
        for idx, example in enumerate(self.dataset[self.split]):
            # # Scrape the desired content from the example
            # header = "|".join(example["table_header"]) + "\n"
            # cells = []
            # table = example["table_data"]
            # for i in range(len(table)):
            #     cells.append("|".join(table[i]) + "\n")
            # cells = "".join(cells)
            # schema_knowledge = "<header>\n" + header + "<cells>\n" + cells + self.end_prompt
            structured_data_dict = {
                "title": "",
                "context": "",
                "table": {
                    "header": example['table_header'],
                    "rows": example['table_data'],
                    "caption": ""
                }
            }
            try:
                schema_knowledge = self.linearize.retrieve_linear_function(self.linearize_function, self.use_partition_mark, self.use_format_explanation, self.change_order, structured_data_dict)
            except:
                continue
            if self.specific_objective == "1-shot":
                if idx == 0:
                    cell_lookup_generation_sample_0 = self.retrieve_sample_0(self.cell_lookup_generation(example['table_data'], schema_knowledge))
                    cell_lookup_pos_generation_sample_0 = self.retrieve_sample_0(self.cell_lookup_pos_generation(example['table_data'], schema_knowledge))
                    row_pair_generation_sample_0 = self.retrieve_sample_0(self.table_row_retrieval_generation(example['table_data'], schema_knowledge))
                    column_pair_generation_sample_0 = self.retrieve_sample_0(self.table_column_retrieval_generation(example['table_header'], schema_knowledge))
                    size_pair_generation_sample_0 = self.retrieve_sample_0(self.table_size_detection_generation(example['table_data'], schema_knowledge))
                    table_partition_pair_generation_sample_0 = self.retrieve_sample_0(self.table_partition_generation(example["table_data"], schema_knowledge))

                cell_lookup_generation_sample = self.append_sample(self.cell_lookup_generation(example['table_data'], schema_knowledge), cell_lookup_generation_sample_0)
                cell_lookup_pos_generation_sample = self.append_sample(self.cell_lookup_pos_generation(example['table_data'], schema_knowledge), cell_lookup_pos_generation_sample_0)
                row_pair_generation_sample = self.append_sample(self.table_row_retrieval_generation(example['table_data'], schema_knowledge), row_pair_generation_sample_0)
                column_pair_generation_sample = self.append_sample(self.table_column_retrieval_generation(example['table_header'], schema_knowledge), column_pair_generation_sample_0)
                size_pair_generation_sample = self.append_sample(self.table_size_detection_generation(example['table_data'], schema_knowledge), size_pair_generation_sample_0)
                table_partition_pair_generation_sample = self.append_sample(self.table_partition_generation(example["table_data"], schema_knowledge), table_partition_pair_generation_sample_0)

                # 1-shot
                if self.fit_heuristics_constraints(cell_lookup_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pair.append(cell_lookup_generation_sample)
                if self.fit_heuristics_constraints(cell_lookup_pos_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pos_pair.append(cell_lookup_pos_generation_sample)
                if self.fit_heuristics_constraints(row_pair_generation_sample['prompt']) is False:
                    continue
                self.row_pair.append(row_pair_generation_sample)
                if self.fit_heuristics_constraints(column_pair_generation_sample['prompt']) is False:
                    continue
                self.column_pair.append(column_pair_generation_sample)
                if self.fit_heuristics_constraints(size_pair_generation_sample['prompt']) is False:
                    continue
                self.size_pair.append(size_pair_generation_sample)
                if self.fit_heuristics_constraints(table_partition_pair_generation_sample["prompt"]) is False:
                    continue
                self.table_partition.append(table_partition_pair_generation_sample)

            elif self.specific_objective == "zero-shot":
                # zero-shot
                cell_lookup_generation_sample = self.cell_lookup_generation(example['table_data'], schema_knowledge)
                cell_lookup_pos_generation_sample = self.cell_lookup_pos_generation(example['table_data'], schema_knowledge)
                row_pair_generation_sample = self.table_row_retrieval_generation(example['table_data'], schema_knowledge)
                column_pair_generation_sample = self.table_column_retrieval_generation(example['table_header'], schema_knowledge)
                size_pair_generation_sample = self.table_size_detection_generation(example['table_data'], schema_knowledge)
                table_partition_pair_generation_sample = self.table_partition_generation(example['table_data'], schema_knowledge)

                if self.fit_heuristics_constraints(cell_lookup_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pair.append(cell_lookup_generation_sample)
                if self.fit_heuristics_constraints(cell_lookup_pos_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pos_pair.append(cell_lookup_pos_generation_sample)
                if self.fit_heuristics_constraints(row_pair_generation_sample['prompt']) is False:
                    continue
                self.row_pair.append(row_pair_generation_sample)
                if self.fit_heuristics_constraints(column_pair_generation_sample['prompt']) is False:
                    continue
                self.column_pair.append(column_pair_generation_sample)
                if self.fit_heuristics_constraints(size_pair_generation_sample['prompt']) is False:
                    continue
                self.size_pair.append(size_pair_generation_sample)
                if self.fit_heuristics_constraints(table_partition_pair_generation_sample['prompt']) is False:
                    continue
                self.table_partition.append(table_partition_pair_generation_sample)

        if self.specific_objective == "few-shot":
            # few-shot
            self.cell_lookup_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.cell_lookup_pos_pair)
            self.cell_lookup_pos_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.cell_lookup_pos_pair)
            self.row_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.row_pair)
            self.column_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.column_pair)
            self.size_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.size_pair)
            self.table_partition = self.dte_few_shot_generator.generate_few_shot_examples(self.table_partition)

        # save as jsonl (sqa)
        logging.info(f"{self.dataset_name} tasks {self.specific_objective} datasets have been generated..")
        save_table_jsonl(self.dataset_name, "cell_lookup", self.mode, self.random_sampling(self.cell_lookup_pair))
        save_table_jsonl(self.dataset_name, "cell_lookup_pos", self.mode, self.random_sampling(self.cell_lookup_pos_pair))
        save_table_jsonl(self.dataset_name, "row_retrieval", self.mode, self.random_sampling(self.row_pair))
        save_table_jsonl(self.dataset_name, "column_retrieval", self.mode, self.random_sampling(self.column_pair))
        save_table_jsonl(self.dataset_name, "size_detection", self.mode, self.random_sampling(self.size_pair))
        save_table_jsonl(self.dataset_name, "table_partition", self.mode, self.random_sampling(self.table_partition))

    def retrieval_hybridqa_info(self):
        for idx, example in enumerate(self.dataset[self.split]):
            # table = example["table"]
            # cells = []
            # header = "|".join(table["header"][:-1]) + "\n"
            # for i in range(len(table["rows"])):
            #     table["rows"][i] = table["rows"][i][:-1]
            #     cells.append("|".join(table["rows"][i]) + "\n")
            # schema_knowledge = "<header>\n" + header + "<cells>\n" + "".join(cells) + self.end_prompt
            structured_data_dict = {
                "title": "",
                "context": example["context"] + example["passage"],
                "table": {
                    "header": example['table']['header'],
                    "rows": example['table']['rows'],
                    "caption": ""
                }
            }
            try:
                schema_knowledge = self.linearize.retrieve_linear_function(self.linearize_function, self.use_partition_mark, self.use_format_explanation, self.change_order, structured_data_dict)
            except:
                continue
            if self.specific_objective == "1-shot":
                if idx == 0:
                    cell_lookup_generation_sample_0 = self.retrieve_sample_0(self.cell_lookup_generation(example['table']['rows'], schema_knowledge))
                    cell_lookup_pos_generation_sample_0 = self.retrieve_sample_0(self.cell_lookup_pos_generation(example['table']['rows'], schema_knowledge))
                    row_pair_generation_sample_0 = self.retrieve_sample_0(self.table_row_retrieval_generation(example['table']['rows'], schema_knowledge))
                    column_pair_generation_sample_0 = self.retrieve_sample_0(self.table_column_retrieval_generation(example['table']['header'], schema_knowledge))
                    size_pair_generation_sample_0 = self.retrieve_sample_0(self.table_size_detection_generation(example["table"]['rows'], schema_knowledge))
                    table_partition_pair_generation_sample_0 = self.retrieve_sample_0(self.table_partition_generation(example["table"]["rows"], schema_knowledge))

                cell_lookup_generation_sample = self.append_sample(self.cell_lookup_generation(example['table']['rows'], schema_knowledge), cell_lookup_generation_sample_0)
                cell_lookup_pos_generation_sample = self.append_sample(self.cell_lookup_pos_generation(example['table']['rows'], schema_knowledge), cell_lookup_pos_generation_sample_0)
                row_pair_generation_sample = self.append_sample(self.table_row_retrieval_generation(example['table']['rows'], schema_knowledge), row_pair_generation_sample_0)
                column_pair_generation_sample = self.append_sample(self.table_column_retrieval_generation(example['table']['header'], schema_knowledge), column_pair_generation_sample_0)
                size_pair_generation_sample = self.append_sample(self.table_size_detection_generation(example["table"]['rows'], schema_knowledge), size_pair_generation_sample_0)
                table_partition_pair_generation_sample = self.append_sample(self.table_partition_generation(example["table"]['rows'], schema_knowledge), table_partition_pair_generation_sample_0)

                # 1-shot
                if self.fit_heuristics_constraints(cell_lookup_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pair.append(cell_lookup_generation_sample)
                if self.fit_heuristics_constraints(cell_lookup_pos_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pos_pair.append(cell_lookup_pos_generation_sample)
                if self.fit_heuristics_constraints(row_pair_generation_sample['prompt']) is False:
                    continue
                self.row_pair.append(row_pair_generation_sample)
                if self.fit_heuristics_constraints(column_pair_generation_sample['prompt']) is False:
                    continue
                self.column_pair.append(column_pair_generation_sample)
                if self.fit_heuristics_constraints(size_pair_generation_sample['prompt']) is False:
                    continue
                self.size_pair.append(size_pair_generation_sample)
                if self.fit_heuristics_constraints(table_partition_pair_generation_sample['prompt']) is False:
                    continue
                self.table_partition.append(table_partition_pair_generation_sample)

            elif self.specific_objective == "zero-shot":
                # zero-shot
                cell_lookup_generation_sample = self.cell_lookup_generation(example['table']['rows'], schema_knowledge)
                cell_lookup_pos_generation_sample = self.cell_lookup_pos_generation(example['table']['rows'], schema_knowledge)
                row_pair_generation_sample = self.table_row_retrieval_generation(example['table']['rows'], schema_knowledge)
                column_pair_generation_sample = self.table_column_retrieval_generation(example['table']['header'], schema_knowledge)
                size_pair_generation_sample = self.table_size_detection_generation(example["table"]['rows'], schema_knowledge)
                table_partition_pair_generation_sample = self.table_partition_generation(example["table"]['rows'], schema_knowledge)

                if self.fit_heuristics_constraints(cell_lookup_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pair.append(cell_lookup_generation_sample)
                if self.fit_heuristics_constraints(cell_lookup_pos_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pos_pair.append(cell_lookup_pos_generation_sample)
                if self.fit_heuristics_constraints(row_pair_generation_sample['prompt']) is False:
                    continue
                self.row_pair.append(row_pair_generation_sample)
                if self.fit_heuristics_constraints(column_pair_generation_sample['prompt']) is False:
                    continue
                self.column_pair.append(column_pair_generation_sample)
                if self.fit_heuristics_constraints(size_pair_generation_sample['prompt']) is False:
                    continue
                self.size_pair.append(size_pair_generation_sample)
                if self.fit_heuristics_constraints(table_partition_pair_generation_sample['prompt']) is False:
                    continue
                self.table_partition.append(table_partition_pair_generation_sample)

        if self.specific_objective == "few-shot":
            # few-shot
            self.cell_lookup_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.cell_lookup_pos_pair)
            self.cell_lookup_pos_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.cell_lookup_pos_pair)
            self.row_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.row_pair)
            self.column_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.column_pair)
            self.size_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.size_pair)
            self.table_partition = self.dte_few_shot_generator.generate_few_shot_examples(self.table_partition)

        # save as jsonl (hybridqa)
        logging.info(f"{self.dataset_name} tasks {self.specific_objective} datasets have been generated..")
        save_table_jsonl(self.dataset_name, "cell_lookup", self.mode, self.random_sampling(self.cell_lookup_pair))
        save_table_jsonl(self.dataset_name, "cell_lookup_pos", self.mode, self.random_sampling(self.cell_lookup_pos_pair))
        save_table_jsonl(self.dataset_name, "row_retrieval", self.mode, self.random_sampling(self.row_pair))
        save_table_jsonl(self.dataset_name, "column_retrieval", self.mode, self.random_sampling(self.column_pair))
        save_table_jsonl(self.dataset_name, "size_detection", self.mode, self.random_sampling(self.size_pair))
        save_table_jsonl(self.dataset_name, "table_partition", self.mode, self.random_sampling(self.table_partition))

    def retrieval_feverous_info(self):
        for idx, example in enumerate(self.dataset[self.split]):
            # Scrape the desired content from the example
            # header = "|".join(example["table"]["header"][0]) + "\n"
            # cells = []
            # for i in range(len(example["table"]["rows"][0])):
            #     cells.append(" | ".join(example["table"]["rows"][0][i]) + "\n")
            # cells = "".join(cells) + "\n"
            # schema_knowledge = "<header>\n" + header + "<cells>\n" + cells + self.end_prompt
            structured_data_dict = {
                "title": "",
                "context": example["context"],
                "table": {
                    "header": example['table']['header'][0],
                    "rows": example['table']['rows'][0],
                    "caption": ""
                }
            }
            try:
                schema_knowledge = self.linearize.retrieve_linear_function(self.linearize_function, self.use_partition_mark, self.use_format_explanation, self.change_order, structured_data_dict)
            except:
                continue
            if self.specific_objective == "1-shot":
                if idx == 0:
                    cell_lookup_generation_sample_0 = self.retrieve_sample_0(self.cell_lookup_generation(example['table']['rows'][0], schema_knowledge))
                    cell_lookup_pos_generation_sample_0 = self.retrieve_sample_0(self.cell_lookup_pos_generation(example['table']['rows'][0], schema_knowledge))
                    row_pair_generation_sample_0 = self.retrieve_sample_0(self.table_row_retrieval_generation(example['table']['rows'][0], schema_knowledge))
                    column_pair_generation_sample_0 = self.retrieve_sample_0(self.table_column_retrieval_generation(example['table']['header'][0], schema_knowledge))
                    size_pair_generation_sample_0 = self.retrieve_sample_0(self.table_size_detection_generation(example["table"]['rows'][0], schema_knowledge))
                    table_partition_pair_generation_sample_0 = self.retrieve_sample_0(self.table_partition_generation(example["table"]["rows"], schema_knowledge))

                cell_lookup_generation_sample = self.append_sample(self.cell_lookup_generation(example['table']['rows'][0], schema_knowledge), cell_lookup_generation_sample_0)
                cell_lookup_pos_generation_sample = self.append_sample(self.cell_lookup_pos_generation(example['table']['rows'][0], schema_knowledge), cell_lookup_pos_generation_sample_0)
                row_pair_generation_sample = self.append_sample(self.table_row_retrieval_generation(example['table']['rows'][0], schema_knowledge), row_pair_generation_sample_0)
                column_pair_generation_sample = self.append_sample(self.table_column_retrieval_generation(example['table']['header'][0], schema_knowledge), column_pair_generation_sample_0)
                size_pair_generation_sample = self.append_sample(self.table_size_detection_generation(example["table"]['rows'][0], schema_knowledge), size_pair_generation_sample_0)
                table_partition_pair_generation_sample = self.append_sample(self.table_partition_generation(example["table"]['rows'][0], schema_knowledge), table_partition_pair_generation_sample_0)

                # 1-shot
                if self.fit_heuristics_constraints(cell_lookup_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pair.append(cell_lookup_generation_sample)
                if self.fit_heuristics_constraints(cell_lookup_pos_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pos_pair.append(cell_lookup_pos_generation_sample)
                if self.fit_heuristics_constraints(row_pair_generation_sample['prompt']) is False:
                    continue
                self.row_pair.append(row_pair_generation_sample)
                if self.fit_heuristics_constraints(column_pair_generation_sample['prompt']) is False:
                    continue
                self.column_pair.append(column_pair_generation_sample)
                if self.fit_heuristics_constraints(size_pair_generation_sample['prompt']) is False:
                    continue
                self.size_pair.append(size_pair_generation_sample)
                if self.fit_heuristics_constraints(table_partition_pair_generation_sample['prompt']) is False:
                    continue
                self.table_partition.append(table_partition_pair_generation_sample)

            elif self.specific_objective == "zero-shot":
                # zero-shot
                cell_lookup_generation_sample = self.cell_lookup_generation(example['table']['rows'][0], schema_knowledge)
                cell_lookup_pos_generation_sample = self.cell_lookup_pos_generation(example['table']['rows'][0], schema_knowledge)
                row_pair_generation_sample = self.table_row_retrieval_generation(example['table']['rows'][0], schema_knowledge)
                column_pair_generation_sample = self.table_column_retrieval_generation(example['table']['header'][0], schema_knowledge)
                size_pair_generation_sample = self.table_size_detection_generation(example["table"]['rows'][0], schema_knowledge)
                table_partition_pair_generation_sample = self.table_partition_generation(example["table"]['rows'][0], schema_knowledge)

                if self.fit_heuristics_constraints(cell_lookup_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pair.append(cell_lookup_generation_sample)
                if self.fit_heuristics_constraints(cell_lookup_pos_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pos_pair.append(cell_lookup_pos_generation_sample)
                if self.fit_heuristics_constraints(row_pair_generation_sample['prompt']) is False:
                    continue
                self.row_pair.append(row_pair_generation_sample)
                if self.fit_heuristics_constraints(column_pair_generation_sample['prompt']) is False:
                    continue
                self.column_pair.append(column_pair_generation_sample)
                if self.fit_heuristics_constraints(size_pair_generation_sample['prompt']) is False:
                    continue
                self.size_pair.append(size_pair_generation_sample)
                if self.fit_heuristics_constraints(table_partition_pair_generation_sample['prompt']) is False:
                    continue
                self.table_partition.append(table_partition_pair_generation_sample)

        if self.specific_objective == "few-shot":
            # few-shot
            self.cell_lookup_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.cell_lookup_pos_pair)
            self.cell_lookup_pos_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.cell_lookup_pos_pair)
            self.row_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.row_pair)
            self.column_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.column_pair)
            self.size_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.size_pair)
            self.table_partition = self.dte_few_shot_generator.generate_few_shot_examples(self.table_partition)

        # save as jsonl (feverous)
        logging.info(f"{self.dataset_name} tasks {self.specific_objective} datasets have been generated..")
        save_table_jsonl(self.dataset_name, "cell_lookup", self.mode, self.random_sampling(self.cell_lookup_pair))
        save_table_jsonl(self.dataset_name, "cell_lookup_pos", self.mode, self.random_sampling(self.cell_lookup_pos_pair))
        save_table_jsonl(self.dataset_name, "row_retrieval", self.mode, self.random_sampling(self.row_pair))
        save_table_jsonl(self.dataset_name, "column_retrieval", self.mode, self.random_sampling(self.column_pair))
        save_table_jsonl(self.dataset_name, "size_detection", self.mode, self.random_sampling(self.size_pair))
        save_table_jsonl(self.dataset_name, "table_partition", self.mode, self.random_sampling(self.table_partition))

    def retrieval_totoo_info(self):
        for idx, example in enumerate(self.dataset[self.split]):
            highlight_idx = example["highlighted_cells"]
            tables = example["table"]
            header_info, table_info = [], []
            highlight_header, highlight_info = [], []
            for r_idx in range(len(tables)):
                if r_idx != 0:
                    table_info.append("\n")
                for c_idx in range(len(tables[r_idx])):
                    if r_idx == 0:
                        if tables[r_idx][c_idx]["column_span"] == 1:
                            header_info.append(tables[r_idx][c_idx]["value"] + "|")
                        else:
                            for time in range(tables[r_idx][c_idx]["column_span"]):
                                header_info.append(tables[r_idx][c_idx]["value"] + "|")
                    else:
                        table_info.append(tables[r_idx][c_idx]["value"] + "|")
            parsed_header = list(x.replace("|", "") for x in header_info)
            parsed_table = []
            for row_idx in range(len(tables)):
                row_table = []
                for col_idx in range(len(tables[row_idx])):
                    row_table.append(tables[row_idx][col_idx]['value'])
                parsed_table.append(row_table)
            try:
                for h_idx in highlight_idx:
                    highlight_info.append(
                        str(h_idx) + ": " + str(parsed_header[h_idx[1]]) + "|" + tables[h_idx[0]][h_idx[1]][
                            "value"] + "\n")
            except:
                for h_idx in highlight_idx:
                    highlight_info.append(str(h_idx) + ": " + "-" + "|" + tables[h_idx[0]][h_idx[1]]["value"] + "\n")
            # schema_knowledge = "<header>\n" + "".join(header_info) + "<cells>\n" + "".join(table_info) + self.end_prompt
            structured_data_dict = {
                "title": example['table_page_title'] + example['table_section_title'],
                "context": '',
                "table": {
                    "header": parsed_header,
                    "rows": parsed_table,
                    "caption": ""
                }
            }

            try:
                schema_knowledge = self.linearize.retrieve_linear_function(self.linearize_function, self.use_partition_mark, self.use_format_explanation, self.change_order, structured_data_dict)
            except:
                continue
            if self.specific_objective == "1-shot":
                if idx == 0:
                    cell_lookup_generation_sample_0 = self.retrieve_sample_0(self.cell_lookup_generation(parsed_table, schema_knowledge))
                    cell_lookup_pos_generation_sample_0 = self.retrieve_sample_0(self.cell_lookup_pos_generation(parsed_table, schema_knowledge))
                    row_pair_generation_sample_0 = self.retrieve_sample_0(self.table_row_retrieval_generation(parsed_table, schema_knowledge))
                    column_pair_generation_sample_0 = self.retrieve_sample_0(self.table_column_retrieval_generation(parsed_header, schema_knowledge))
                    size_pair_generation_sample_0 = self.retrieve_sample_0(self.table_size_detection_generation(parsed_table, schema_knowledge))
                    column_span_pair_generation_sample_0 = self.retrieve_sample_0(self.column_merged_cell_detection_generation(tables, schema_knowledge))
                    table_partition_pair_generation_sample_0 = self.retrieve_sample_0(self.table_partition_generation(parsed_table, schema_knowledge))

                cell_lookup_generation_sample = self.append_sample(self.cell_lookup_generation(parsed_table, schema_knowledge), cell_lookup_generation_sample_0)
                cell_lookup_pos_generation_sample = self.append_sample(self.cell_lookup_pos_generation(parsed_table, schema_knowledge), cell_lookup_pos_generation_sample_0)
                row_pair_generation_sample = self.append_sample(self.table_row_retrieval_generation(parsed_table, schema_knowledge), row_pair_generation_sample_0)
                column_pair_generation_sample = self.append_sample(self.table_column_retrieval_generation(parsed_header, schema_knowledge), column_pair_generation_sample_0)
                size_pair_generation_sample = self.append_sample(self.table_size_detection_generation(parsed_table, schema_knowledge), size_pair_generation_sample_0)
                column_span_pair_generation_sample = self.append_sample(self.column_merged_cell_detection_generation(tables, schema_knowledge), column_span_pair_generation_sample_0)
                table_partition_pair_generation_sample = self.append_sample(self.table_partition_generation(parsed_table, schema_knowledge), table_partition_pair_generation_sample_0)

                # 1-shot
                if self.fit_heuristics_constraints(cell_lookup_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pair.append(cell_lookup_generation_sample)
                if self.fit_heuristics_constraints(cell_lookup_pos_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pos_pair.append(cell_lookup_pos_generation_sample)
                if self.fit_heuristics_constraints(row_pair_generation_sample['prompt']) is False:
                    continue
                self.row_pair.append(row_pair_generation_sample)
                if self.fit_heuristics_constraints(column_pair_generation_sample['prompt']) is False:
                    continue
                self.column_pair.append(column_pair_generation_sample)
                if self.fit_heuristics_constraints(size_pair_generation_sample['prompt']) is False:
                    continue
                self.size_pair.append(size_pair_generation_sample)
                if self.fit_heuristics_constraints(column_span_pair_generation_sample['prompt']) is False:
                    continue
                self.column_span_pair.append(column_span_pair_generation_sample)
                if self.fit_heuristics_constraints(table_partition_pair_generation_sample['prompt']) is False:
                    continue
                self.table_partition.append(table_partition_pair_generation_sample)

            elif self.specific_objective == "zero-shot":
                # zero-shot
                cell_lookup_generation_sample = self.cell_lookup_generation(parsed_table, schema_knowledge)
                cell_lookup_pos_generation_sample = self.cell_lookup_pos_generation(parsed_table, schema_knowledge)
                row_pair_generation_sample = self.table_row_retrieval_generation(parsed_table, schema_knowledge)
                column_pair_generation_sample = self.table_column_retrieval_generation(parsed_header, schema_knowledge)
                size_pair_generation_sample = self.table_size_detection_generation(parsed_table, schema_knowledge)
                table_partition_pair_generation_sample = self.table_partition_generation(parsed_table, schema_knowledge)

                if self.fit_heuristics_constraints(cell_lookup_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pair.append(cell_lookup_generation_sample)
                if self.fit_heuristics_constraints(cell_lookup_pos_generation_sample['prompt']) is False:
                    continue
                self.cell_lookup_pos_pair.append(cell_lookup_pos_generation_sample)
                if self.fit_heuristics_constraints(row_pair_generation_sample['prompt']) is False:
                    continue
                self.row_pair.append(row_pair_generation_sample)
                if self.fit_heuristics_constraints(column_pair_generation_sample['prompt']) is False:
                    continue
                self.column_pair.append(column_pair_generation_sample)
                if self.fit_heuristics_constraints(size_pair_generation_sample['prompt']) is False:
                    continue
                self.size_pair.append(size_pair_generation_sample)
                if self.fit_heuristics_constraints(table_partition_pair_generation_sample['prompt']) is False:
                    continue
                self.table_partition.append(table_partition_pair_generation_sample)

        if self.specific_objective == "few-shot":
            # few-shot
            self.cell_lookup_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.cell_lookup_pos_pair)
            self.cell_lookup_pos_pair = self.dte_few_shot_generator.generate_few_shot_examples(
                self.cell_lookup_pos_pair)
            self.row_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.row_pair)
            self.column_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.column_pair)
            self.size_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.size_pair)
            self.table_partition = self.dte_few_shot_generator.generate_few_shot_examples(self.table_partition)

        # save as jsonl (totto)
        logging.info(f"{self.dataset_name} tasks {self.specific_objective} datasets have been generated..")
        save_table_jsonl(self.dataset_name, "cell_lookup", self.mode, self.random_sampling(self.cell_lookup_pair))
        save_table_jsonl(self.dataset_name, "cell_lookup_pos", self.mode, self.random_sampling(self.cell_lookup_pos_pair))
        save_table_jsonl(self.dataset_name, "row_retrieval", self.mode, self.random_sampling(self.row_pair))
        save_table_jsonl(self.dataset_name, "column_retrieval", self.mode, self.random_sampling(self.column_pair))
        save_table_jsonl(self.dataset_name, "size_detection", self.mode, self.random_sampling(self.size_pair))
        save_table_jsonl(self.dataset_name, "merged_cell_detection", self.mode, self.random_sampling(self.column_span_pair))
        save_table_jsonl(self.dataset_name, "table_partition", self.mode, self.random_sampling(self.table_partition))

    def cell_lookup_generation(self, cells, schema_knowledge):
        """
        generate cell lookup dataset, e.g., Retrieve position of cell value “Mary Rose”: (use row index, and column index to answer)
        """
        self.task = "cell_lookup"
        cell_lookup_pair = {}
        # generate ground_truth
        row_idx, column_idx, cell_value = retrieve_unique_cell_element(cells)
        self.request = f"What is the position of the cell value {cell_value}? Use row index and column index to answer, e.g., 2 | 3)\n"
        if self.use_role_prompting:
            cell_lookup_pair["prompt"] = self.instruction + schema_knowledge + "<request>\n" + self.request + self.end_prompt
        else:
            cell_lookup_pair["prompt"] = schema_knowledge + "<request>\n" + self.request + self.end_prompt
        cell_lookup_pair["completion"] = f"{row_idx} | {column_idx}"  # consider header
        return cell_lookup_pair

    def cell_lookup_pos_generation(self, cells, schema_knowledge):
        """
        generate cell lookup[pos] dataset, e.g., Retrieve the cell value of row 3 with column name ‘notes'
        """
        self.task = "cell_lookup_pos"
        cell_lookup_pos_pair = {}
        # generate ground_truth
        row_idx, column_idx, cell_value = retrieve_unique_cell_element(cells)
        self.request = f"What is the cell value of row index {row_idx} column index {column_idx}? Only output the cell value without other information, e.g., Tom)\n"
        if self.use_role_prompting:
            cell_lookup_pos_pair["prompt"] = self.instruction + schema_knowledge + "<request>\n" + self.request + self.end_prompt
        else:
            cell_lookup_pos_pair["prompt"] = schema_knowledge + "<request>\n" + self.request + self.end_prompt
        cell_lookup_pos_pair["completion"] = str(cell_value)
        return cell_lookup_pos_pair

    def table_row_retrieval_generation(self, rows, schema_knowledge):
        """
        generate table row retrieval dataset, e.g.,
        “Act as a table parser, only list the cell values (separating by comma) of the 3rd row of following table”,
        “Act as a table parser, only list the cell values (separating by comma) of the ‘Country’ column of following table”
        """
        self.task = "row_retrieval"
        row_retrieve_pair = {}
        # generate ground_truth
        row_idx, row_value = retrieve_unique_row_element(rows)
        self.request = f"What are the cell values of the {row_idx} row in following table? Only list the cell values one by one using | to split the answers \n"
        if self.use_role_prompting:
            row_retrieve_pair["prompt"] = self.instruction + schema_knowledge + "<request>\n" + self.request + self.end_prompt
        else:
            row_retrieve_pair["prompt"] = schema_knowledge + "<request>\n" + self.request + self.end_prompt
        row_retrieve_pair["completion"] = " | ".join(row_value)
        return row_retrieve_pair

    def table_column_retrieval_generation(self, columns, schema_knowledge):
        """
        generate table column retrieval dataset
        """
        self.task = "column_retrieval"
        row_retrieve_pair = {}
        # generate ground_truth
        column_idx, column_name = retrieve_unique_row_element(columns)
        self.request = f"What is the column name with the index {column_idx} of the following table? Only give the column name without any explanation \n"
        if self.use_role_prompting:
            row_retrieve_pair["prompt"] = self.instruction + schema_knowledge + "<request>\n" + self.request + self.end_prompt
        else:
            row_retrieve_pair["prompt"] = schema_knowledge + "<request>\n" + self.request + self.end_prompt
        row_retrieve_pair["completion"] = str(column_name)
        return row_retrieve_pair

    def table_size_detection_generation(self, rows, schema_knowledge):
        """
        generate scope detection tasks, e.g.,
        "Act as a table parser, What is the scope of this table (The table has <masked> columns, <masked> rows)) Only give the masked value, e.g., 2 | 3?
        """
        self.task = "size_detection"
        ans_detection_pair = {}
        # generate scope ground_truth
        rows_len = len(rows)
        column_len = len(rows[0])
        self.request = f"How many rows in the table? How many columns in the table. Answer the questions one by one and use | to split the answer, e.g., 2 | 3 \n"
        if self.use_role_prompting:
            ans_detection_pair["prompt"] = self.instruction + schema_knowledge + "<request>\n" + self.request + self.end_prompt
        else:
            ans_detection_pair["prompt"] = schema_knowledge + "<request>\n" + self.request + self.end_prompt
        ans_detection_pair["completion"] = f"{rows_len} | {column_len}"
        return ans_detection_pair

    def column_merged_cell_detection_generation(self, columns, schema_knowledge):
        """
        Retrieve the index of the column which span is over 1. (e.g., 3), the column index starts from 0
        """
        self.task = "merged_cell_detection"
        ans_detection_pair = {}
        # generate span ground_truth
        column_idx_list = retrieve_unique_column_span(columns)
        column_idx_parsed = []
        for i in column_idx_list:
            column_idx_parsed.append(i[0])
        self.request = f"What is the index of the column which span is over 1. use | to split the answer (e.g., 3 | 4), the column index starts from 0 \n"
        if self.use_role_prompting:
            ans_detection_pair["prompt"] = self.instruction + schema_knowledge + "<request>\n" + self.request + self.end_prompt
        else:
            ans_detection_pair["prompt"] = schema_knowledge + "<request>\n" + self.request + self.end_prompt
        ans_detection_pair["completion"] = " | ".join([str(x) for x in column_idx_parsed])
        return ans_detection_pair

    def table_partition_generation(self, rows, schema_knowledge):
        """
        Detect the boundary of the table within a given user input design.
        """
        self.task = "table_partition"
        ana_detection_pair = {}
        # generate partition ground_truth
        head_token = rows[0][0]
        end_token = rows[-1][-1]
        self.request = f"What is the first token of the given table? What is the end token of the given table? Answer questions one by one and use | to split the answer.\n"
        ana_detection_pair["prompt"] = self.instruction + "<request>\n" + self.request + schema_knowledge
        ana_detection_pair["completion"] = str(head_token) + "|" + str(end_token)
        return ana_detection_pair

    def table_transpose_generation(self, rows, schema_knowledge):
        """
        Operation and Manipulation. Transpose a table
        """
        self.task = "table_transpose"
        ana_detection_pair = {}
        # generate ground_truth
        transposed_table = np.array(rows).T
        partition = []
        for i in range(len(transposed_table)):
            partition.append(transposed_table[i] + "\n")
            if i == 5:
                break
        ground_truth = " | ".join(partition)
        self.request = f"Transpose the table (including the headers) and output the first 5 rows one by one, use | to split cell value, and use \n to split rows\n"
        if self.use_role_prompting:
            ana_detection_pair["prompt"] = self.instruction + schema_knowledge + "<request>\n" + self.request + self.end_prompt
        else:
            ana_detection_pair["prompt"] = schema_knowledge + "<request>\n" + self.request + self.end_prompt
        ana_detection_pair["completion"] = ground_truth

    def column_swap_generation(self, rows, schema_knowledge):
        """
        swap two columns
        """
        self.task = "column_swap"
        ana_detection_pair = {}
        start_index = random.choice(len(rows[0]))
        end_index = random.choice(len(rows[0]))
        while start_index == end_index:
            end_index = random.choice(len(rows[0]))
        self.request = f"Swap column {start_index} and column {end_index}, and output the column {start_index} value one by one, use | to split cell value\n"
        swap_rows = retrieve_swap_tables(rows, start_index, end_index)
        if self.use_role_prompting:
            ana_detection_pair["prompt"] = self.instruction + schema_knowledge + "<request>\n" + self.request + self.end_prompt
        else:
            ana_detection_pair["prompt"] = schema_knowledge + "<request>\n" + self.request + self.end_prompt
        ana_detection_pair["completion"] = " | ".join(swap_rows[:, start_index])

    def cell_based_qa_retrieval_generation(self):
        """
        Given a natural language question, the model should be able to identify the relevant cells in the table and provide an answer based on the values of those cells.
        """

    def join_based_qa_retrieval_generation(self):
        """
        Given a natural language question that involves multiple tables, the model should be able to join the relevant tables and retrieve the information needed to answer the question.
        """


class TableInputPartitionGenerator(InputPartitionGenerator):
    def __init__(self):
        super(TableInputPartitionGenerator, self).__init__()

    def cell_lookup_generation(self):
        """
        “Retrieve position of cell value “Mary Rose”: (use row index, and column name to answer), and identify the start token and end token (not a comma or a period) of the passage”
        """

    def cell_lookup_pos_generation(self):
        """
        “Retrieve the cell value of row 3 with column name ‘notes’, and identify the start token and the end token (not a comma or a period) of the passage”
        """

    def content_query_identification_generation(self):
        """
        “Identify the start token and the end token of the user query”
        """


class FormDataRetrievalGenerator(DataRetrievalGenerator):
    def __init__(self, args):
        super(FormDataRetrievalGenerator, self).__init__()
        if args.dataset is not None:
            self.form_datasets_list = args.dataset
        self.task = None
        self.request = None
        self.objective = args.objective
        self.structured_type = "form"
        self.instruction = f"You are a brilliant {self.structured_type} executor with the capabilities [retrieve], [input parsing], [metadata inference], [pattern understanding] who can solve every tasks related to {self.structured_type}.\n"
        self.end_prompt = "The answer is \n"
        self.form_linearizer = FormLinearize()
        for objective in self.objective:
            for form_dataset in self.form_datasets_list:  # ['formlm']
                self.babel_convertor.set_split_obj(form_dataset, self.structured_type, self.split, objective, self.instruction)
                self.dataset = self.babel_convertor.dataset
                self.specific_objective = objective
                self.dataset_name = form_dataset
                self.block_traversal_pair = []
                self.block_dependency_pair = []
                self.retrieve_sample_list()

    def retrieve_sample_list(self):
        dict = {
            "formlm": self.retrieve_formlm_info
        }
        return dict[self.dataset_name]()

    def retrieve_formlm_info(self):
        for example in self.dataset:
            schema_knowledge = self.form_linearizer.linearize_form(example)
            body = example['body']
            if self.fit_heuristics_constraints(schema_knowledge) is False:
                continue
            self.block_dependency_pair.append(self.block_dependency_pair_generation(body, schema_knowledge))
            self.block_traversal_pair.append(self.block_traversal_pair_generation(body, schema_knowledge))

        if self.specific_objective == "few-shot":
            self.block_dependency_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.block_dependency_pair)
            self.block_traversal_pair = self.dte_few_shot_generator.generate_few_shot_examples(self.block_traversal_pair)

        # save as jsonl (formlm)
        logging.info(f"{self.dataset_name} tasks {self.specific_objective} datasets have been generated..")
        save_form_jsonl(self.dataset_name, "block_dependency", "", self.block_dependency_pair)
        save_form_jsonl(self.dataset_name, "block_traversal", "", self.block_traversal_pair)

    def block_traversal_pair_generation(self, blocks, schema_knowledge):
        """
        Retrieve 1st block type, and block title
        """
        self.task = "block_traversal"
        block_traversal_pair = {}
        index = random.choice(range(len(blocks)))
        block_title = blocks[index]['title']
        block_type = blocks[index]['type']
        self.request = f"Retrieve {index} block's title and type value; use | to split the answer (e.g., Name(Optional) | None), the block index starts from 0 \n"
        block_traversal_pair["prompt"] = self.instruction + "<request>\n" + self.request + schema_knowledge + self.end_prompt
        block_traversal_pair["completion"] = block_title + " | " + block_type
        return block_traversal_pair

    def block_dependency_pair_generation(self, blocks, schema_knowledge):
        """
        Detect whether the block "Age Group" is preceded by "Name", if true return 1, else return 0
        """
        self.task = "block_dependency"
        ans_detection_pair = {}
        idx_1 = random.choice(range(len(blocks)))
        idx_2 = random.choice(range(len(blocks)))
        block1 = blocks[idx_1]['title']  # or hard code choose idx 3 block and idx 4 block
        block2 = blocks[idx_2]['title']
        self.request = f"detect whether the block '{block1}' appears preceded by '{block2}', if true return 1, else return 0 \n"
        ans_detection_pair[
            "prompt"] = self.instruction + "<request>\n" + self.request + schema_knowledge + self.end_prompt
        ans_detection_pair["completion"] = "1" if idx_1 < idx_2 else "0"
        return ans_detection_pair


def get_arguments():
    # required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=["totto"], nargs="+", help="Please specifiy the task name.")
    parser.add_argument("--split", default=["validation"], nargs="+", help="Please specify which split you want to generate/parse.")  # choices = ['train', 'validation', 'test']
    parser.add_argument("--objective", default=["zero-shot", "1-shot", "few-shot"], nargs="+", help="Please specify the parsing objective.")  # choices = ['zero', 'heur_{idx}', 'linear_{idx}']
    # linearize
    parser.add_argument("--linearize_list", default=["markdown", "html", "json", "latex", "nl_sep"], nargs="+")
    parser.add_argument("--use_partition_mark", default=False, action="store_true")
    parser.add_argument("--use_format_explanation", default=False, action="store_true")
    parser.add_argument("--use_role_prompting", default=False, action="store_true")
    parser.add_argument("--change_order", default=False, action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    structured_type = get_keys(DATASETS, args.dataset[0])[0]  # retrieve first dataset type
    if structured_type == "table":
        TableDataRetrievalGenerator(args)
    elif structured_type == "form":
        FormDataRetrievalGenerator(args)


if __name__ == "__main__":
    main()
