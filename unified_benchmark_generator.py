import argparse
import json
import logging
import os
import sys
import random

from unified_babel_convertor import BabelConvertor
from config import DATASETS

sys.path.insert(0, "utils")

from datasets import load_dataset
from utils import get_unique_items
from config import DATASETS, get_heuristics, get_requests

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class BabelBenchmarkGenerator:
    def __init__(self):
        self.babel_convertor = BabelConvertor()


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


class DataRetrievalGenerator(BabelBenchmarkGenerator):
    def __init__(self):
        super(DataRetrievalGenerator, self).__init__()
        self.table_datasets_list = DATASETS["table"]
        self.form_datasets_list = DATASETS["form"]
        self.split = "validation"
        self.objective = "zero-shot"  # benchmark setting


class InputPartitionGenerator(BabelBenchmarkGenerator):
    def __init__(self):
        super(InputPartitionGenerator, self).__init__()


def save_jsonl(dataset_name, task, content_list: list):
    """
    save as jsonl file
    """
    os.makedirs(f"./generated/benchmark/table/", exist_ok=True)
    with open(f"./generated/benchmark/table/{dataset_name}_{task}.jsonl", "w") as outfile:
        for content in content_list:
            outfile.write(json.dumps(content) + "\n")


class TableDataRetrievalGenerator(DataRetrievalGenerator):
    def __init__(self, args):
        super(TableDataRetrievalGenerator, self).__init__()
        if args.dataset is not None:
            self.table_datasets_list = args.dataset
        self.task = None
        self.request = None
        self.cell_lookup_pair = []
        self.cell_lookup_pos_pair = []
        self.row_pair = []
        self.column_pair = []
        self.scope_pair = []
        self.structured_type = "table"
        self.instruction = f"You are a brilliant {self.structured_type} executor with the capabilities [retrieve], [input parsing], [metadata inference], [pattern understanding] who can solve every tasks related to {self.structured_type}.\n"
        self.end_prompt = "The answer is "
        for table_dataset in self.table_datasets_list:  # ["tabfact", "sqa", "hybridqa", "gittables", "feverous", "totto"]
            self.babel_convertor.set_split_obj(table_dataset, self.structured_type, self.split, self.objective,
                                               self.instruction)
            self.dataset = self.babel_convertor.dataset
            self.dataset_name = table_dataset
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

    def retrieval_tabfact_info(self):
        for example in self.dataset[self.split]:
            input = example["table"]
            cells = []
            header = "|".join(input["header"]) + "\n"
            for i in range(len(input["rows"])):
                cells.append("|".join(input["rows"][i]) + "\n")
            schema_knowledge = "<header>\n" + header + "<cells>\n" + "".join(cells) + self.end_prompt
            self.cell_lookup_pair.append(self.cell_lookup_generation(example['table']['rows'], schema_knowledge))
            self.cell_lookup_pos_pair.append(
                self.cell_lookup_pos_generation(example['table']['rows'], schema_knowledge))
            self.row_pair.append(self.table_row_retrieval_generation(example['table']['rows'], schema_knowledge))
            self.column_pair.append(
                self.table_column_retrieval_generation(example['table']['header'], schema_knowledge))
            self.scope_pair.append(self.table_scope_detection_generation(example["table"]['rows'], schema_knowledge))

        # save as jsonl (tabfact)
        logging.info(f"{self.dataset_name} tasks datasets have been generated..")
        save_jsonl(self.dataset_name, "cell_lookup", self.cell_lookup_pair)
        save_jsonl(self.dataset_name, "cell_lookup_pos", self.cell_lookup_pos_pair)
        save_jsonl(self.dataset_name, "row_retrieval", self.row_pair)
        save_jsonl(self.dataset_name, "column_retrieveal", self.column_pair)
        save_jsonl(self.dataset_name, "scope_detection", self.scope_pair)

    def retrieval_sqa_info(self):
        for example in self.dataset[self.split]:
            # Scrape the desired content from the example
            header = "|".join(example["table_header"]) + "\n"
            cells = []
            table = example["table_data"]
            for i in range(len(table)):
                cells.append("|".join(table[i]) + "\n")
            cells = "".join(cells)
            schema_knowledge = "<header>\n" + header + "<cells>\n" + cells + self.end_prompt
            self.cell_lookup_pair.append(self.cell_lookup_generation(example['table_data'], schema_knowledge))
            self.cell_lookup_pos_pair.append(self.cell_lookup_pos_generation(example['table_data'], schema_knowledge))
            self.row_pair.append(self.table_row_retrieval_generation(example['table_data'], schema_knowledge))
            self.column_pair.append(self.table_column_retrieval_generation(example['table_header'], schema_knowledge))
            self.scope_pair.append(self.table_scope_detection_generation(example['table_data'], schema_knowledge))

        # save as jsonl (sqa)
        logging.info(f"{self.dataset_name} tasks datasets have been generated..")
        save_jsonl(self.dataset_name, "cell_lookup", self.cell_lookup_pair)
        save_jsonl(self.dataset_name, "cell_lookup_pos", self.cell_lookup_pos_pair)
        save_jsonl(self.dataset_name, "row_retrieval", self.row_pair)
        save_jsonl(self.dataset_name, "column_retrieveal", self.column_pair)
        save_jsonl(self.dataset_name, "scope_detection", self.scope_pair)

    def retrieval_hybridqa_info(self):
        for example in self.dataset[self.split]:
            table = example["table"]
            cells = []
            header = "|".join(table["header"][:-1]) + "\n"
            for i in range(len(table["rows"])):
                table["rows"][i] = table["rows"][i][:-1]
                cells.append("|".join(table["rows"][i]) + "\n")
            schema_knowledge = "<header>\n" + header + "<cells>\n" + "".join(cells) + self.end_prompt
            self.cell_lookup_pair.append(self.cell_lookup_generation(example['table']['rows'], schema_knowledge))
            self.cell_lookup_pos_pair.append(
                self.cell_lookup_pos_generation(example['table']['rows'], schema_knowledge))
            self.row_pair.append(self.table_row_retrieval_generation(example['table']['rows'], schema_knowledge))
            self.column_pair.append(
                self.table_column_retrieval_generation(example['table']['header'], schema_knowledge))
            self.scope_pair.append(self.table_scope_detection_generation(example['table']['rows'], schema_knowledge))

        # save as jsonl (hybridqa)
        logging.info(f"{self.dataset_name} tasks datasets have been generated..")
        save_jsonl(self.dataset_name, "cell_lookup", self.cell_lookup_pair)
        save_jsonl(self.dataset_name, "cell_lookup_pos", self.cell_lookup_pos_pair)
        save_jsonl(self.dataset_name, "row_retrieval", self.row_pair)
        save_jsonl(self.dataset_name, "column_retrieveal", self.column_pair)
        save_jsonl(self.dataset_name, "scope_detection", self.scope_pair)

    def retrieval_feverous_info(self):
        for example in self.dataset[self.split]:
            # Scrape the desired content from the example
            header = "|".join(example["table"]["header"][0]) + "\n"
            cells = []
            for i in range(len(example["table"]["rows"][0])):
                cells.append(" | ".join(example["table"]["rows"][0][i]) + "\n")
            cells = "".join(cells) + "\n"
            schema_knowledge = "<header>\n" + header + "<cells>\n" + cells + self.end_prompt
            self.cell_lookup_pair.append(self.cell_lookup_generation(example['table']['rows'][0], schema_knowledge))
            self.cell_lookup_pos_pair.append(
                self.cell_lookup_pos_generation(example['table']['rows'][0], schema_knowledge))
            self.row_pair.append(self.table_row_retrieval_generation(example['table']['rows'][0], schema_knowledge))
            self.column_pair.append(
                self.table_column_retrieval_generation(example['table']['header'][0], schema_knowledge))
            self.scope_pair.append(self.table_scope_detection_generation(example["table"]['rows'][0], schema_knowledge))

        # save as jsonl (feverous)
        logging.info(f"{self.dataset_name} tasks datasets have been generated..")
        save_jsonl(self.dataset_name, "cell_lookup", self.cell_lookup_pair)
        save_jsonl(self.dataset_name, "cell_lookup_pos", self.cell_lookup_pos_pair)
        save_jsonl(self.dataset_name, "row_retrieval", self.row_pair)
        save_jsonl(self.dataset_name, "column_retrieveal", self.column_pair)
        save_jsonl(self.dataset_name, "scope_detection", self.scope_pair)

    def retrieval_totoo_info(self):
        for example in self.dataset[self.split]:
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
            parsed_table = list()
            try:
                for h_idx in highlight_idx:
                    highlight_info.append(
                        str(h_idx) + ": " + str(parsed_header[h_idx[1]]) + "|" + tables[h_idx[0]][h_idx[1]]["value"] + "\n")
            except:
                for h_idx in highlight_idx:
                    highlight_info.append(str(h_idx) + ": " + "-" + "|" + tables[h_idx[0]][h_idx[1]]["value"] + "\n")
            schema_knowledge = "<header>\n" + "".join(header_info) + "<cells>\n" + "".join(table_info) + self.end_prompt




    def cell_lookup_generation(self, cells, schema_knowledge):
        """
        generate cell lookup dataset, e.g., Retrieve position of cell value “Mary Rose”: (use row index, and column index to answer)
        """
        self.task = "cell_lookup"
        cell_lookup_pair = {}
        # generate ground_truth
        row_idx, column_idx, cell_value = retrieve_unique_cell_element(cells)
        self.request = f"Retrieve position of the cell value {cell_value} (Use row index and column index to answer, e.g., 2 | 3)\n"
        cell_lookup_pair["prompt"] = self.instruction + "<request>\n" + self.request + schema_knowledge
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
        self.request = f"Retrieve cell value of row index {row_idx} column index {column_idx} (Only output the answer without other information, e.g., Tom)\n"
        cell_lookup_pos_pair["prompt"] = self.instruction + "<request>\n" + self.request + schema_knowledge
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
        self.request = f"Only list the cell values (separating by |) of the {row_idx} row of following table \n"
        row_retrieve_pair["prompt"] = self.instruction + "<request>\n" + self.request + schema_knowledge
        row_retrieve_pair["completion"] = "|".join(row_value)
        return row_retrieve_pair

    def table_column_retrieval_generation(self, columns, schema_knowledge):
        """
        generate table column retrieval dataset
        """
        self.task = "column_retrieval"
        row_retrieve_pair = {}
        # generate ground_truth
        column_idx, column_name = retrieve_unique_row_element(columns)
        self.request = f"Only output the column name of the {column_idx} column of following table \n"
        row_retrieve_pair["prompt"] = self.instruction + "<request>\n" + self.request + schema_knowledge
        row_retrieve_pair["completion"] = str(column_name)
        return row_retrieve_pair

    def table_scope_detection_generation(self, rows, schema_knowledge):
        """
        generate scope detection tasks, e.g.,
        "Act as a table parser, What is the scope of this table (The table has <masked> columns, <masked> rows)) Only give the masked value, e.g., 2 | 3?
        """
        self.task = "scope_detection"
        ans_detection_pair = {}
        # generate scope ground_truth
        rows_len = len(rows)
        column_len = len(rows[0])
        self.request = f"What is the scope of this table (The table has <masked> columns, <masked> rows)) Only give the value, e.g., 2 | 3 \n"
        ans_detection_pair["prompt"] = self.instruction + "<request>\n" + self.request + schema_knowledge
        ans_detection_pair["completion"] = f"{rows_len} | {column_len}"
        return ans_detection_pair

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
    def __init__(self):
        super(FormDataRetrievalGenerator, self).__init__()


def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=["feverous", "tabfact", "sqa", "hybridqa"], nargs="+",
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


def main():
    args = get_arguments()
    TableDataRetrievalGenerator(args)


if __name__ == "__main__":
    main()
