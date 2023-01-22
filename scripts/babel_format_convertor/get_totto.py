import argparse
import heapq

from datasets import load_dataset
import json
import random
import jsonlines
import os
import numpy as np

Task = "totto"
dataset = load_dataset('../dataset_collection/totto.py')
print(dataset)
print(dataset["validation"][0])

Heuristics = {
    "0": "Give structural information that will be beneficial for understanding: \n",
    "1": "Let's think step by step: \n",
    "2": "Add structural information: \n",
    "3": "Let's solve this problem by splitting it into steps: \n",
    "4": "First analyze, \n",
    "5": "The answer is after the structural information, \n",
    "6": "Before we dive into the answer, think about the structural information, \n",
    "7": "Give attention to structural information, \n"
}

Linearization_Functions = {
    "0": "Column Name",
    "1": "Column Name | Value",
    "2": "Structure Mark",
    "3": "Structure Mark Unified",
    "4": "Hyper-text",
    "5": "NL + [sep]",
    "6": "NL + Structure Mark + [sep]",
    "7": "NL + Structure Mark + [sep] + content_snapshot"
}

def save_raw_jsonl(set):
    os.makedirs(f"generated/{Task}/raw/", exist_ok=True)
    with open(f"generated/{Task}/raw/{set}.jsonl", "w") as outfile:
        for example in dataset[set]:
            outfile.write(json.dumps(example) + "\n")

def get_content_snapshot(highlight_cells, table_cells):
    from openai.embeddings_utils import get_embedding, cosine_similarity
    import openai
    openai.api_key = "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"

    def deal(list_ori, p):
        list_new = []
        list_short = []
        for i in list_ori:
            if i != p:
                list_short.append(i)
            else:
                list_new.append(list_short)
                list_short = []
        list_new.append(list_short)
        return list_new

    table_cells_split = deal(table_cells, "\n")
    tab_cel = []
    for i in range(len(table_cells_split)):
        tab_cel.append("".join(table_cells_split[i]))
    h_cel = "".join(highlight_cells).replace("\n", ",")
    vector_h_cel = get_embedding(h_cel, engine='text-embedding-ada-002')
    vector_tab_cel = []
    scores = []
    for tab_c in tab_cel:
        vector_tab_cel.append(get_embedding(tab_c, engine="text-embedding-ada-002"))
        scores.append(cosine_similarity(vector_h_cel, vector_tab_cel))
    n_samples = map(scores.index, heapq.nlargest(10, scores))
    content_list = []
    for i in n_samples:
        content_list.append(tab_cel[i])
    content_snap = "".join(content_list)
    return content_snap


def save_jsonl(set, args):
    if args.mode == "heur":
        for idx in Heuristics.keys():
            os.makedirs(f'generated/{Task}/heur_{idx}/', exist_ok=True)
            # Open the output file
            with open(f'generated/{Task}/heur_{idx}/{set}.jsonl', 'w') as outfile:
                # Iterate through the examples in the dataset
                for example in dataset[set]:
                    content = {}
                    highlight_idx = example["highlighted_cells"]
                    final_questions = example["final_sentences"]
                    table_page_title = example["table_page_title"]
                    table_section_title = example["table_section_title"]
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
                    try:
                        for h_idx in highlight_idx:
                            highlight_info.append(
                                str(h_idx) + ": " + str(parsed_header[h_idx[1]]) + "|" + tables[h_idx[0]][h_idx[1]][
                                    "value"] + "\n")
                    except:
                        for h_idx in highlight_idx:
                            highlight_info.append(
                                str(h_idx) + ": " + "-" + "|" + tables[h_idx[0]][h_idx[1]]["value"] + "\n")
                    table_info = "<Page>\n" + table_page_title + "\n" + "<Table>\n" + table_section_title + "\n" + "<Cells>\n" + "".join(
                        header_info) + "\n" + "".join(table_info) + "\n" + "<Highlighted>\n" + "".join(
                        highlight_info) + "\n"
                    # Instruction
                    heur = Heuristics[idx]
                    content["prompt"] = "<request>\n" + heur + table_info + "\t\n\n=>"
                    content["completion"] = "\n".join(final_questions)
                    # Write the content to the output file as a JSON object
                    outfile.write(json.dumps(content) + '\n')
    elif args.mode == "zero":
        os.makedirs(f'generated/{Task}/zero/', exist_ok=True)
        # Open the output file
        with open(f'generated/{Task}/zero/{set}.jsonl', 'w') as outfile:
            # Iterate through the examples in the dataset
            for example in dataset[set]:
                content = {}
                highlight_idx = example["highlighted_cells"]
                final_questions = example["final_sentences"]
                table_page_title = example["table_page_title"]
                table_section_title = example["table_section_title"]
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
                try:
                    for h_idx in highlight_idx:
                        highlight_info.append(str(h_idx)+ ": " +str(parsed_header[h_idx[1]]) + "|" + tables[h_idx[0]][h_idx[1]]["value"] + "\n")
                except:
                    for h_idx in highlight_idx:
                        highlight_info.append(str(h_idx)+ ": " + "-" + "|" + tables[h_idx[0]][h_idx[1]]["value"] + "\n" )
                table_info = "<Page>\n" + table_page_title + "\n" + "<Table>\n" + table_section_title + "\n" + "<Cells>\n" + "".join(
                    header_info) + "\n" + "".join(table_info) + "\n" + "<Highlighted>\n" + "".join(
                    highlight_info) + "\n"
                # Instruction
                instruct = "Generate natural language description for each highlighted part of the table: \t\n\n"
                # Instruction
                content["prompt"] = "<request>\n" + instruct + table_info + "\t\n\n=>"
                content["completion"] = "\n".join(final_questions)
                # Write the content to the output file as a JSON object
                outfile.write(json.dumps(content) + '\n')

    elif args.mode == "linearization":
        for idx in Linearization_Functions.keys():
            os.makedirs(f'generated/{Task}/linear_{idx}/', exist_ok=True)
            # Open the output file
            with open(f'generated/{Task}/linear_{idx}/{set}.jsonl', 'w') as outfile:
                # Iterate through the examples in the dataset
                for example in dataset[set]:
                    content = {}
                    highlight_idx = example["highlighted_cells"]
                    final_questions = example["final_sentences"]
                    table_page_title = example["table_page_title"]
                    table_section_title = example["table_section_title"]
                    tables = example["table"]
                    header_info, cells = [], []
                    highlight_header, highlight_info = [], []
                    for r_idx in range(len(tables)):
                        if r_idx != 0:
                            cells.append("\n")
                        for c_idx in range(len(tables[r_idx])):
                            if r_idx == 0:
                                if tables[r_idx][c_idx]["column_span"] == 1:
                                    header_info.append(tables[r_idx][c_idx]["value"] + "|")
                                else:
                                    for time in range(tables[r_idx][c_idx]["column_span"]):
                                        header_info.append(tables[r_idx][c_idx]["value"] + "|")
                            else:
                                cells.append(tables[r_idx][c_idx]["value"] + "|")
                    parsed_header = list(x.replace("|", "") for x in header_info)
                    try:
                        for h_idx in highlight_idx:
                            highlight_info.append(
                                str(h_idx) + ": " + str(parsed_header[h_idx[1]]) + "|" + tables[h_idx[0]][h_idx[1]][
                                    "value"] + "\n")
                    except:
                        for h_idx in highlight_idx:
                            highlight_info.append(
                                str(h_idx) + ": " + "-" + "|" + tables[h_idx[0]][h_idx[1]]["value"] + "\n")
                    if idx == "0":
                        # Column Name
                        table_info = "".join(header_info) + "\n" + "=============\n" + "".join(highlight_info) + "\n"
                    elif idx == "1":
                        # Column Name | Value
                        table_info = "".join(header_info) + "\n" + "".join(cells) + "\n" + "=============\n" + "".join(highlight_info) + "\n"
                    elif idx == "2":
                        # Structure Mark
                        table_info = "<Table>\n" + table_section_title + "\n" + "<Cells>\n" + "".join(header_info) + "\n" + "".join(cells) + "\n" + "<Highlighted>\n" + "".join(highlight_info) + "\n"
                    elif idx == "3":
                        # Structure Mark Unified
                        table_info = "<Table>\n" + table_section_title + "\n" + "<Cells>\n" + "<Header>\n" + "".join(header_info) + "\n" + "<Value>\n" + "".join(cells) + "\n" + "<Highlighted>\n" + "".join(highlight_info) + "\n"
                    elif idx == "4":
                        # Hyper-text
                        table_info = "<!DOCTYPE html>\n" + "<html>\n" + "\t<title>" + table_page_title + " | " +table_section_title + "/<title>\n" + "\t<body>\n" + "\t\t" + "".join(header_info) + "\n" + "\t\t" + "".join(cells) + "\n" + "\t\t\t<highlight_cell>\n" + "".join(highlight_info) + "\n" + "\t\t\t</highlight_cell>\n" + "\t</body>\n" + "</html>\n"
                    elif idx == "5":
                        # NL + [Sep]
                        table_info = table_page_title + "\n" + table_section_title + "\n" + "".join(header_info) + "\n" + "=============\n" + "".join(highlight_info) + "\n"
                    elif idx == "6":
                        # NL + Structure Mark + [Sep]
                        table_info = "<Page>\n" + table_page_title + "\n" + "<Table>\n" + table_section_title + "\n" + "<Cells>\n" + "".join(
                            header_info) + "\n" + "".join(cells) + "\n" + "<Highlighted>\n" + "".join(
                            highlight_info) + "\n"
                    elif idx == "7":
                        #TODO
                        # NL + Structure Mark + [Sep] + content_snapshot
                        # content_snapshot = get_content_snapshot(highlight_info, cells)
                        content_snapshot = cells
                        table_info = "<Page>\n" + table_page_title + "\n" + "<Table>\n" + table_section_title + "\n" + "<Cells>\n" + "".join(
                            header_info) + "\n" + "".join(content_snapshot) + "\n" + "<Highlighted>\n" + "".join(
                            highlight_info) + "\n"
                    else:
                        table_info = ""
                    # Instruction
                    instruct = "Generate natural language description for each highlighted part of the table: \n"
                    # Instruction
                    content["prompt"] = "<request>\n" + instruct + table_info + "\t\n\n=>"
                    content["completion"] = "\n".join(final_questions)
                    # Write the content to the output file as a JSON object
                    outfile.write(json.dumps(content) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero", "linearization", "heur"], type=str, default="zero")
    args = parser.parse_args()
    for set in ["train","validation"]:
        # save_jsonl(set, args)
        save_raw_jsonl(set)
if __name__ == "__main__":
    main()
