from datasets import load_dataset
import json
import random
import jsonlines
import os

Task = "hybridqa"
dataset = load_dataset('../dataset_collection/hybridqa.py')
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

def save_raw_jsonl(set):
    os.makedirs(f"generated/{Task}/raw/", exist_ok=True)
    with open(f"generated/{Task}/raw/{set}.jsonl", "w") as outfile:
        for example in dataset[set]:
            outfile.write(json.dumps(example) + "\n")

def save_jsonl(set):
    for idx in Heuristics.keys():
        os.makedirs(f'generated/{Task}/heur_{idx}/', exist_ok=True)
        # Open the output file
        with open(f'generated/{Task}/heur_{idx}/{set}.jsonl', 'w') as outfile:
            # Iterate through the examples in the dataset
            for example in dataset[set]:
                content = {}
                # Scrape the desired content from the example
                table = example["table"]
                question = example["question"]
                passage = example["passage"]
                if table is None:
                    continue
                cells = []
                context = example["context"]
                header = "|".join(table["header"][:-1]) + "\n"
                for i in range(len(table["rows"])):
                    table["rows"][i] = table["rows"][i][:-1]
                    cells.append("|".join(table["rows"][i]) + "\n")
                table_info = header + "".join(cells)
                label = example["answer_text"]
                # Instruction
                heur = Heuristics[idx]
                content["prompt"] = "<request>\n" + heur + "<question>\n" + question + "\n<context>\n" + context + "\n<passage>\n" + passage + "\n<table>\n" + table_info + "\n" + "=>"
                content["completion"] = label
                # Write the content to the output file as a JSON object
                outfile.write(json.dumps(content) + '\n')

    os.makedirs(f'generated/{Task}/zero/', exist_ok=True)
    # Open the output file
    with open(f'generated/{Task}/zero/{set}.jsonl', 'w') as outfile:
        # Iterate through the examples in the dataset
        for example in dataset[set]:
            content = {}
            # Scrape the desired content from the example
            table = example["table"]
            question = example["question"]
            passage = example["passage"]
            if table is None:
                continue
            cells = []
            context = example["context"]
            header = "|".join(table["header"][:-1]) + "\n"
            for i in range(len(table["rows"])):
                table["rows"][i] = table["rows"][i][:-1]
                cells.append("|".join(table["rows"][i]) + "\n")
            table_info = header + "".join(cells)
            label = example["answer_text"]
            # Instruction
            instruct = "Aggregate both tabular information and text information to answer the question (Do not repeat the question, and shorten the answers as much as possible): \n"
            content["prompt"] = "<request>\n" + instruct + "<question>\n" + question + "\n<context>\n" + context + "\n<passage>\n" + passage + "\n<table>\n" + table_info + "\n" + "=>"
            content["completion"] = label
            # Write the content to the output file as a JSON object
            outfile.write(json.dumps(content) + '\n')

def main():
    for set in ["train", "validation"]:
        # save_jsonl(set)
        save_raw_jsonl(set)
if __name__ == "__main__":
    main()
