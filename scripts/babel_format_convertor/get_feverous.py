from datasets import load_dataset
import json
import random
import jsonlines
import os

Task = "feverous"
dataset = load_dataset('../dataset_collection/feverous.py', ignore_verifications=True)
print(dataset["validation"])
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
                context = "|".join(example["context"])
                label = example["label"]
                statement = example["statement"]
                header = "|".join(example["table"]["header"][0]) + "\n"
                cells = []
                for i in range(len(example["table"]["rows"][0])):
                    cells.append("|".join(example["table"]["rows"][0][i]) + "\n")
                cells = "".join(cells)
                table_info = header + cells
                # Instruction
                heur = Heuristics[idx]
                content["prompt"] = "<request>\n" +  heur + "\n" + "<statement>\n" + statement + "\n" + "<context>\n" + context + "\n" + "<table>\n" + table_info + "\t\n\n\n=>"
                content["completion"] = "0" if label == "REFUTES" else "1"
                # Write the content to the output file as a JSON object
                outfile.write(json.dumps(content) + '\n')

    os.makedirs(f'generated/{Task}/zero/', exist_ok=True)
    # Open the output file
    with open(f'generated/{Task}/zero/{set}.jsonl', 'w') as outfile:
        # Iterate through the examples in the dataset
        for example in dataset[set]:
            content = {}
            # Scrape the desired content from the example
            context = "|".join(example["context"])
            label = example["label"]
            statement = example["statement"]
            header = "|".join(example["table"]["header"][0]) + "\n"
            cells = []
            for i in range(len(example["table"]["rows"][0])):
                cells.append("|".join(example["table"]["rows"][0][i]) + "\n")
            cells = "".join(cells)
            table_info = header + cells
            # Instruction
            instruct = "Verified the claim with the evidence in the forms of sentences and cells from tables, the answer should be 0 if refutes, or 1 if supports"
            content["prompt"] = "<request>\n" + instruct + "\n" + "<statement>\n" + statement + "\n" + "<context>\n" + context + "\n" + "<table>\n" + table_info + "\t\n\n\n=>"
            content["completion"] = "0" if label == "REFUTES" else "1"
            # Write the content to the output file as a JSON object
            outfile.write(json.dumps(content) + '\n')

def main():
    for set in ["train", "validation"]:
        # save_jsonl(set)
        save_raw_jsonl(set)

if __name__ == "__main__":
    main()
