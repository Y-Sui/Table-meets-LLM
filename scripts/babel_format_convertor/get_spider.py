from datasets import load_dataset
import json
import random
import jsonlines
import os

Task = "spider"
dataset = load_dataset('./scripts/dataset_collection/spider.py', ignore_verifications=True)

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
                input = example["question"]
                label = example["query"]
                db_table_names = example["db_table_names"]
                db_column_names = example["db_column_names"]["column_name"]
                db_column_types = example["db_column_types"]
                db_primary_keys = example["db_primary_keys"]["column_id"]
                db_foreign_keys = example["db_foreign_keys"]["column_id"]
                cells = []
                for i in range(len(db_column_names)):
                    cells.append(db_column_names[i] + "|" + db_column_types[i])
                # Instruction
                heur = Heuristics[idx]
                content["prompt"] = "<request>\n" +  heur + "<question>\n" + input + "\n" + "<database>\n" +"|".join(db_table_names) + "\n" + "\n".join(cells) + "\t\n\n\n=>"
                content["completion"] = label
                # Write the content to the output file as a JSON object
                outfile.write(json.dumps(content) + '\n')

    os.makedirs(f'generated/{Task}/zero/', exist_ok=True)
    # Open the output file
    with open(f'generated/{Task}/zero/{set}.jsonl', 'w') as outfile:
        for example in dataset[set]:
            content = {}
            # Scrape the desired content from the example
            input = example["question"]
            label = example["query"]
            db_table_names = example["db_table_names"]
            db_column_names = example["db_column_names"]["column_name"]
            db_column_types = example["db_column_types"]
            db_primary_keys = example["db_primary_keys"]["column_id"]
            db_foreign_keys = example["db_foreign_keys"]["column_id"]
            cells = []
            for i in range(len(db_column_names)):
                cells.append(db_column_names[i] + "|" + db_column_types[i])
            # Instruction
            instruct = "Generate SQL from the given natural language question: \n"
            content["prompt"] = "<request>\n" + instruct + "<question>\n" + input + "\n" + "<database>\n" +"|".join(db_table_names) + "\n" + "\n".join(cells) + "\t\n\n\n=>"
            content["completion"] = label
            # Write the content to the output file as a JSON object
            outfile.write(json.dumps(content) + '\n')

def main():
    for set in ["train", "validation"]:
        # save_jsonl(set)
        save_raw_jsonl(set)
if __name__ == "__main__":
    main()
