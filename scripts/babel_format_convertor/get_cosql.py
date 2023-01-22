from datasets import load_dataset
import json
import random
import jsonlines
import os

Task = "cosql"
dataset = load_dataset('./scripts/cosql.py')

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
                query = example["query"]
                utterance = example["utterances"][0]
                db_table_names = "<db_table_names>\n" + "|".join(example["db_table_names"]) + "\n"
                db_column_names = "<db_column_names>\n" + "|".join(example["db_column_names"]["column_name"]) + "\n<db_column_type>\n" + "|".join(
                    example["db_column_types"]) + "\n"
                db_primary_keys = "<primary_key>\n" + "|".join(list(map(lambda x: str(x), example["db_primary_keys"]["column_id"]))) + "\n"
                db_foreign_keys = "<foreign_key>\n" + "|".join(list(map(lambda x: str(x), example["db_foreign_keys"]["column_id"]))) + "\n"
                db_info = db_table_names + db_column_names + db_primary_keys + db_foreign_keys
                # Instruction
                heur = Heuristics[idx]
                content["prompt"] = "<request>\n" + heur + "\n<utterance>\n" + utterance + db_info + "\t\n\n\n=>"
                content["completion"] = query
                # Write the content to the output file as a JSON object
                outfile.write(json.dumps(content) + '\n')

    os.makedirs(f'generated/{Task}/zero/', exist_ok=True)
    # Open the output file
    with open(f'generated/{Task}/zero/{set}.jsonl', 'w') as outfile:
        # Iterate through the examples in the dataset
        for example in dataset[set]:
            content = {}
            # Scrape the desired content from the example
            query = example["query"]
            utterance = example["utterances"][0]
            db_table_names = "<db_table_names>\n" + "|".join(example["db_table_names"]) + "\n"
            db_column_names = "<db_column_names>\n" + "|".join(
                example["db_column_names"]["column_name"]) + "\n<db_column_type>\n" + "|".join(
                example["db_column_types"]) + "\n"
            db_primary_keys = "<primary_key>\n" + "|".join(
                list(map(lambda x: str(x), example["db_primary_keys"]["column_id"]))) + "\n"
            db_foreign_keys = "<foreign_key>\n" + "|".join(
                list(map(lambda x: str(x), example["db_foreign_keys"]["column_id"]))) + "\n"
            db_info = db_table_names + db_column_names + db_primary_keys + db_foreign_keys
            # Instruction
            instruct = "Generate SQL based on the utterance and database information: "
            content["prompt"] = "<request>\n" + instruct + "\n<utterance>\n" + utterance + "\n" + db_info + "\n=>"
            content["completion"] = query
            # Write the content to the output file as a JSON object
            outfile.write(json.dumps(content) + '\n')

def main():
    for set in ["train", "validation"]:
        save_raw_jsonl(set)
        # save_jsonl(set)

if __name__ == "__main__":
    main()
