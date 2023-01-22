from datasets import load_dataset
import json
import random
import jsonlines
import os

Task = "webqsp"
dataset = load_dataset('./scripts/webqsp.py')
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
                kg_tuples = example["kg_tuples"]
                question = example["question"]
                answer = example["answers"]
                answer_cell = []
                flag = 1
                for i in range(len(answer)):
                    if None not in answer[i]:
                        answer_cell.append("|".join(answer[i]))
                    else:
                        flag = 0
                if flag == 0:
                    continue
                answer_cell = "".join(answer_cell)
                entities = example["entities"]
                cells = []
                for i in range(len(kg_tuples)):
                    cells.append("|".join(kg_tuples[i]) + "\n")
                mentioned_cells = []
                for i in range(len(entities)):
                    mentioned_cells.append(" | ".join(entities[i]) + "\n")
                mentioned_cells = "|".join(mentioned_cells)
                kg_info = "".join(cells)
                # Instruction
                heur = Heuristics[idx]
                content["prompt"] = "<request>\n" + heur + "<question>\n" + question + "\n" + "<knowledge_graph>\n" + kg_info + "\n" + "<mentioned_cell>\n" + mentioned_cells + "\n" + "=>"
                content["completion"] = answer_cell
                # Write the content to the output file as a JSON object
                outfile.write(json.dumps(content) + '\n')

    os.makedirs(f'generated/{Task}/zero/', exist_ok=True)
    # Open the output file
    with open(f'generated/{Task}/zero/{set}.jsonl', 'w') as outfile:
        # Iterate through the examples in the dataset
        for example in dataset[set]:
            content = {}
            # Scrape the desired content from the example
            kg_tuples = example["kg_tuples"]
            question = example["question"]
            answer = example["answers"]
            answer_cell = []
            flag = 1
            for i in range(len(answer)):
                if None not in answer[i]:
                    answer_cell.append("|".join(answer[i]))
                else:
                    flag = 0
            if flag == 0:
                continue
            answer_cell = "".join(answer_cell)
            entities = example["entities"]
            cells = []
            for i in range(len(kg_tuples)):
                cells.append("|".join(kg_tuples[i]) + "\n")
            mentioned_cells = []
            for i in range(len(entities)):
                mentioned_cells.append(" | ".join(entities[i]) + "\n")
            mentioned_cells = "|".join(mentioned_cells)
            kg_info = "".join(cells)
            # Instruction
            instruct = "Answer the question using entity id instead of entity name, with the mentioned entity and knowledge graph information: \n"
            content["prompt"] = "<request>\n" + instruct + "<question>\n" + question + "\n" + "<knowledge_graph>\n" + kg_info + "\n" + "<mentioned_cell>\n" + mentioned_cells + "\n" + "=>"
            content["completion"] = answer_cell
            # Write the content to the output file as a JSON object
            outfile.write(json.dumps(content) + '\n')

def main():
    for set in ["train", "validation"]:
        # save_jsonl(set)
        save_raw_jsonl(set)
if __name__ == "__main__":
    main()
