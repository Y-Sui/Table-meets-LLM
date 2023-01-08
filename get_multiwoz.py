from datasets import load_dataset
import json
import random
import jsonlines
import os

flag = 1

if flag == 0:
    Task = "multi_woz_intent"
    instruct_idx = "intent_state_tracking"
elif flag == 1:
    Task = "multi_woz_dia"
    instruct_idx = "dialog_act_prediction"
# dataset = load_dataset('./scripts/multi_woz_22.py', cache_dir="C:/Users/suiyu/.cache/huggingface/datasets/multi_woz_22/default/2.2.0/multiwoz-44f0f8479f11721831c5591b839ad78827da197b")
dataset = load_dataset("multi_woz_v22")

instruct = {
    "intent_state_tracking": "Predict overall conversation's intent from the user side (return short answers): \n", # 0
    "dialog_act_prediction": "Predict each dialog utterance's intent (one by one) (return short answers): \n" # 1
}

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

def get_unique_items(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def save_jsonl(set):
    for idx in Heuristics.keys():
        os.makedirs(f'generated/{Task}/heur_{idx}/', exist_ok=True)
        # Open the output file
        with open(f'generated/{Task}/heur_{idx}/{set}.jsonl', 'w') as outfile:
            # Iterate through the examples in the dataset
            for example in dataset[set]:
                content = {}
                # Scrape the desired content from the example
                utterance = example["turns"]["utterance"]
                for i in range(len(utterance)):
                    if i % 2 ==0:
                        utterance[i] = "User: " + utterance[i]
                    else:
                        utterance[i] = "System: " + utterance[i]
                dialogues = example["turns"]["dialogue_acts"]
                frames = example["turns"]["frames"]
                activate_intents = [] # user intents
                dialogue_acts = [] # dialogue intents for each utterance
                for i in range(len(frames)):
                    for j in range(len(frames[i]['state'])):
                        active_intent = frames[i]['state'][j]["active_intent"]
                        if active_intent != "None":
                            activate_intents.append(active_intent)
                activate_intents = get_unique_items(activate_intents)
                for i in range(len(dialogues)):
                    dialogue_act = "&".join(dialogues[i]["dialog_act"]["act_type"])
                    dialogue_acts.append(dialogue_act)
                dia_info = "<dialogue>\n" + "\n".join(utterance) + "\n"
                # Instruction
                heur = Heuristics[idx]
                content["prompt"] = "<request>\n" + heur + dia_info + "=>"
                # Set the task
                if flag == 0:
                    content["completion"] = "\t\n".join(activate_intents)
                elif flag == 1:
                    content["completion"] = "\t\n".join(dialogue_acts)
                # Write the content to the output file as a JSON object
                outfile.write(json.dumps(content) + '\n')

    os.makedirs(f'generated/{Task}/zero/', exist_ok=True)
    # Open the output file
    with open(f'generated/{Task}/zero/{set}.jsonl', 'w') as outfile:
        # Iterate through the examples in the dataset
        for example in dataset[set]:
            content = {}
            # Scrape the desired content from the example
            utterance = example["turns"]["utterance"]
            for i in range(len(utterance)):
                if i % 2 == 0:
                    utterance[i] = "User: " + utterance[i]
                else:
                    utterance[i] = "System: " + utterance[i]
            dialogues = example["turns"]["dialogue_acts"]
            frames = example["turns"]["frames"]
            activate_intents = []  # user intents
            dialogue_acts = []  # dialogue intents for each utterance
            for i in range(len(frames)):
                for j in range(len(frames[i]['state'])):
                    active_intent = frames[i]['state'][j]["active_intent"]
                    if active_intent != "None":
                        activate_intents.append(active_intent)
            activate_intents = get_unique_items(activate_intents)
            for i in range(len(dialogues)):
                dialogue_act = "&".join(dialogues[i]["dialog_act"]["act_type"])
                dialogue_acts.append(dialogue_act)
            dia_info = "<dialogue>\n" + "\n".join(utterance) + "\n"
            # Instruction
            instruct_info = instruct[instruct_idx]
            content["prompt"] = "<request>\n" + instruct_info + dia_info + "=>"
            # Set the task
            if flag == 0:
                content["completion"] = "\t\n".join(activate_intents)
            elif flag == 1:
                content["completion"] = "\t\n".join(dialogue_acts)
            # Write the content to the output file as a JSON object
            outfile.write(json.dumps(content) + '\n')

def main():
    for set in ["train", "validation"]:
        save_jsonl(set)

if __name__ == "__main__":
    main()
