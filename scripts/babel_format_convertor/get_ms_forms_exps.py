import json
import random
from itertools import chain
import pandas as pd

import jsonlines
import os

def read_ms_form_json(form_jsons):
    form_list = []
    for form_idx in form_jsons:
        with open(os.path.join(f"./ms_forms_templates/ms_forms_templates_parsed_20220428", form_idx), "r") as js:
            info_dict = {}
            form = json.load(js)
            # preprocessing (remove Id)
            form.pop("Id")
            for intent_type in ["purpose", "target domain", "target audiences"]:
                # concatenating dictionary value lists
                form_info = []
                for key, value in list(form.items()):
                    if type(value) != list:
                        # form_info.append(f"{key}: {str(value)}") # 20221206_v1
                        form_info.append(f"The {key} of this form is {str(value)}")
                    else:
                        body_info = []
                        for idx in value:
                            for body_key, body_value in list(idx.items()):
                                if body_key in ["Choices", "Labels", "Scores"]:
                                    body_value = list(map(lambda x:str(x), body_value)) # convert int to str
                                    # body_info.append(f"{body_key}: {' '.join(body_value)}") # 20221206_v1
                                    body_info.append(f"The {body_key} of this question are {' '.join(body_value)}") # 20221206_v2
                                if body_key in ["Title", "Type"] and body_value not in [None, "", "null", "false"]:
                                    # body_info.append(f"{str(body_key)}: {str(body_value)}")
                                    body_info.append(f"The {str(body_key)} of this question is: {str(body_value)}")
                            body_info.append("\r\n")
                        # form_info.append(f"body_info: {' '.join(body_info)}") 20221206_v1
                        form_info.append(f"Here are the details of the form, including the title of each question, the category (type), "
                                         f"and the possible options for each question whose type is choice options:\r\n {' '.join(body_info)}") # 20221206_v2 instruction
                form_info.append(
                    f"Based on the form content, Could you please tell me what is the {intent_type} of the form?")  # prompt query
                form_info = "@@".join(form_info)
                info_dict["prompt"] = form_info
                # info_dict["completion"] = intent_info
                form_list.append(info_dict)
    return form_list

def data_split(full_list, ratio0, ratio1, shuffle=False):
    n_total = len(full_list)
    offset0 = int(n_total * ratio0)
    offset1 = int(n_total - n_total * ratio1)
    if n_total == 0 or offset0 < 1:
        return [], full_list, full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset0]
    sublist_2 = full_list[offset0:offset1]
    sublist_3 = full_list[offset1:]
    return sublist_1, sublist_2, sublist_3

def json_to_jsonl(set):
    form_jsons = os.listdir("./ms_forms_templates/ms_forms_templates_parsed_20220428") # get each set's json files
    form_contents = read_ms_form_json(form_jsons)
    train_split, val_split, test_split = data_split(form_contents, 0.7, 0.2, shuffle=True)
    split_dict = {"train": train_split, "validation": val_split, "test": test_split}
    with jsonlines.open(rf"./generated/ms_forms_templates_content/ms_forms_templates_20221206_v2/ms_forms_templates_{set}.jsonl", mode="w") as writer:
        writer.write_all(split_dict[f"{set}"])

def main():
    for set in ["train", "validation", "test"]:
        json_to_jsonl(set)

if __name__ == "__main__":
    main()
