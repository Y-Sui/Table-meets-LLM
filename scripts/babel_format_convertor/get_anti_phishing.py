import random
import json

import pandas as pd
from jsonlines import jsonlines


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

def read_atp_form_json(form_jsons):
    form = json.loads(form_jsons)
    # concatenating dictionary value lists
    form_info = []
    for key, value in list(form.items()):
        if type(value) != list:
            if value not in [None, "", "null", "false"]:
                form_info.append(f"The {key} of the form is {str(value)}")
        else:
            body_info = []
            for idx in value:
                for body_key, body_value in list(idx.items()):
                    if body_key in ["options"] and body_value not in [None, "", [], "null", "false"]:
                        body_value = list(map(lambda x:str(x), body_value)) # convert int to str
                        body_info.append(f"The {body_key} of this question are {' '.join(body_value)}")
                    if body_key in ["title", "type", "description"] and body_value not in [None, "", " ", "null", "false"]:
                        body_info.append(f"The {str(body_key)} of this question is {str(body_value)}")
                body_info.append("\r\n")
            form_info.append(
                f"Here are the details of the form, including the title of each question, the category (type), "
                f"and the possible options for each question whose type is choice options: \r\n {' '.join(body_info)}")
    form_info.append(
        f"Based on the form content, Could you please tell me whether the form content is phishing or not?")  # prompt query
    form_info = "@@".join(form_info)
    return form_info

df = pd.read_excel("./all_labeled_dataset_content_available_for_fine_tune.xlsx")
prompt_l = df["form_open_json"]
completion = df["label"]

anti_form = []
error_list = []
for idx in range(len(df)):
    anti_dict = {}
    try:
        anti_dict["prompt"] = read_atp_form_json(prompt_l[idx])
        anti_dict["completion"] = "Yes, the form content is probably phishing." +  "<|endoftext|>" if completion[idx] == 1 else "No, the form content is normal with no tendency to be phishing" +  "<|endoftext|>"
        anti_form.append(anti_dict)
    except:
        error_list.append(idx)

print(f"Error json encoding: {error_list}")

for set in ["train", "validation", "test"]:
    train_split, val_split, test_split = data_split(anti_form, 0.7, 0.2, shuffle=True)
    split_dict = {"train": train_split, "validation": val_split, "test": test_split}
    with jsonlines.open(rf"./generated/anti_phishing/atp_20221206_v2/anti_phishing_{set}.jsonl", mode="w") as writer:
        writer.write_all(split_dict[f"{set}"])


