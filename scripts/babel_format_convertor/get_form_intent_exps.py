import json
import random
from itertools import chain
import pandas as pd

import jsonlines
import os

# <|endoftext|>

def read_public_form_json(form_jsons, set):
    form_list = []
    for form_idx in form_jsons:
        with open(os.path.join(f"./PublicFormDataset_with_url/{set}", form_idx), "r") as js:
            info_dict = {}
            form = json.load(js)
            # preprocessing (remove URL, GUID)
            form.pop("URL")
            form.pop("GUID")
            if form.get("intents", None) == None:
                continue # remove json file without intents labeled
            for intent_type in ["purpose", "target domain", "target audiences"]:
                # retrieval form interns
                intent = form.pop("intents", None)
                if intent == None:
                    continue
                intent_info = intent["form"][intent_type]
                if intent_info == "Unknown":
                    continue
                # concatenating dictionary value lists
                form_info = []
                for key, value in list(form.items()):
                    if type(value) != list:
                        form_info.append(f"The {key} of the form is {str(value)}")
                    else:
                        body_info = []
                        for idx in value:
                            for body_key, body_value in list(idx.items()):
                                if body_key == "options":
                                    # body_info.append(f"{body_key}: {' '.join(body_value)}")
                                    body_info.append(f"The {body_key} of this question are {' '.join(body_value)}")
                                if body_key in ["title", "description", "type"] and body_value not in [None, "null", "false"]:
                                    # body_info.append(f"{str(body_key)}: {str(body_value)}")
                                    body_info.append(f"The {str(body_key)} of this question is {str(body_value)}")
                            body_info.append("\r\n")
                        # form_info.append(f"body_info: {' '.join(body_info)}")
                        form_info.append(
                            f"Here are the details of the form, including the title of each question, the category (type), "
                            f"and the possible options for each question whose type is choice options: \r\n {' '.join(body_info)}")
                form_info.append(f"Based on the form content, Could you please tell me what is the {intent_type} of the form?") # prompt query
                form_info = "@@".join(form_info)
                info_dict["prompt"] = form_info
                info_dict["completion"] = intent_info + "<|endoftext|>" # append end of context
                form_list.append(info_dict)
    return form_list

def json_to_jsonl(set):
    form_jsons = os.listdir(f"./PublicFormDataset_with_url/{set}") # get each set's json files
    form_contents = read_public_form_json(form_jsons, set)
    with jsonlines.open(rf"./generated/public_form_intent/public_forms_intents_20221206_v2/public_form_intent_{set}.jsonl", mode="w") as writer:
        writer.write_all(form_contents)

def main():
    for set in ["train", "dev", "test"]:
        json_to_jsonl(set)

if __name__ == "__main__":
    main()
