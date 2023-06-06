import json
import os
import csv
import pandas as pd


# load the evaluation results from json file
def load_results_from_json(dir_path: str, log_dir_name: str):
    dir_path_files = []

    # search the dir_path if there is a log_dir_name
    for file in os.listdir(dir_path):
        if file.__contains__(log_dir_name):
            full_path = os.path.join(dir_path, file)
            if os.path.isfile(full_path):
                with open(full_path, "r") as f:
                    results = json.load(f)
                dir_path_files.append(results)

    return dir_path_files


def convert_json_to_csv(json_data: list, output_file_name: str):
    # save the results to csv file
    with open(
        output_file_name,
        "w",
        newline=[
            'Choice',
            'TabFact',
            'HybridQA',
            'SQA',
            'Feverous',
            'ToTTo',
            'ToTTo',
            'ToTTo',
            'ToTTo',
        ],
    ) as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(len(json_data)):
            for key, value in json_data[i].items():
                task, file_format, metric = key.split("_")
                writer.writerow([task, file_format, metric, value])


def convert_pandas_to_csv(json_data: list, output_file_name: str):
    df = pd.DataFrame(
        columns=[
            'Choice',
            'TabFact',
            'HybridQA',
            'SQA',
            'Feverous',
            'ToTTo',
            'ToTTo',
            'ToTTo',
            'ToTTo',
        ],
        index=[
            "NL+Sep",
            "NL+Sep w/o format explanation",
            "NL+Sep w/o partition mark",
            "NL+Sep w/o role prompting",
            "Markdown",
            "Markdown w/o format explanation",
            "Markdown w/o partition mark",
            "Markdown w/o role prompting",
            "Json",
            "Json w/o format explanation",
            "Json w/o partition mark",
            "Json w/o role prompting",
            "XML",
            "XML w/o format explanation",
            "XML w/o partition mark",
            "XML w/o role prompting",
            "HTML",
            "HTML w/o format explanation",
            "HTML w/o partition mark",
            "HTML w/o role prompting",
        ],
    )

    for i in range(len(json_data)):
        for key, value in json_data[i].items():
            _ = key.replace("['", "").replace("']", "").split("_")
            task, input_choices, metric = (
                _[0].title(),
                format_mapping_func('_'.join(_[1:-1])),
                _[-1],
            )
            df.loc[input_choices, task] = value

    print(df)


def format_mapping_func(input_choices: str):
    _ = input_choices.split("_")

    format, option = _[0], "_".join(_[1:])
    if option.__contains__("sep"):
        option.replace("sep_", "")

    if format == "nl":
        if option == "0_1":
            return "NL+Sep w/o role prompting"
        elif option == "0_1_3":
            return "NL+Sep"
        elif option == "0_3":
            return "NL+Sep w/o format explanation"
        elif option == "1_3":
            return "NL+Sep w/o partition mark"

    elif format == "markdown":
        if option == "0_1":
            return "Markdown w/o role prompting"
        elif option == "0_1_3":
            return "Markdown"
        elif option == "0_3":
            return "Markdown w/o format explanation"
        elif option == "1_3":
            return "Markdown w/o partition mark"

    elif format == "xml":
        if option == "0_1":
            return "xml w/o role prompting"
        elif option == "0_1_3":
            return "xml"
        elif option == "0_3":
            return "xml w/o format explanation"
        elif option == "1_3":
            return "xml w/o partition mark"

    elif format == "markdown":
        if option == "0_1":
            return "markdown w/o role prompting"
        elif option == "0_1_3":
            return "markdown"
        elif option == "0_3":
            return "markdown w/o format explanation"
        elif option == "1_3":
            return "markdown w/o partition mark"

    elif format == "json":
        if option == "0_1":
            return "json w/o role prompting"
        elif option == "0_1_3":
            return "json"
        elif option == "0_3":
            return "json w/o format explanation"
        elif option == "1_3":
            return "json w/o partition mark"

    elif format == "html":
        if option == "0_1":
            return "html w/o role prompting"
        elif option == "0_1_3":
            return "html"
        elif option == "0_3":
            return "html w/o format explanation"
        elif option == "1_3":
            return "html w/o partition mark"


dir_path = "exps"
log_dir_name = "downstream_tasks_20230308_self_augmented_p2_logheur_8"

convert_pandas_to_csv(load_results_from_json(dir_path, log_dir_name), "")
