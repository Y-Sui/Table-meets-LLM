import argparse
import json
import logging
import os
import sys

sys.path.insert(0, "utils")

from datasets import load_dataset
from utils import get_unique_items, load_json, FormLinearize
from config import DATASETS, get_heuristics, get_requests

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class BabelConvertor:
    def __init__(self):
        self.form_linearizer = None
        self.prompt_input = None
        self.split = None
        self.objective = None
        self.instruct = None
        self.data_type = None
        self.task = None

    def set_split_obj(self, task: str, structured_type: str, split: str, objective: str, instruction: str):
        self.prompt_input = []  # refresh when init set_split_obj
        self.split = split
        self.objective = objective
        self.task = task
        self.data_type = structured_type
        self.instruct = instruction
        try:
            self.dataset = load_dataset(f"./scripts/unifiedSKG/{task}.py", ignore_verifications=True)
        except:
            if self.task.__contains__("multi"):
                huggingface_hub = "multi_woz_22"
            elif self.task == "sqa":
                huggingface_hub = "msr_sqa"
            elif self.task == "dart":
                huggingface_hub = "dart"
            elif self.task.__contains__("formlm"):
                huggingface_hub = "./scripts/formlm_dataset_OOF/dev_selected_data0426.json"
            else:
                huggingface_hub = ""
            # load formlm dataset from local
            if self.task.__contains__("formlm"):
                validation_data = load_json(huggingface_hub)["data"]
                self.form_linearizer = FormLinearize()
                self.dataset = []
                for data in validation_data:
                    self.dataset.append(data)
            else:
                self.dataset = load_dataset(huggingface_hub, ignore_verifications=True)
        self.flag = 0  # no heuristics generation (zero-shot)
        self.request = get_requests(self.task)
        self.end_prompt = "The answer is \n"
        if objective.__contains__("heur"):
            self.request = get_heuristics(self.data_type)[objective]
            self.end_prompt = "The structural information is \n"
            self.flag = 1

    def retrieve_sample_list(self):
        dict = {"feverous": self.retrieve_feverous, "dart": self.retrieve_dart, "cosql": self.retrieve_cosql,
                "gittables": self.retrieve_gittables, "hybridqa": self.retrieve_hybridqa,
                "logic2text": self.retrieve_logic2text, "sql2text": self.retrieve_sql2text,
                "multi_woz_dia": self.retrieve_multi_woz_dia, "multi_woz_intent": self.retrieve_multi_woz_intent,
                "spider": self.retrieve_spider, "totto": self.retrieve_totto, "tabfact": self.retrieve_tabfact,
                "sqa": self.retrieve_sqa, "webqsp": self.retrieve_webqsp, "formlm_opt": self.retrieve_formlm_opt_recommend,
                "formlm_qa": self.retrieve_formlm_qa_recommend, "formlm_block_type": self.retrieve_formlm_block_type_classification}
        return dict[self.task]()

    # def retrieve_formlm(self):
    #     linearized_form = [] # linearized_form
    #     for example in self.dataset:
    #          linearized_form.append(self.form_linearizer.linearize_form(example))

    def retrieve_formlm_opt_recommend(self):
        """
        formlm subtasks --> options recommendation
        """
        for example in self.dataset:
            inputs, targets = self.form_linearizer.linearize_form_for_option_recommend(example)
            for i in range(len(inputs)):
                content = {"prompt": self.request + inputs[i] + "\n" + self.end_prompt,
                           "completion": targets[i]}
                self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_formlm_qa_recommend(self):
        """
        formlm subtask --> question recommendation
        """
        for example in self.dataset:
            inputs, targets = self.form_linearizer.linearize_form_for_question_recommend(example, with_context=True)
            for i in range(len(inputs)):
                content = {"prompt": self.request + inputs[i] + "\n" + self.end_prompt,
                           "completion": targets[i]}
                self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_formlm_block_type_classification(self):
        """
        formlm subtask --> block type classification
        """
        for example in self.dataset:
            inputs, targets = self.form_linearizer.linearize_form_for_block_type_classification(example, with_context=True)
            for i in range(len(inputs)):
                content = {"prompt": self.request + inputs[i] + "\n" + self.end_prompt,
                           "completion": targets[i]}
                self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_feverous(self):
        for example in self.dataset[self.split]:
            content = {}
            # Scrape the desired content from the example
            context = "|".join(example["context"]) + "\n"
            label = example["label"]
            statement = example["statement"] + "\n"
            header = "|".join(example["table"]["header"][0]) + "\n"
            cells = []
            for i in range(len(example["table"]["rows"][0])):
                cells.append("|".join(example["table"]["rows"][0][i]) + "\n")
            cells = "".join(cells) + "\n"
            table_info = header + cells
            content[
                "prompt"] = self.instruct + "<request>\n" + self.request + "<statement>\n" + statement + "<context>\n" + context + "<table>\n" + table_info + self.end_prompt
            content["completion"] = "0" if label == "REFUTES" else "1"
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_dart(self):
        for example in self.dataset[self.split]:
            content = {}
            # Scrape the desired content from the example
            input = example["tripleset"]
            input_seq = []
            if input is None:
                continue
            for i in range(len(input)):
                input_seq.append(input[i][0] + " | " + input[i][1] + " | " + input[i][2] + "\n")
            input_seq = "".join(input_seq) + "\n"
            label = example["annotations"]["text"][0]
            content[
                "prompt"] = self.instruct + "<request>\n" + self.request + "<RDF_Triplets>\n" + input_seq + self.end_prompt
            content["completion"] = label
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_cosql(self):
        for example in self.dataset[self.split]:
            content = {}
            # Scrape the desired content from the example
            query = example["query"]
            utterance = example["utterances"][0] + "\n"
            db_table_names = "<db_table_names>\n" + "|".join(example["db_table_names"]) + "\n"
            db_column_names = "<db_column_names>\n" + "|".join(
                example["db_column_names"]["column_name"]) + "\n<db_column_type>\n" + "|".join(
                example["db_column_types"]) + "\n"
            db_primary_keys = "<primary_key>\n" + "|".join(
                list(map(lambda x: str(x), example["db_primary_keys"]["column_id"]))) + "\n"
            db_foreign_keys = "<foreign_key>\n" + "|".join(
                list(map(lambda x: str(x), example["db_foreign_keys"]["column_id"]))) + "\n"
            db_info = db_table_names + db_column_names + db_primary_keys + db_foreign_keys
            content[
                "prompt"] = self.instruct + "<request>\n" + self.request + "<utterance>\n" + utterance + db_info + self.end_prompt
            content["completion"] = query
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_gittables(self):
        for example in self.dataset[self.split]:
            content = {}
            # Scrape the desired content from the example
            table_id = example["table_id"]
            column_text = example["column_text"]
            if column_text is None or table_id is None:
                continue
            column_text = column_text.replace("[", "").replace("]", "").replace("'", "").replace(",", "|") + "\n"
            label = example["annotation_label"]
            content[
                "prompt"] = self.instruct + "<request>\n" + self.request + "<column>\n" + column_text + self.end_prompt
            content["completion"] = label
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_hybridqa(self):
        for example in self.dataset[self.split]:
            content = {}
            table = example["table"]
            question = example["question"] + "\n"
            passage = example["passage"] + "\n"
            if table is None:
                continue
            cells = []
            context = example["context"] + "\n"
            header = "|".join(table["header"][:-1]) + "\n"
            for i in range(len(table["rows"])):
                table["rows"][i] = table["rows"][i][:-1]
                cells.append("|".join(table["rows"][i]) + "\n")
            table_info = header + "".join(cells) + "\n"
            label = example["answer_text"]
            content[
                "prompt"] = self.instruct + "<request>\n" + self.request + "<question>\n" + question + "<context>\n" + context + "<passage>\n" + passage + "<table>\n" + table_info + self.end_prompt
            content["completion"] = label
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_logic2text(self):
        for example in self.dataset[self.split]:
            content = {}
            # Scrape the desired content from the example
            logic_str = example["logic_str"] + "\n"
            question = example["question"]
            caption = example["table"]["caption"] + "\n"
            header = "|".join(example["table"]["header"]) + "\n"
            cells = []
            for i in range(len(example["table"]["content"])):
                cells.append("|".join(example["table"]["content"][i]) + "\n")
            cells = "".join(cells) + "\n"
            table_info = "<caption>\n" + caption + "<table>\n" + header + cells
            content[
                "prompt"] = self.instruct + "<request>\n" + self.request + "<logic_form>\n" + logic_str + table_info + self.end_prompt
            content["completion"] = question
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_sql2text(self):
        for example in self.dataset[self.split]:
            content = {}
            # Scrape the desired content from the example
            input = example["query"] + "\n"
            if input is None:
                continue
            label = example["question"]
            content["prompt"] = self.instruct + "<request>\n" + self.request + "<sql>\n" + input + self.end_prompt
            content["completion"] = label
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_multi_woz_dia(self):
        for example in self.dataset[self.split]:
            content = {}
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
            content["prompt"] = self.instruct + "<request>\n" + self.request + dia_info + self.end_prompt
            content["completion"] = "\n".join(dialogue_acts)
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_multi_woz_intent(self):
        for example in self.dataset[self.split]:
            content = {}
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
            content["prompt"] = self.instruct + "<request>\n" + self.request + dia_info + self.end_prompt
            content["completion"] = "\n".join(activate_intents)
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_spider(self):
        for example in self.dataset[self.split]:
            content = {}
            input = example["question"] + "\n"
            label = example["query"]
            db_table_names = example["db_table_names"]
            db_column_names = example["db_column_names"]["column_name"]
            db_column_types = example["db_column_types"]
            # db_primary_keys = example["db_primary_keys"]["column_id"] + "\n"
            # db_foreign_keys = example["db_foreign_keys"]["column_id"] + "\n"
            cells = []
            for i in range(len(db_column_names)):
                cells.append(db_column_names[i] + "|" + db_column_types[i])
            # content["prompt"] = self.instruct + "<request>\n" + self.request + "<question>\n" + input + "<database>\n" + "|".join(
            #     db_table_names) + "\n".join(cells) + "<primary_keys>\n" + db_primary_keys + "<foreign_keys>\n" + db_foreign_keys + self.end_prompt
            content[
                "prompt"] = self.instruct + "<request>\n" + self.request + "<question>\n" + input + "<database>\n" + "|".join(
                db_table_names) + "\n" + "\n".join(cells) + self.end_prompt
            content["completion"] = label
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_sqa(self):
        for example in self.dataset[self.split]:
            content = {}
            # id = example["id"]
            question = example["question"] + "\n"
            answer = "|".join(example["answer_text"])
            # Scrape the desired content from the example
            header = "|".join(example["table_header"]) + "\n"
            cells = []
            table = example["table_data"]
            for i in range(len(table)):
                cells.append("|".join(table[i]) + "\n")
            cells = "".join(cells)
            tb_info = header + cells + "\n"
            content[
                "prompt"] = self.instruct + "<request>\n" + self.request + "<question>\n" + question + "<table>\n" + tb_info + self.end_prompt
            content["completion"] = answer
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_tabfact(self):
        for example in self.dataset[self.split]:
            content = {}
            input = example["table"]
            statement = example["statement"] + "\n"
            if input is None:
                continue
            cells = []
            caption = "<caption>\n" + input["caption"] + "\n"
            header = "|".join(input["header"]) + "\n"
            for i in range(len(input["rows"])):
                cells.append("|".join(input["rows"][i]) + "\n")
            table_info = caption + "<cells>\n" + header + "".join(cells) + "\n"
            label = example["label"]
            content[
                "prompt"] = self.instruct + "<request>\n" + self.request + "<statement>\n" + statement + "<table>\n" + table_info + self.end_prompt
            content["completion"] = str(label)
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_totto(self):
        for example in self.dataset[self.split]:
            content = {}
            highlight_idx = example["highlighted_cells"]
            final_questions = example["final_sentences"]
            table_page_title = example["table_page_title"]
            table_section_title = example["table_section_title"]
            tables = example["table"]
            header_info, table_info = [], []
            highlight_header, highlight_info = [], []
            for r_idx in range(len(tables)):
                if r_idx != 0:
                    table_info.append("\n")
                for c_idx in range(len(tables[r_idx])):
                    if r_idx == 0:
                        if tables[r_idx][c_idx]["column_span"] == 1:
                            header_info.append(tables[r_idx][c_idx]["value"] + "|")
                        else:
                            for time in range(tables[r_idx][c_idx]["column_span"]):
                                header_info.append(tables[r_idx][c_idx]["value"] + "|")
                    else:
                        table_info.append(tables[r_idx][c_idx]["value"] + "|")
            parsed_header = list(x.replace("|", "") for x in header_info)
            try:
                for h_idx in highlight_idx:
                    highlight_info.append(
                        str(h_idx) + ": " + str(parsed_header[h_idx[1]]) + "|" + tables[h_idx[0]][h_idx[1]][
                            "value"] + "\n")
            except:
                for h_idx in highlight_idx:
                    highlight_info.append(str(h_idx) + ": " + "-" + "|" + tables[h_idx[0]][h_idx[1]]["value"] + "\n")
            table_info = "<Page>\n" + table_page_title + "\n" + "<Table>\n" + table_section_title + "\n" + "<Cells>\n" + "".join(
                header_info) + "\n" + "".join(table_info) + "\n" + "<Highlighted>\n" + "".join(highlight_info) + "\n"
            content["prompt"] = self.instruct + "<request>\n" + self.request + table_info + self.end_prompt
            content["completion"] = "\n".join(final_questions)
            self.prompt_input.append(content)
        return self.prompt_input

    def retrieve_webqsp(self):
        for example in self.dataset[self.split]:
            content = {}
            kg_tuples = example["kg_tuples"]
            question = example["question"] + "\n"
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
            mentioned_cells = "|".join(mentioned_cells) + "\n"
            kg_info = "".join(cells) + "\n"
            content[
                "prompt"] = self.instruct + "<request>\n" + self.request + "<question>\n" + question + "<knowledge_graph>\n" + kg_info + "<mentioned_cell>\n" + mentioned_cells + self.end_prompt
            content["completion"] = answer_cell
            self.prompt_input.append(content)
        return self.prompt_input


def get_keys(dict, value):
    return [k for k, v in dict.items() if value in v]


def save_raw_jsonl(task: str, split_set: str):
    try:
        dataset = load_dataset(f"./scripts/unifiedSKG/{task}.py", ignore_verifications=True)
    except FileNotFoundError:
        if task.__contains__("multi"):
            huggingface_hub = "multi_woz_22"
        elif task == "sqa":
            huggingface_hub = "msr_sqa"
        elif task == "dart":
            huggingface_hub = "dart"
        elif task.__contains__("formlm"):
            huggingface_hub = "./scripts/formlm_dataset_OOF/dev_selected_data0426.json"
        else:
            huggingface_hub = ""

        if task.__contains__("formlm"):
            validation_data = load_json(huggingface_hub)["data"]
            dataset = []
            for data in validation_data:
                dataset.append(data)
        else:
            dataset = load_dataset(huggingface_hub, ignore_verifications=True)

    os.makedirs(f"./generated/{task}/raw/", exist_ok=True)
    with open(f"./generated/{task}/raw/{split_set}.jsonl", "w") as outfile:
        if task.__contains__("formlm") is False:
            for example in dataset[split_set]:
                outfile.write(json.dumps(example) + "\n")
        else:
            for example in dataset:
                outfile.write(json.dumps(example) + "\n")


def save_jsonl(objective: str, task: str, split_set: str, content_list: list):
    os.makedirs(f"./generated/{task}/{objective}/", exist_ok=True)
    with open(f"./generated/{task}/{objective}/{split_set}.jsonl", "w") as outfile:
        for content in content_list:
            outfile.write(json.dumps(content) + "\n")


def save_unified_jsonl(output_path: str, unified_dict: dict):
    os.makedirs(output_path, exist_ok=True)
    content_list = []
    split = 0
    with open(f"{output_path}/validation.txt", "a", encoding="utf-8") as log_f:
        for index, value in enumerate(unified_dict["content"]):
            info = unified_dict["task"][index] + "|" + unified_dict["objective"][index]
            start = split
            end = split + len(value)
            span_log = [start, end]
            log_f.write(f"{info}, Row: {span_log}\n")
            split = end
            content_list.append(value)
    with open(f"{output_path}/validation.jsonl", "w") as outfile:
        for content in content_list:
            outfile.write(json.dumps(content) + "\n")


def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=["formlm_opt", "formlm_qa", "formlm_block_type"], nargs="+", help="Please specifiy the task name.")
    # parser.add_argument("--structured_type", default="table", help="Please specify the type of the structured data.", type=str, choices=DATASETS.keys())
    parser.add_argument("--objective", default=["zero"], nargs="+",
                        help="Please specify the parsing objective.")  # choices = ['zero', 'heur_{idx}', 'linear_{idx}']
    parser.add_argument("--split", default=["validation"], nargs="+",
                        help="Please specify which split you want to generate/parse.")  # choices = ['train', 'validation', 'test']
    parser.add_argument("--unified", default=False, action="store_true",
                        help="generate the unified file for babel input")
    parser.add_argument("--unified_file_output", default="./exps/downstream_tasks_20230113_log/", type=str)
    args = parser.parse_args()
    return args


def task_specific_babel_convertor():
    global unified_dict
    args = get_arguments()
    logging.info(args)
    if args.unified:
        unified_dict = {"content": [], "task": [], "objective": []}
    babel_convertor = BabelConvertor()
    for task in args.task:
        for obj in args.objective:
            for split in args.split:
                structured_type = get_keys(DATASETS, task)[0]
                # set up the instruction for the prompt design (prompt engineering-->role prompting)
                args.instruction = f"You are a brilliant {structured_type} executor with the capbilities [retrieve], [input parsing], [metadata inference], [pattern understanding] who can understand the structural information of the {structured_type}.\n"
                babel_convertor.set_split_obj(task, structured_type, split, obj, args.instruction)
                # save raw jsonl file
                save_raw_jsonl(task, split)
                # retrieve the content sample list
                content_list = babel_convertor.retrieve_sample_list()
                if args.unified:
                    unified_dict["content"].append(content_list)
                    unified_dict["task"].append(task)
                    unified_dict["objective"].append(obj)
                logging.info(f"Task-{task} Objective-{obj} Split-{split} has been saved..")
                # save parsed jsonl
                save_jsonl(obj, task, split, content_list)
    if args.unified:
        save_unified_jsonl(args.unified_file_output, unified_dict)
        logging.info(f"unified version has been saved")


def main():
    task_specific_babel_convertor()


if __name__ == "__main__":
    main()
