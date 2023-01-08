import argparse
import json
import os

def concat_json(args):
    # concat_json_file
    concat_data = []
    split = 0
    tasks = os.listdir("../generated")
    if args.job == "all_individual_tasks":
        tasks = list(filter(
            lambda x: x not in ["anti_phishing", "ms_forms_response", "ms_forms_templates_content", "public_form_intent"],
            tasks))
        with open("../exps/all_individual_tasks_20221227.txt", "a", encoding="utf-8") as log_f:
            for task in tasks:
                dirs = os.listdir(f"../generated/{task}")
                for dir in dirs:
                    filenames = os.listdir(f"../generated/{task}/{dir}")
                    for filename in filenames:
                        if filename == "validation.jsonl":
                            filepath = os.path.join(f"../generated/{task}/{dir}", filename)
                            with open(filepath, "r") as f:
                                start = split
                                end = split + len(f.readlines())
                                span_log = [start, end]
                                log_f.write(f"{filepath}, Row: {span_log}\n")
                                split = end
                            with open(filepath, "r") as f:
                                for line in f.readlines():
                                    concat_data.append(json.loads(line))

        with open("../exps/all_individual_tasks_20221227.jsonl", "w", encoding="utf-8") as w_f:
            for item in concat_data:
                w_f.write(json.dumps(item) + "\n")
    elif args.job == "few_shot_samples":
        tasks = list(filter(
            lambda x: x in ["spider", "totto", "tabfact"], tasks
        ))
        with open("../exps/few_shot_samples_20230105_log/few_shot_samples.txt", "a", encoding="utf-8") as log_f:
            for task in tasks:
                for dir in ["few_shot_2", "few_shot_4"]:
                    filepath = f"../../gpt3_dataset_generation/generated/{task}/{dir}/validation.jsonl"
                    with open(filepath, "r") as f:
                        start = split
                        end = split + len(f.readlines())
                        span_log = [start, end]
                        log_f.write(f"{filepath}, Row: {span_log}\n")
                        split = end
                    with open(filepath, "r") as f:
                        for line in f.readlines():
                            concat_data.append(json.loads(json.loads(line))) # since line is type of string instead of json type
        with open("../exps/few_shot_samples_20230105_log/few_shot_samples.jsonl", "w", encoding="utf-8") as w_f:
            for item in concat_data:
                w_f.write(json.dumps(item) + "\n")


def get_task_span():
    task_file, row_index = [], []
    with open("./all_individual_tasks_20221227_log/all_individual_tasks_20221227.txt", "r") as txt_file:
        task_spans = txt_file.readlines() # ../generated/dart/heur_7\validation.jsonl, Row: [31076, 33844]
    for i in range(len(task_spans)):
        if not task_spans[i].split(",")[0].__contains__("zero"):
            continue
        task_file.append(task_spans[i].split(",")[0])
        row_index_number = task_spans[i].split(": ")[-1].replace("[", "").replace("]", "").replace("\n", "").split(",") # list
        row_index.append({"start": int(row_index_number[0]), "end": int(row_index_number[1])})
    return task_file, row_index

def retrieve_pair(output_dir):
    task_file, row_index = get_task_span()
    pair_list = []
    for idx in range(len(task_file)):
        pred, grd, err = [], [], []
        task = task_file[idx].split("/")[2] # task name, like spider, sqa, sql2text
        # load output
        with open(output_dir, 'rb') as f:
            for line in f.readlines()[row_index[idx]["start"]: row_index[idx]["end"]]:
                obj = json.loads(line)
                generated = obj["choices"][0]["text"]
                pred.append(generated)
        pair_list.append({"task": task, "prediction": pred, "ground_truth": grd, "error_list": err})
    return pair_list

def phase_2_processing():
    raw_data = []
    structural_info = []
    updated_prompt = []
    answer = []
    with open("../exps/all_individual_tasks_20221227_log/all_individual_tasks_20221227.jsonl", "r", encoding="utf-8") as f1:
        i = 0
        for line in f1:
            i += 1
            raw_input = json.loads(line)
            if raw_input["prompt"].split("\n")[0] != "<request>":
                raw_input["prompt"] = "<request>\n" + raw_input["prompt"][:-5] # delete the tail 4 char
                # print(i)
            raw_data.append(raw_input)

    with open("../exps/all_individual_tasks_20221227_log/text003/all_individual_tasks_20221227_samples.0.jsonl", "r", encoding="utf-8") as f2:
        for line in f2:
            structural_info.append(json.loads(line)["choices"][0]["text"])

    with open("../exps/all_individual_tasks_20230101_log/all_individual_tasks_20230101_samples_phase2.0_text003_samples.0.jsonl", "r", encoding="utf-8") as phase_file:
        for line in phase_file:
            answer.append(json.loads(line)["choices"][0]["text"])

    with open("../exps/all_individual_tasks_20221230_log/all_individual_tasks_20221230_samples_phase2.0_text003.jsonl", "w", encoding="utf-8") as f3:
        # concat phase1 output to generate phase2
        for i in range(len(raw_data)):
            request_idx = raw_data[i]["prompt"].split("\n")[1]
            if i < 10340:
                update_prompt = "Generate SQL based on the utterance and database information: "
            elif i < 36612:
                update_prompt = "Generate natural language text from the given RDF triplets: "
            elif i < 75177:
                update_prompt = "Verified the claim with the evidence in the forms of sentences and cells from tables, the answer should be 0 if refutes, or 1 if supports: "
            elif i < 79695:
                update_prompt = "Please annotate table column with entity name from DBpedia and Schema.org Ontologies: "
            elif i < 110529:
                update_prompt = "Aggregate both tabular information and text information to answer the question (Do not repeat the question, and shorten the answers as much as possible): "
            elif i < 120384:
                update_prompt = "Generate natural language description based on the logic_form and the tabular information: "
            elif i < 129384:
                update_prompt = "Predict each dialog utterance's intent (one by one) (return short answers): "
            elif i < 138384:
                update_prompt = "Predict overall conversation's intent from the user side (return short answers): "
            elif i < 147690:
                update_prompt = "Generate SQL from the given natural language question: " # spider
            elif i < 168075:
                update_prompt = "Answer the question with the tabular information: "
            elif i < 180675:
                update_prompt = "Convert the given SQL to natural language query: "
            elif i < 295803:
                update_prompt = "Verify the statement against the seen tables, output 1 when it's entailed, 0 when it's refuted" # tabfact
            elif i < 419940:
                update_prompt = "Generate natural language description for each highlighted part of the table: " # totto
            elif i < 422676:
                update_prompt = "Answer the question using entity id instead of entity name, with the mentioned entity and knowledge graph information: "
            raw_data[i]["prompt"] = raw_data[i]["prompt"].replace(request_idx, update_prompt)
            raw_data[i]["prompt"] = raw_data[i]["prompt"] + "\n<structure>\n" +structural_info[i] + "\n===>"
            f3.write(json.dumps(raw_data[i]) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="atp_20221206_instruction")
    parser.add_argument("--topK", type=int, default="4")
    parser.add_argument("--job", type=str, default="few_shot_samples")
    args = parser.parse_args()
    concat_json(args)
    # phase_2_processing()

if __name__ == "__main__":
    main()