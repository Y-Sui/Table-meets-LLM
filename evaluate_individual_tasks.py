import argparse
import copy
import difflib
import os.path
import json
import re

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="atp_20221206_instruction")
parser.add_argument("--job", type=str, choices=["linear", "zero-shot", "all"], default="all")
args = parser.parse_args()


def get_task_span():
    task_file, row_index = [], []
    with open("exps/all_individual_tasks_20221227_log/all_individual_tasks_20221227.txt", "r") as txt_file:
        task_spans = txt_file.readlines() # ../generated/dart/heur_7\validation.jsonl, Row: [31076, 33844]
    for i in range(len(task_spans)):
        if args.job == "zero-shot":
            if not task_spans[i].split(",")[0].__contains__("zero"):
                continue
        elif args.job == "linear":
            if not task_spans[i].split(",")[0].__contains__("linear"):
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
        split = task_file[idx].split("/")[3]
        # load output
        with open(output_dir, 'rb') as f:
            for line in f.readlines()[row_index[idx]["start"]: row_index[idx]["end"]]:
                obj = json.loads(line)
                generated = obj["choices"][0]["text"]
                pred.append(generated)
        # load ground_truth
        with open("exps/all_individual_tasks_20221227_log/all_individual_tasks_20221227.jsonl", "rb") as f:
            for line in f.readlines()[row_index[idx]["start"]: row_index[idx]["end"]]:
                obj = json.loads(line)
                completion = obj["completion"] # lack labelling here
                grd.append(completion)
            # error_list
            i = 0
            for line in f:
                obj = json.loads(line)
                if grd[i] != pred[i]:
                    err.append({"task": task, "prompt": obj["prompt"], "completion": grd[i], "prediction": pred[i]})
                i += 1
        pair_list.append({"task": task, "split": split, "prediction": pred, "ground_truth": grd, "error_list": err})
    return pair_list


def f1_metric(ground_truth, predicate, average="micro"):
    return f1_score(ground_truth, predicate, average=average, zero_division=0, pos_label="Yes")


def reformat_sequence(sequence):
    re_ = '[^\*"/:;|?\\|<>" "],'
    sequence = re.sub(re_, "", sequence)
    sequence = sequence.replace(" ", "").upper()
    sequence = "".join(sequence)
    return sequence


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


def evaluate_individual_task(pair_list):
    results = {}
    log_idx = 0
    weights=(1, 0, 0, 0) # bleu_weights
    chencherry = SmoothingFunction() # smoothing function
    for pair in pair_list:
        raw_pred = pair["prediction"]
        updated_pred = copy.deepcopy(raw_pred)
        ground_truth = pair["ground_truth"]
        for i in range(len(raw_pred)):
            raw_pred[i] = raw_pred[i].strip()
            updated_pred[i] = updated_pred[i].strip()
            ground_truth[i] = ground_truth[i].strip()
        if pair["task"] == "webqsp":
            # F1 metrics
            for i in range(len(raw_pred)):
                if ground_truth[i].strip().__contains__(raw_pred[i].strip()):
                    updated_pred[i] = ground_truth[i] # mapping
            results[f"webqsp_{pair['split']}"] = f1_metric(ground_truth, updated_pred)
        elif pair["task"] == "spider":
            # Exact Set Match without Values: decompose each SQL into several clauses (set comparison in each SQL clause)
            for i in range(len(raw_pred)):
                raw_pred[i] = reformat_sequence(raw_pred[i])
                ground_truth[i] = reformat_sequence(ground_truth[i])
                if ground_truth[i].strip().__contains__(raw_pred[i].strip()):
                    updated_pred[i] = ground_truth[i]  # mapping
            results[f"spider_{pair['split']}"] = f1_metric(ground_truth, updated_pred)
        elif pair["task"] == "sqa":
            # Overall Acc
            acc = 0
            for i in range(len(raw_pred)):
                acc_sample = 0
                grd_split = ground_truth[i].split("|")
                for j in range(len(grd_split)):
                    if raw_pred[i].strip().__contains__(grd_split[j].strip()):
                        acc_sample += 1
                acc_sample = acc_sample / len(grd_split)
                acc += acc_sample
            acc = acc / len(raw_pred)
            results[f"sqa_{pair['split']}"] = acc
        elif pair["task"] == "hybridqa":
            # Acc
            acc = 0
            for i in range(len(raw_pred)):
                if raw_pred[i].strip().__contains__(ground_truth[i].strip()):
                    acc += 1
                else:
                    raw_split = raw_pred[i].strip().split(",")
                    for j in range(len(raw_split)):
                        if string_similar(ground_truth[i].strip(), raw_split[j]) > 0.85:
                            acc += 1
                            continue
            acc = acc / len(raw_pred)
            results[f"hybridqa_{pair['split']}"] = acc
        elif pair["task"] == "totto":
            # BLEU, BERTScore
            bleu = 0
            for i in range(len(raw_pred)):
                x, y = raw_pred[i].upper().strip().split(), ground_truth[i].upper().strip().split()
                bleu += sentence_bleu([raw_pred[i].upper().strip().split()], ground_truth[i].upper().strip().split(), weights, smoothing_function=chencherry.method1)
            bleu /= len(raw_pred)
            if args.job == "zero-shot":
                results["totto"] = bleu
            elif args.job == "linear":
                results[f"totto_{log_idx}"] = bleu
                log_idx += 1
        elif pair["task"] == "dart":
            # BLEU, BERTScore
            bleu = 0
            for i in range(len(raw_pred)):
                bleu += sentence_bleu([raw_pred[i].upper().strip().split()], ground_truth[i].upper().strip().split(), weights, smoothing_function=chencherry.method1)
            bleu /= len(raw_pred)
            results[f"dart_{pair['split']}"] = bleu
        elif pair["task"] == "sql2text":
            # BLEU, BERTScore
            bleu = 0
            for i in range(len(raw_pred)):
                bleu += sentence_bleu([raw_pred[i].upper().strip().split()], ground_truth[i].upper().strip().split(), weights, smoothing_function=chencherry.method1)
            bleu /= len(raw_pred)
            results[f"sql2text_{pair['split']}"] = bleu
        elif pair["task"] == "logic2text":
            # BLEU, BERTScore
            bleu = 0
            for i in range(len(raw_pred)):
                bleu += sentence_bleu([raw_pred[i].upper().strip().split()], ground_truth[i].upper().strip().split(), weights, smoothing_function=chencherry.method1)
            bleu /= len(raw_pred)
            results[f"logic2text_{pair['split']}"] = bleu
        elif pair["task"] == "feverous":
            # Acc
            acc = 0
            for i in range(len(raw_pred)):
                if raw_pred[i].__contains__("supported") or raw_pred[i].__contains__("1"):
                    updated_pred[i] = "1"
                if raw_pred[i].__contains__("0") or not raw_pred[i].__contains__("1") and not raw_pred[i].__contains__("0"):
                    updated_pred[i] = "0"
                if str(updated_pred[i].strip()) == str(ground_truth[i].strip()):
                    acc += 1
                else:
                    print(pair)
            acc /= len(raw_pred)
            results[f"feverous_{pair['split']}"] = acc
        elif pair["task"] == "cosql":
            # Exact Matching without values # BLEU
            # BLEU, BERTScore
            bleu = 0
            for i in range(len(raw_pred)):
                bleu += sentence_bleu([raw_pred[i].upper().strip().split()], ground_truth[i].strip().split(), weights, smoothing_function=chencherry.method1)
            bleu /= len(raw_pred)
            results[f"cosql_{pair['split']}"] = bleu
        elif pair["task"] == "multi_woz_dia":
            # F1
            for i in range(len(raw_pred)):
                if ground_truth[i].strip().__contains__(raw_pred[i].lower().replace(" ", "_").strip()):
                    updated_pred[i] = ground_truth[i]
            results[f"mul_woz_dia_{pair['split']}"] = f1_metric(ground_truth, updated_pred)
        elif pair["task"] == "multi_woz_intent":
            # F1
            for i in range(len(raw_pred)):
                if ground_truth[i].strip().__contains__(raw_pred[i].lower().replace(" ", "_").strip()):
                    updated_pred[i] = ground_truth[i]
            results[f"mul_woz_intent_{pair['split']}"] = f1_metric(ground_truth, updated_pred)
        elif pair["task"] == "tabfact":
            # Acc
            acc = 0
            for i in range(len(raw_pred)):
                if str(raw_pred[i].strip()) == str(ground_truth[i].strip()):
                    acc += 1
            acc /= len(raw_pred)
            results[f"tabfact_{pair['split']}"] = acc
        elif pair["task"] == "gittables":
            # BLUE
            bleu = 0
            for i in range(len(raw_pred)):
                bleu += sentence_bleu([raw_pred[i].upper().strip().split()], ground_truth[i].upper().strip().split(), weights, smoothing_function=chencherry.method1)
            bleu /= len(raw_pred)
            results[f"gittables_{pair['split']}"] = bleu

    if args.job == "zero-shot":
        with open("exps/all_individual_tasks_20221227_log/text003/all_individual_tasks_20221227_evaluation.json", "w") as file:
            json_dump = json.dumps(results)
            file.write(json_dump)
    elif args.job == "linear":
        with open("./all_individual_tasks_20221227_log/text003/linear_evaluation.json", "w") as file:
            json_dump = json.dumps(results)
            file.write(json_dump)
    with open("exps/all_individual_tasks_20230101_log/all_individual_tasks_20230101_evaluation.json", "w") as file:
        json_dump = json.dumps(results)
        file.write(json_dump)

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)

def main():
    pair_list = retrieve_pair(
        "exps/all_individual_tasks_20230101_log/all_individual_tasks_20230101_samples_phase2.0_text003_samples.0.jsonl")
    evaluate_individual_task(pair_list)

if __name__ == "__main__":
    main()


