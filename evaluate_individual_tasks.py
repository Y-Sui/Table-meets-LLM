import argparse
import copy
import difflib
import os.path
import json
import re
import pprint
pp = pprint.PrettyPrinter(indent=4)

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

parser = argparse.ArgumentParser()
parser.add_argument("--job", type=str, choices=["linear", "zero-shot", "all", "oneshot"], default="oneshot")
parser.add_argument("--log_file_path", default="downstream_tasks_20230119_manual_log/0_1_2_3", type=str, help="Please indicate the log file path")
parser.add_argument("--model", default=["text003"], nargs="+", help="Please give the model results you want to evaluate")
args = parser.parse_args()


def get_task_span():
    task_file, row_index = [], []
    with open(f"exps/{args.log_file_path}/validation.txt", "r") as txt_file:
        task_spans = txt_file.readlines() # ../generated/dart/heur_7\validation.jsonl, Row: [31076, 33844]
    for i in range(len(task_spans)):
        if args.job == "zero-shot":
            if not task_spans[i].split(",")[0].__contains__("zero"):
                continue
        elif args.job == "linear":
            if not task_spans[i].split(",")[0].__contains__("linear"):
                continue
        elif args.job == "oneshot":
            if not task_spans[i].split(",")[0].__contains__("oneshot"):
                continue
        task_file.append(task_spans[i].split(",")[0])
        row_index_number = task_spans[i].split(": ")[-1].replace("[", "").replace("]", "").replace("\n", "").split(",") # list
        row_index.append({"start": int(row_index_number[0]), "end": int(row_index_number[1])})
    return task_file, row_index


def retrieve_pair():
    task_file, row_index = get_task_span()
    pair_list = []
    for idx in range(len(task_file)):
        pred, grd, err = [], [], []
        task = task_file[idx].split("|")[0]  # task name, like spider, sqa, sql2text
        split = task_file[idx].split("|")[1]
        path_files = os.listdir(f"exps/{args.log_file_path}")
        for i in range(len(path_files)):
            if path_files[i].__contains__("text003"):
                model_output = path_files[i]
        try:
            # load predict file
            with open(f"exps/{args.log_file_path}/{model_output}/output_dataset/validation_samples.0.jsonl", 'rb') as f:
                for line in f.readlines()[row_index[idx]["start"]: row_index[idx]["end"]]:
                    obj = json.loads(line)
                    generated = obj['samples'][0]
                    pred.append(generated)
        except:
            print("Error, the output has not generated")
        # load ground_truth
        with open(f"exps/{args.log_file_path}/validation.jsonl", "rb") as f:
            for line in f.readlines()[row_index[idx]["start"]: row_index[idx]["end"]]:
                obj = json.loads(line)
                completion = obj["completion"] # lack labelling here
                grd.append(completion)
        pair_list.append({"task": task, "split": split, "prediction": pred, "ground_truth": grd})
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
    # bleu_weights
    weights_list={
                  "BLEU-1": (1, 0, 0, 0),
                  "BLEU-2": (0.5, 0.5, 0, 0),
                  "BLEU-3": (0.33, 0.33, 0.34, 0),
                  "BLEU-4": (0.25, 0.25, 0.25, 0.25)
    }
    for pair in pair_list:
        raw_pred = pair["prediction"]
        updated_pred = copy.deepcopy(raw_pred)
        ground_truth = pair["ground_truth"]
        for i in range(len(raw_pred)):
            raw_pred[i] = raw_pred[i].strip()
            updated_pred[i] = updated_pred[i].strip()
            ground_truth[i] = ground_truth[i].strip()

        if pair["task"] == "sqa":
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
            for bleu_metric, weights in enumerate(weights_list):
                bleu = 0
                for i in range(len(raw_pred)):
                    references = []
                    for j in range(len(ground_truth[i].split("\n"))):
                        references.append(ground_truth[i].split("\n")[j].split())
                    processed_candidates = raw_pred[i].strip("\n").replace("Line 1: ", "").replace("1. ", "").replace("2. ", "").replace("Line 2: ", "")
                    candidates = []
                    try:
                        for j in range(len(processed_candidates.split("\n"))):
                            candidates.append(processed_candidates.split("\n")[j].split())
                    except:
                        candidates.append(processed_candidates.split())
                    candidate_score = 0
                    for candidate in candidates:
                        candidate_score += corpus_bleu([references], [candidate], weights=weights_list[weights])
                    bleu += candidate_score
                bleu /= len(raw_pred)
                if args.job == "zero-shot":
                    results["totto"] = bleu
                elif args.job == "linear":
                    results[f"totto_{log_idx}"] = bleu
                    log_idx += 1
                else:
                    results[f"totto_{pair['split']}_{weights}"] = bleu

        elif pair["task"] == "feverous":
            # Acc
            acc = 0
            for i in range(len(raw_pred)):
                try:
                    raw_pred[i] = raw_pred[i].split("\n")[0].strip()
                except:
                    raw_pred[i].strip()
                if raw_pred[i].__contains__("supported") or raw_pred[i].__contains__("1"):
                    updated_pred[i] = "1"
                if raw_pred[i].__contains__("0"):
                    updated_pred[i] = "0"
                if str(updated_pred[i]) == str(ground_truth[i].strip()):
                    acc += 1
            acc /= len(raw_pred)
            results[f"feverous_{pair['split']}"] = acc

        elif pair["task"] == "tabfact":
            # Acc
            acc = 0
            for i in range(len(raw_pred)):
                try:
                    raw_pred[i] = raw_pred[i].split("\n")[0].strip()
                except:
                    raw_pred[i].strip()
                if str(raw_pred[i]) == str(ground_truth[i].strip()):
                    acc += 1
            acc /= len(raw_pred)
            results[f"tabfact_{pair['split']}"] = acc

    with open(f"output/evaluation/{args.log_file_path.split('/')[0]}_evaluation.json", "w") as file:
        json_dump = json.dumps(results)
        file.write(json_dump)


    pp.pprint(args.log_file_path.split('/')[1])
    pp.pprint(results)

def main():
    pair_list = retrieve_pair()
    evaluate_individual_task(pair_list)

if __name__ == "__main__":
    main()


