import argparse
import os.path
import json
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="atp_20221206_instruction")
parser.add_argument("--job", type=str, choices=["finetuned", "inference"], default="finetuned")
parser.add_argument("--task", type=str, choices=["atp", "formlm", "msform"], default="atp")
args = parser.parse_args()

Task = {
    "atp": {
        "src": f'./generated/anti_phishing/{args.file_name}/anti_phishing_test.jsonl',
        "pred": f'./output/{args.file_name}/anti_phishing_test_{args.job}.jsonl' # load the specific idx file
    },
    "formlm":{
        "src": f'./generated/public_form_intent/{args.file_name}/public_form_intent_test.jsonl',
        "pred": f'./output/{args.file_name}/public_form_intent_test_{args.job}.jsonl' # load the specific idx file
    },
    "msform":{
        "src": f'./generated/ms_forms_templates_content/{args.file_name}/ms_forms_templates_test.jsonl',
        "pred": f'./output/{args.file_name}/ms_forms_templates_test_{args.job}.jsonl' # load the specific idx file
    }
}

def retrieve_pair(args):
    predicate, ground_truth, error_list = [], [], []
    with open(Task[args.task]["pred"], 'rb') as f:
        for line in f:
            obj = json.loads(line)
            generated = obj["choices"][0]["text"].split(",")[0]
            predicate.append(generated)
    with open(Task[args.task]["src"], "rb") as f:
        for line in f:
            obj = json.loads(line)
            if args.task == "formlm":
                completion = obj["completion"].split("<")[0]
            elif args.task == "atp":
                completion = obj["completion"].split(",")[0] # Yes or No
            elif args.task == "msform":
                completion = obj["completion"] # lack labelling here
            ground_truth.append(completion)
    with open(Task[args.task]["src"], "rb") as f:
        i = 0
        for line in f:
            obj = json.loads(line)
            if ground_truth[i] != predicate[i]:
                error_list.append(obj["prompt"])
            i += 1

    return predicate, ground_truth, error_list


def evaluation(args):
    predicate, ground_truth, error_list = retrieve_pair(args)
    print(error_list[:5])
    eva = {}

    if args.task == "atp":
        if args.job == "inference":
            for i in range(len(ground_truth)):
                if predicate[i].__contains__(ground_truth[i]):
                    predicate[i] = ground_truth[i]
                elif predicate[i].__contains__("is phishing"):
                    predicate[i] = "Yes"
                elif predicate[i].__contains__("is not phishing"):
                    predicate[i] = "No"
                else:
                    predicate[i] = "No"
        average = "binary"
    else:
        if args.job == "inference":
            for i in range(len(ground_truth)):
                if predicate[i].__contains__(ground_truth[i]):
                    predicate[i] = ground_truth[i]
                else:
                    predicate[i] = "Unknown"
        average = "macro"

    # print confusion matrix:
    conf_matrix = confusion_matrix(ground_truth, predicate)
    print(f"confusion_matrix: {conf_matrix}")
    print(f"tn, fp, fn, tp = {conf_matrix.ravel()}")
    print(predicate[:10])
    print(ground_truth[:10])

    acc = 0
    for i in range(len(ground_truth)):
        if predicate[i].__contains__(ground_truth[i]):
            acc += 1
    acc = acc / len(ground_truth)


    f1 = f1_score(ground_truth, predicate, average=average, zero_division=0, pos_label="Yes")
    recall = recall_score(ground_truth, predicate, average=average, zero_division=0, pos_label="Yes")
    precision = precision_score(ground_truth, predicate, average=average, zero_division=0, pos_label="Yes")
    print(f"acc, f1, recall, prec = {acc, f1, recall, precision}")

    eva["f1"] = f1
    eva["acc"] = acc
    eva["recall"] = recall
    eva["precision"] = precision
    eva["conf_matrix"] = str(conf_matrix) + f" tn, fp, fn, tp = {conf_matrix.ravel()}"

    # with open(os.path.join(Task[args.task]["pred"].replace(Task[args.task]["pred"].split("/")[-1], ''), f"{args.file_name}_evaluation.json"), "w") as file:
    #     json_dump = json.dumps(eva)
    #     file.write(json_dump)

    with open(os.path.join("../gpt3_dataset_generation/output/evaluation", f"{args.file_name}_{args.job}_evaluation.json"), "w") as file:
        json_dump = json.dumps(eva)
        file.write(json_dump)

def main():
    evaluation(args)

if __name__ == "__main__":
    main()


