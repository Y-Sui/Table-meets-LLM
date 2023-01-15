import json
import os

import wandb
from revChatGPT.ChatGPT import Chatbot # https://github.com/acheong08/ChatGPT

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

TASK = ["hybridqa"]
with open("config.json", "r") as f:
    session_token = json.load(f)["session_token"]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def retrieval_ans(prompt):
    chatbot = Chatbot({
        "session_token": session_token
    }, conversation_id=None, parent_id=None)
    response = chatbot.ask(prompt, conversation_id=None, parent_id=None)  # You can specify custom conversation and parent ids. Otherwise it uses the saved conversation (yes. conversations are automatically saved)
    return response["message"]

def main():
    tasks = os.listdir("../../generated")
    tasks = list(filter(lambda x: x in TASK, tasks))
    wandb.init(project="LLMs + Structured Data", job_type="eval", entity="dki", group="Prompt_insights", name=f"sample_0_221230_chatgpt")
    prediction_table = wandb.Table(columns=["task", "prompt", "completion", "groundtruth"])
    grd, pred = [], []
    results = {}
    for task in tasks:
        prompt_file = f"../../generated/{task}/zero/validation.jsonl"
        with open(prompt_file, "r", encoding="utf-8") as input_file:
            for line in input_file:
                sample = json.loads(line)
                prompt = sample["prompt"] # prompt_test only consider the first prompt
                ground_truth = sample["completion"]
                try:
                    completion = retrieval_ans(prompt)
                except:
                    completion = ""

                results["prompt"] = prompt
                results["pred"] = completion
                results["grd"] = ground_truth

                import pprint
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(results)

                pred.append(completion)
                grd.append(ground_truth)
                prediction_table.add_data(task, prompt, completion, ground_truth)
                with open(f"./{task}.txt", "a", encoding="utf-8") as txt_f:
                    # txt_f.write(task + "\n============================\n")
                    txt_f.write(prompt + "\n")
                    txt_f.write(completion + "\n")
                    txt_f.write(str(ground_truth) + "\n")
        # Acc
        acc = 0
        for i in range(len(pred)):
            if pred[i].__contains__("supported"):
                pred[i] = "1"
            if pred[i].__contains__("provided"):
                pred[i] = "0"
            if str(pred[i]) == str(grd[i]):
                acc += 1
        acc /= len(pred)
        print(acc)

    wandb.log({"predictions": prediction_table, "metrics": acc})
    wandb.finish()

    # # test
    # prompt = "Visualize the table: [[{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': '#'}, {'column_span': 2, 'is_header': True, 'row_span': 1, 'value': 'Governor'}, {'column_span': 1, 'is_header': True, 'row_span': 1, 'value': 'Took Office'}, {'column_span': 1, 'is_header': True, 'row_span': 1, 'value': 'Left Office'}, {'column_span': 1, 'is_header': True, 'row_span': 1, 'value': 'Lt. Governor'}, {'column_span': 1, 'is_header': True, 'row_span': 1, 'value': 'Party'}, {'column_span': 1, 'is_header': True, 'row_span': 1, 'value': 'Notes'}], [{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': '74'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': '-'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Robert Kingston Scott'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'July 6, 1868'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'December 7, 1872'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Lemuel Boozer Alonzo J. Ransier'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Republican'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': ''}], [{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': '75'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': '-'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Franklin J. Moses, Jr.'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'December 7, 1872'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'December 1, 1874'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Richard Howell Gleaves'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Republican'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': ''}], [{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': '76'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': '-'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Daniel Henry Chamberlain'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'December 1, 1874'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'December 14, 1876'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Richard Howell Gleaves'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Republican'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Claimed Governorship after 1876 election'}], [{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': '77'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': '-'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Wade Hampton III'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'December 14, 1876'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'February 26, 1879'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'William Dunlap Simpson'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Democratic'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Resigned'}], [{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': '78'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': '-'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'William Dunlap Simpson'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'February 26, 1879'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'September 1, 1880'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'John D. Kennedy'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Democratic'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Not elected'}], [{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': '79'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': '-'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Thomas Bothwell Jeter'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'September 1, 1880'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'November 30, 1880'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'vacant'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Democratic'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': ''}], [{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': '80'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': '-'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Johnson Hagood'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'November 30, 1880'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'December 1, 1882'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'John D. Kennedy'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Democratic'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': ''}], [{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': '81'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': '-'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Hugh Smith Thompson'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'December 1, 1882'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'July 10, 1886'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'John Calhoun Sheppard'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Democratic'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Resigned'}], [{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': '82'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': '-'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'John Calhoun Sheppard'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'July 10, 1886'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'November 30, 1886'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'vacant'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Democratic'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Not elected'}], [{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': '83'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': '-'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'John Peter Richardson III'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'November 30, 1886'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'December 4, 1890'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'William Mauldin'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Democratic'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': ''}], [{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': '84'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': '-'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Benjamin Ryan Tillman'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'December 4, 1890'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'December 4, 1894'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Eugene Gary W.H. Timmerman'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Democratic'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': ''}], [{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': '85'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': '-'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'John Gary Evans'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'December 4, 1894'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'January 18, 1897'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'W.H. Timmerman'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': 'Democratic'}, {'column_span': 1, 'is_header': False, 'row_span': 1, 'value': ''}]]"
    # print(retrieval_ans(prompt))

if __name__ == "__main__":
    main()