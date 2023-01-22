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

    wandb.log({"predictions": prediction_table})
    wandb.finish()

if __name__ == "__main__":
    main()