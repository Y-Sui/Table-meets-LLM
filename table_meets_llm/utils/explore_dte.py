import json

from tprompt.dte.embedding import DTEEmbedding
from tprompt.dte.generator import generate_embeddings
from tprompt.dte.download import download_dte
from tprompt.dte.retriever import retrieve
import os
import argparse
import random

# import model config
model_path = download_dte(use_cached=True)
dte = DTEEmbedding(os.path.join(model_path, "model.config"))
task = ["spider", "totto", "tabfact"] # downstream tasks
request = {
    "cosql": "Generate SQL based on the utterance and database information: ",
    "dart": "Generate natural language text from the given RDF triplets: ",
    "feverous": "Verified the claim with the evidence in the forms of sentences and cells from tables, the answer should be 0 if refutes, or 1 if supports: ",
    "hybridqa": "Aggregate both tabular information and text information to answer the question (Do not repeat the question, and shorten the answers as much as possible): ",
    "spider": "Generate SQL from the given natural language question: ",
    "sqa": "Answer the question with the tabular information: ",
    "tabfact": "Verify the statement against the seen tables, output 1 when it's entailed, 0 when it's refuted",
    "totto": "Generate natural language description for each highlighted part of the table: ",
    "webqsp": "Answer the question using entity id instead of entity name, with the mentioned entity and knowledge graph information: ",
}

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="atp_20221206_instruction")
parser.add_argument("--topK", type=int, default="4")
args = parser.parse_args()


def generate_samples_embedding(Q, A):
    return generate_embeddings(dte, Q, A)


def retrieve_few_shot(train_split, validation_split, topK, instruct, prompt_delimiter, completion_delimiter):
    random_sample_pair = random.sample(list(enumerate(train_split["Q"])), k=150) # random.sample
    train_split_q, train_split_a = [], []
    for idx, val in random_sample_pair:
        train_split_q.append(val)
        train_split_a.append(train_split["A"][idx])
    training_split_embedding = generate_samples_embedding(train_split_q, train_split_a)
    validation_split_embedding = generate_samples_embedding(validation_split["Q"], validation_split["A"])
    return retrieve(validation_split_embedding, training_split_embedding, batch_size=16, topK=topK, instruct=instruct, prompt_delimiter=prompt_delimiter, completion_delimiter=completion_delimiter)


def generate_few_shot_examples():
    train_Q, train_A, validation_Q, validation_A = [], [], [],[]
    for single_task in task:
        with open(f'../../generated/{single_task}/zero/train.jsonl', 'r') as train_file:
            for line in train_file:
                train_Q.append(json.loads(line)["prompt"])
                train_A.append(json.loads(line)["completion"])
        with open(f'../../generated/{single_task}/zero/validation.jsonl', "r") as valid_file:
            for line in valid_file:
                validation_Q.append(json.loads(line)["prompt"])
                validation_A.append(json.loads(line)["completion"])
        train_split = {"Q": train_Q, "A": train_A}
        valid_split = {"Q": validation_Q, "A": validation_A}
        few_shot_examples = retrieve_few_shot(train_split, valid_split, topK=args.topK, instruct=f"{request[single_task]}", prompt_delimiter="input: ", completion_delimiter="output: ")
        if not os.path.exists(f'../../generated/{single_task}/few_shot_{args.topK}'):
            os.makedirs(f'../../generated/{single_task}/few_shot_{args.topK}')
        with open(f'../../generated/{single_task}/few_shot_{args.topK}/validation.jsonl', "w") as save_file:
            for single_example in few_shot_examples:
                save_file.write(json.dumps(single_example) + "\n")


def main():
    generate_few_shot_examples()


if __name__ == "__main__":
    main()


