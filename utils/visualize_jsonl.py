import json


def visualize_jsonl():
    file_path = "D:/StructuredLLM/exps/downstream_tasks_form_20230115_log/validation.jsonl"
    with open(file_path, "r") as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
    print(lines[:2])


def main():
    visualize_jsonl()


if __name__ == "__main__":
    main()