DATASETS = {
    "table": ["tabfact", "sqa", "hybridqa", "gittables", "feverous", "totto"],
    "database": ["spider", "cosql", "logic2text", "sql2text"],
    "form": ["formlm", "formlm_opt", "formlm_qa", "formlm_block_type"],
    "knowledge_graph": ["webqsp", "dart"]  # todo add comwebq
}

MODELS = {
    "chatgpt": "chat002",
    "instruct-davinci-003": "text003",
    "instruct-davinci-002": "text002",
    "instruct-davinci-001": "text001",
}

TASKS = {
    "table": ["cell_lookup", "cell_lookup_pos", "column_retrieval", "row_retrieval", "scope_detection",
              "span_detection"],
    "form": ["block_dependency", "block_traversal"]
}

LINEARIZE_ = ["markdown", "markdown_grid", "html", "json", "latex", "nl_sep"]
LINEARIZE_0 = ["markdown_0", "markdown_grid_0", "html_0", "json_0", "latex_0", "nl_sep_0"]  # add structure mark
LINEARIZE_1 = ["markdown_1", "markdown_grid_1", "html_1", "json_1", "latex_1", "nl_sep_1"]  # add grammar
# LINEARIZE_2 = ["markdown_2", "markdown_grid_2", "html_2", "json_2", "latex_2", "nl_sep_2"] # change order
LINEARIZE_0_1 = ["markdown_0_1", "markdown_grid_0_1", "html_0_1", "json_0_1", "latex_0_1", "nl_sep_0_1"]
LINEARIZE = LINEARIZE_ + LINEARIZE_0 + LINEARIZE_1 + LINEARIZE_0_1


def get_heuristics(data_type, context_type="statement"):
    return {
        "heur_0": f"Generate structural information that will be beneficial for understanding {data_type}: \n",
        "heur_1": f"Let's think step by step: \n",  # zero-shot-CoT
        "heur_2": f"Add structural information for {data_type}: \n",
        "heur_3": f"Let's solve this problem by splitting it into steps: \n",  # zero-shot-CoT
        "heur_4": f"First analyze, \n",
        "heur_5": f"The answer is after the structural information of {data_type}, \n",
        "heur_6": f"Before we dive into the answer, think about the structural information of {data_type}, \n",
        "heur_7": f"Give attention to structural information of {data_type}, \n",
        "heur_8": f"Generate short format specification and description of the last {data_type} within five sentences. \n",
        "heur_9": f"Identify critical values and ranges of the last {data_type} related to the {context_type} within five sentences. \n",
        "heur_10": f"Describe structural information, patterns and statistics of the last {data_type} related to the {context_type} within five sentences. \n"
    }


def get_end_prompt():
    return {
        "heur_0": "The structural information are \n",
        "heur_1": "The chain of thought is \n",  # zero-shot-CoT
        "heur_2": "The structural information are: \n",
        "heur_3": "The thinking steps are\n",  # zero-shot-CoT
        "heur_4": "The chain of thought is, \n",
        "heur_5": "The structural information are, \n",
        "heur_6": "The structural information are, \n",
        "heur_7": "The structural information are, \n",
        "heur_8": "The format specification and description are \n",
        "heur_9": "The critical values and ranges are \n",
        "heur_10": "The structural information, patterns and statistics are \n"
    }


def get_requests(data):
    requests_dict = {
        "tabfact": "Verify the statement against the last table. Return 1 when it's entailed, 0 when it's refuted. Do not return any addition information.\n",
        "feverous": "Verified the claim with the evidence in the forms of sentences and cells from last table. The answer should be 0 if it refutes or 1 if it supports. \n",
        "hybridqa": "Aggregate tabular information and text information to answer the question (Do not repeat the question, and shorten the answer as much as possible). \n",
        "sqa": "Answer the question with the tabular information. \n",
        "totto": "Generate natural language description for each highlighted part of the last table line by line: \n",
    }
    return requests_dict[data]
