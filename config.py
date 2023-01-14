DATASETS = {
    "table": ["tabfact", "sqa", "hybridqa", "gittables", "feverous", "totto"],
    "database": ["spider", "cosql", "logic2text", "sql2text"],
    "form": ["formlm"],
    "knowledge_graph": ["webqsp", "dart"] # todo add comwebq
}

def get_heuristics(data_type):
    return {
        "heur_0": f"Generate structural information that will be beneficial for understanding {data_type}: \n",
        "heur_1": f"Let's think step by step: \n", # zero-shot-CoT
        "heur_2": f"Add structural information for {data_type}: \n",
        "heur_3": f"Let's solve this problem by splitting it into steps: \n", # zero-shot-CoT
        "heur_4": f"First analyze, \n",
        "heur_5": f"The answer is after the structural information of {data_type}, \n",
        "heur_6": f"Before we dive into the answer, think about the structural information of {data_type}, \n",
        "heur_7": f"Give attention to structural information of {data_type}, \n"
    }

def get_requests(data):
    requests_dict = {
        "tabfact": "Verify the statement against the seen tables, output 1 when it's entailed, 0 when it's refuted, \n",
        "cosql": "Generate SQL based on the utterance and database information, \n",
        "dart": "Generate natural language text from the given RDF triplets: \n",
        "feverous": "Verified the claim with the evidence in the forms of sentences and cells from tables, the answer should be 0 if refutes, or 1 if supports: \n",
        "gittables": "Please annotate table column with entity name from DBpedia and Schema.org Ontologies: \n",
        "hybridqa": "Aggregate both tabular information and text information to answer the question (Do not repeat the question, and shorten the answers as much as possible): \n",
        "logic2text": "Generate natural language description based on the logic_form and the tabular information: \n",
        "multi_woz_dia": "Predict each dialog utterance's intent (one by one) (return short answers): \n",
        "multi_woz_intent": "Predict overall conversation's intent from the user side (return short answers): \n",
        "spider": "Generate SQL from the given natural language question: \n",
        "sqa": "Answer the question with the tabular information: \n",
        "sql2text": "Convert the given SQL to natural language query: \n",
        "totto": "Generate natural language description for each highlighted part of the table: \n",
        "webqsp": "Answer the question using entity id instead of entity name, with the mentioned entity and knowledge graph information: \n"
    }
    return requests_dict[data]