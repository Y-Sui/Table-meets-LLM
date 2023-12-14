import csv
import json
from typing import List
import jsonlines

import datasets
import os

_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

_HOMEPAGE = "https://gittables.github.io/"

_LICENSE = ""

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URL = "https://huggingface.co/datasets/yuansui/GitTables/resolve/main/"  # be attention to the url, use resolve instead of blob
_URLS = {
    "dbpedia": {
        "train": _URL + "dbpedia" + "_train.jsonl",
        "dev": _URL + "dbpedia" + "_val.jsonl",
        "test": _URL + "dbpedia" + "_test.jsonl",
    },
    "schema": {
        "train": _URL + "schema" + "_train.jsonl",
        "dev": _URL + "schema" + "_val.jsonl",
        "test": _URL + "schema" + "_test.jsonl",
    }
}


class GitTablesConfig(datasets.BuilderConfig):
    """GitTablesConfig for GitTables"""

    def __init__(self, features, data_url, citation, url, label_classes=("False", "True"), **kwargs):
        """BuilderConfig for SuperGLUE.

        Args:
        features: *list[string]*, list of the features that will appear in the
            feature dict. Should not include "label".
        data_url: *string*, url to download the zip file from.
        citation: *string*, citation for the data set.
        url: *string*, url for information about the data set.
        label_classes: *list[string]*, the list of classes for the label if the
            label is present as a string. Non-string labels will be cast to either
            'False' or 'True'.
        **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        # 1.0.1: Fixed non-nondeterminism in ReCoRD.
        super().__init__(version=datasets.Version("1.0.1"), **kwargs)
        self.features = features
        self.label_classes = label_classes
        self.data_url = data_url
        self.citation = citation
        self.url = url


class GitTables(datasets.GeneratorBasedBuilder):
    """GitTables benchmark"""
    DEFAULT_CONFIG_NAME = "schema"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    BUILDER_CONFIGS = [
        GitTablesConfig(
            name="dbpedia",
            description="Subsets of 1M gittables for column type classification with dbpedia",
            features=["id", "table_id", "target_column", "annotation_id", "annotation_label", "table_text",
                      "column_text"],
            data_url=_URLS["dbpedia"],
            citation=_CITATION,
            url=""
        ),
        GitTablesConfig(
            name="schema",
            description="Subsets of 1M gittables for column type classification with schema",
            features=["id", "table_id", "target_column", "annotation_id", "annotation_label", "table_text",
                      "column_text"],
            data_url=_URLS["schema"],
            citation=_CITATION,
            url=""
        )
    ]

    def _info(self):

        features = datasets.Features(
            {
                "id": datasets.Value("int32"),
                "table_id": datasets.Value("string"),
                "target_column": datasets.Value("string"),
                "annotation_id": datasets.Value("string"),
                "annotation_label": datasets.Value("string"),
                "table_text": datasets.Value("string"),
                "column_text": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  # Here we define them above because they are different between the two configurations
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls_to_download = _URLS[self.config.name]
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        # These kwargs will be passed to _generate_examples
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["dev"],
                    "split": "dev"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test"
                }
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with jsonlines.open(filepath, mode="r") as reader:
            for key, row in enumerate(reader):
                data = json.loads(row)
                if self.config.name == "dbpedia":
                    table_text = {}
                    if data["table_text"] != None:
                        for i in range(35):
                            col_info = data["table_text"].get(f"col{i}", "")
                            if col_info != "":
                                table_text[f"col{i}"] = col_info
                        # Yields examples as (key, example) tuples
                    yield key, {
                        "id": data["id"],
                        "table_id": data["table_id"],
                        "target_column": data["target_column"],
                        "column_text": data["column_text"],
                        "annotation_id": data["annotation_id"],
                        "annotation_label": data["annotation_label"],
                        "table_text": str(table_text)
                    }
                else:
                    table_text = {}
                    for i in range(35):
                        col_info = data["table_text"].get(f"col{i}", "")
                        if col_info != "":
                            table_text[f"col{i}"] = col_info
                    yield key, {
                        "id": data["id"],
                        "table_id": data["table_id"],
                        "target_column": data["target_column"],
                        "column_text": data["column_text"],
                        "annotation_id": data["annotation_id"],
                        "annotation_label": data["annotation_label"],
                        "table_text": str(table_text),
                    }