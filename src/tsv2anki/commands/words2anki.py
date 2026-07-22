#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR

    Sebastien Le Maguer <lemagues@surface>

DESCRIPTION

LICENSE
    This script is in the public domain, free from copyrights or restrictions.
    Created:  4 August 2024
"""

# Python
import argparse

# Data / Processing
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from tsv2anki.helpers.theme import color_from_string

from tsv2anki.helpers.common import AnkiConnector


def add_subparsers(subparsers):
    parser = subparsers.add_parser("words2anki", help="Generate the anki package for word learning")

    # Add options
    parser.add_argument("-n", "--dry-run", action="store_true", help="Activate dry-run mode")
    parser.add_argument("-t", "--tags", default=None, help="List of tags comma separated")

    # Add arguments
    parser.add_argument(
        "input_tsv",
        help="The input TSV file which should contains the following columns ['Category', 'Word', 'Translated Word', 'Example', 'Translated Example']",
    )
    parser.add_argument("deck", help="The name of the deck")

    parser.set_defaults(func=main)


def main(args: argparse.Namespace):

    # Load dataframe
    df = pd.read_csv(args.input_tsv, sep="\t")

    print(df)
    anki_connector = AnkiConnector()
    basic_model = anki_connector.ensure_basic_model()

    with logging_redirect_tqdm():
        for i_row, row in tqdm(
            df.iterrows(),
            total=df.shape[0],
            position=0,
            leave=False,
        ):
            row = row.fillna("")
            category = str(row["Category"])

            tags = []
            if args.tags is not None:
                tags = [s.strip().lower() for s in args.tags.split(",")]

            fields = {
                "Front": str(row["Word"]).replace("*", ""),
                "Back": str(row["Translated Word"]).replace("*", ""),
                "Category": "Word",
                "SubCategory": str(row["Category"]),
                "Example": str(row["Example"]),
                "TranslatedExample": str(row["Translated Example"]).replace("*", ""),
                "CategoryColor": color_from_string("Word".lower().strip(), True),
                "SubCategoryColor": color_from_string(category.lower().strip(), False),
                "tags": ["words", category.replace(" ", "_").lower()] + tags,
            }

            anki_connector.upsert_basic("Word", args.deck, basic_model, fields, args.dry_run)
