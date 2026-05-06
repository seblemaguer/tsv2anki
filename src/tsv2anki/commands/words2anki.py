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
import random
import pathlib

# Logging
import logging
import shutil

# Data / Processing
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# anki
import genanki

# local
from tsv2anki.helpers.theme import (
    color_from_string,
    FRONT_TEMPLATE,
    BACK_TEMPLATE,
    FLIPPED_FRONT_TEMPLATE,
    FLIPPED_BACK_TEMPLATE,
    CARD_CSS,
)
from tsv2anki.helpers.network import download_image


# TODO: fix this
DICT_INFO = {
    "Category": "Kategoria",
    "Word": "Suomea",
    "Translation": "Englantia",
    "Example": "Esimerkki",
    "Alternative": "Puhekieli",
    "Level": "Taso",
}


def add_subparsers(subparsers):
    parser = subparsers.add_parser("words2anki", help="Generate the anki package for word learning")

    parser.add_argument(
        "-r", "--raw", action="store_true", help="Keep the raw generation instead of just getting the archive"
    )
    parser.add_argument("-t", "--tags", default=None, help="List of tags comma separated")

    # Add arguments
    parser.add_argument(
        "input_tsv",
        help="The input TSV file which should contains the following columns ['Kategoria', 'Suomi', 'Englanti', 'Esimerkki']",
    )
    parser.add_argument(
        "output_dir",
        help="The output directory which contains the necessary files: level_1_word.apkg & level_2_word.apkg (level above 3 are ignored)",
    )

    parser.set_defaults(func=main)


def main(args: argparse.Namespace):
    logger = logging.getLogger(__name__)

    # Load dataframe
    df = pd.read_csv(args.input_tsv, sep="\t")

    # Prepare output directory
    output_dir = pathlib.Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True, parents=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True, parents=True)

    # FIXME: I am a bit confused about this - should it be dynamic or hardcoded?
    # model_id = random.randrange(1 << 30, 1 << 31)
    model_id = 1957644191
    my_model = genanki.Model(
        model_id,
        "finnish_learning",
        fields=[
            {"name": "Front"},
            {"name": "Alternative Form"},
            {"name": "Back"},
            {"name": "Category"},
            {"name": "SubCategory"},
            {"name": "CategoryColor"},
            {"name": "SubCategoryColor"},
            {"name": "Example"},
        ],
        # FIXME: be more generic with templates
        templates=[
            {
                "name": "Finnish to English",
                "qfmt": FRONT_TEMPLATE,
                "afmt": BACK_TEMPLATE,
            },
            {
                "name": "English",
                "qfmt": FLIPPED_FRONT_TEMPLATE,
                "afmt": FLIPPED_BACK_TEMPLATE,
            },
        ],
        css=CARD_CSS,
    )

    decks = []
    images: list[pathlib.Path] = []
    audios: list[pathlib.Path] = []

    with logging_redirect_tqdm():
        level_groups = df.groupby(DICT_INFO["Level"])
        for level, level_content in tqdm(level_groups, desc="Overall", position=1, leave=False):
            deck_id = random.randrange(1 << 30, 1 << 31)
            cur_deck = genanki.Deck(deck_id, f"level_{level}")
            logger.info(f"Level {level} has {level_content.shape[0]} cards")
            for i_row, row in tqdm(
                level_content.iterrows(),
                total=level_content.shape[0],
                desc=str(level),
                position=0,
                leave=False,
            ):
                row = row.fillna("")
                target_word = str(row[DICT_INFO["Word"]]).replace("*", "")
                alternative_word = str(row[DICT_INFO["Alternative"]]).replace("*", "")  # TODO: other category?
                known_translation = str(row[DICT_INFO["Translation"]]).replace("*", "")
                category = str(row[DICT_INFO["Category"]])
                example = str(row[DICT_INFO["Example"]])

                tags = []
                if args.tags is not None:
                    tags = [s.strip().lower() for s in args.tags.split(",")]

                try:
                    if args.with_multimedia:
                        # Retrieve an example image
                        cur_image = download_image(target_word, category, image_dir)
                        images.append(cur_image)

                        # Synthesis an example and use it for the answer
                        cur_audio = generate_audio(known_translation, example.replace("*", ""), audio_dir)
                        audios.append(cur_audio)

                    # Create and add note to the deck
                    my_note = genanki.Note(
                        model=my_model,
                        fields=[
                            target_word,
                            alternative_word,
                            known_translation,
                            "Word",
                            category,
                            # f'<img src="{cur_image.name}" />',
                            # f"[sound:{cur_audio.name}]",
                        ]
                        + [color_from_string("Word".lower().strip(), True)]
                        + [color_from_string(category.lower().strip(), False)]
                        + [example],
                        tags=["words", category.replace(" ", "_").lower()] + tags,
                    )
                    cur_deck.add_note(my_note)
                except Exception as ex:
                    logger.warning(f"Cannot add {target_word}: {ex}")

            decks.append(cur_deck)

    # Generate and save package
    my_package = genanki.Package(decks)
    my_package.media_files = images  # + audios
    my_package.write_to_file(output_dir / "to_import.apkg")

    # if not args.raw:
    #     output = output_dir.with_suffix(".apkg")
    #     (output_dir / "to_import.apkg").rename(output)
    #     shutil.rmtree(output_dir)
