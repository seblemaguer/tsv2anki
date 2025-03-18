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
import random
import pathlib
import requests
import argparse
import concurrent.futures
import functools
import time

# Messaging/logging
import logging
from logging.config import dictConfig

# Image search
from PIL import Image
from duckduckgo_search import DDGS
from fastcore.all import *

# Data / Processing
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# anki
import genanki


###############################################################################
# global constants
###############################################################################
LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]


###############################################################################
# Functions
###############################################################################
def configure_logger(args) -> logging.Logger:
    """Setup the global logging configurations and instanciate a specific logger for the current script

    Parameters
    ----------
    args : dict
        The arguments given to the script

    Returns
    --------
    the logger: logger.Logger
    """
    # create logger and formatter
    logger = logging.getLogger()

    # Verbose level => logging level
    log_level = args.verbosity
    if args.verbosity >= len(LEVEL):
        log_level = len(LEVEL) - 1
        # logging.warning("verbosity level is too high, I'm gonna assume you're taking the highest (%d)" % log_level)

    # Define the default logger configuration
    logging_config = dict(
        version=1,
        disable_existing_logger=True,
        formatters={
            "f": {
                "format": "[%(asctime)s] [%(levelname)s] — [%(name)s — %(funcName)s:%(lineno)d] %(message)s",
                "datefmt": "%d/%b/%Y: %H:%M:%S ",
            }
        },
        handlers={
            "h": {
                "class": "logging.StreamHandler",
                "formatter": "f",
                "level": LEVEL[log_level],
            }
        },
        root={"handlers": ["h"], "level": LEVEL[log_level]},
    )

    # Add file handler if file logging required
    if args.log_file is not None:
        logging_config["handlers"]["f"] = {
            "class": "logging.FileHandler",
            "formatter": "f",
            "level": LEVEL[log_level],
            "filename": args.log_file,
        }
        logging_config["root"]["handlers"] = ["h", "f"]

    # Setup logging configuration
    dictConfig(logging_config)

    # Retrieve and return the logger dedicated to the script
    logger = logging.getLogger(__name__)
    return logger


def define_argument_parser() -> argparse.ArgumentParser:
    """Defines the argument parser

    Returns
    --------
    The argument parser: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="")

    # Add options
    parser.add_argument("-l", "--log_file", default=None, help="Logger file")
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="increase output verbosity",
    )

    # Add arguments
    parser.add_argument(
        "input_tsv",
        help="The input TSV file which should contains the following columns ['Kategoria', 'Suomi', 'Englanti', 'Esimerkki']",
    )
    parser.add_argument(
        "output_dir",
        help="The output directory which contains the necessary files. The file to import in anki is named to_import.apkg",
    )
    # TODO

    # Return parser
    return parser


# Define the timeout decorator
def timeout(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timeout = 3

        # Use ThreadPoolExecutor to run the function with a timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Function '{func.__name__}' timed out after {timeout} seconds"
                )

    return wrapper


@timeout
def search_images(term, max_images=1):
    with DDGS() as ddgs:
        search_results = ddgs.images(keywords=term, region="fi-fi")
        image_data = list(search_results)
        image_urls = [item.get("image") for item in image_data[:max_images]]
        return L(image_urls)


def download_image(keyword: str, cat: str, output_dir: pathlib.Path) -> pathlib.Path:
    url = search_images(f"+{keyword} {cat}", 1)[0]

    image_filename = output_dir / f"{keyword.replace('/', ',')}.jpg"
    if image_filename.exists():
        return image_filename

    try:
        img_data = requests.get(url).content
        img = Image.open(io.BytesIO(img_data))
        img.verify()
        with open(image_filename, "wb") as f:
            f.write(img_data)
    except Exception as e:
        raise Exception(f"Error downloading {image_filename}: {e}")

    return image_filename


def generate_audio(
    keyword: str, sentence: str, output_dir: pathlib.Path
) -> pathlib.Path:
    keyword = "test"
    output_file = output_dir / f"{keyword}.mp3"
    return output_file


###############################################################################
#  Envelopping
###############################################################################
if __name__ == "__main__":
    # Initialization
    arg_parser = define_argument_parser()
    args = arg_parser.parse_args()
    logger = configure_logger(args)

    # TODO: your code comes here
    df = pd.read_csv(args.input_tsv, sep="\t")

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
            {"name": "Word"},
            {"name": "Image"},
            {"name": "Audio"},
            {"name": "Example"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Image}}",
                "afmt": '{{FrontSide}}<hr id="answer"><center><b>{{Word}}</b><br />{{Example}}<br />{{Audio}}</center>',
            },
        ],
    )

    decks = []
    images: list[pathlib.Path] = []
    audios: list[pathlib.Path] = []

    with logging_redirect_tqdm():
        group_cat = df.groupby("Kategoria")
        for cat, cat_values in tqdm(group_cat, desc="Overall", position=1, leave=False):
            deck_id = random.randrange(1 << 30, 1 << 31)
            cur_deck = genanki.Deck(deck_id, cat)
            for i_row, row in tqdm(
                cat_values.iterrows(),
                total=cat_values.shape[0],
                desc=cat,
                position=0,
                leave=False,
            ):
                row = row.fillna("")
                fi_word = row["Suomi"].replace("*", "")
                en_word = row["Englanti"].replace("*", "")
                example = row["Esimerkki"]
                logger.warning(fi_word)

                if not example:
                    logger.warning(f'no example for word "{fi_word}"')
                    continue

                try:
                    # Retrieve an example image
                    cur_image = download_image(fi_word, cat, image_dir)
                    images.append(cur_image)

                    # Synthesis an example and use it for the answer
                    cur_audio = generate_audio(
                        en_word, example.replace("*", ""), audio_dir
                    )
                    audios.append(cur_audio)

                    # Create and add note to the deck
                    my_note = genanki.Note(
                        model=my_model,
                        fields=[
                            fi_word,
                            f'<img src="{cur_image.name}" />',
                            f"[sound:{cur_audio.name}]",
                            example,
                        ],
                    )
                    cur_deck.add_note(my_note)
                except Exception as ex:
                    logger.warning(f"Cannot add {fi_word}: {ex}")

            decks.append(cur_deck)

    # Generate and save package
    my_package = genanki.Package(decks)
    my_package.media_files = images # + audios
    my_package.write_to_file(output_dir / "to_import.apkg")
