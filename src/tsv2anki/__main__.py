#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR

    Sébastien Le Maguer <sebastien.lemaguer@helsinki.fi>

DESCRIPTION

LICENSE
    This script is in the public domain, free from copyrights or restrictions.
    Created: 29 October 2025
"""

# Core Python
import argparse

# Messaging/logging
import logging
from logging.config import dictConfig
try:
    import pythonjsonlogger
    JSON_LOGGER = True
except Exception:
    JSON_LOGGER = False

# Commands
from tsv2anki.commands import words2anki

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

    parser.add_argument(
        "--filter-known",
        "-k",
        type=str,
        default=None,
        help="Filter out the known/unkown considering the score (e.g., <2 or >3 or =3)"
    )

    parser.add_argument(
        "--filter-needed",
        "-n",
        type=str,
        default=None,
        help="Filter out the needed/unneeded considering the score (e.g., <2 or >3 or =3)"
    )
    parser.add_argument(
        "--with-multimedia",
        "-m",
        action="store_true",
        help="Activate multimedia data"
    )

    # Add subparsers
    subparsers = parser.add_subparsers(title="Subcommands", dest="subcommand", help="Available subcommands")
    words2anki.add_subparsers(subparsers)

    # Return parser
    return parser




###############################################################################
# Entry point
###############################################################################
def main():
    # Initialization of the argument parser and the logger
    arg_parser = define_argument_parser()
    args = arg_parser.parse_args()
    configure_logger(args)

    if args.subcommand:
        args.func(args)
    else:
        arg_parser.print_help()


###############################################################################
# Wrapping for directly calling the scripts
###############################################################################
if __name__ == "__main__":
    main()
