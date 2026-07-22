import hashlib
import requests
import logging

from tsv2anki.helpers.theme import (
    FRONT_TEMPLATE,
    BACK_TEMPLATE,
    # FLIPPED_FRONT_TEMPLATE,
    # FLIPPED_BACK_TEMPLATE,
    CARD_CSS,
)

ANKI_URL = "http://localhost:8765"

CATEGORIES = ["Word", "Grammar", "Cultural", "Phrase"]
DECK_PATH_SEP = "::"

class AnkiConnector:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def invoke(self, action, **params):
        r = requests.post(ANKI_URL, json={"action": action, "version": 6, "params": params}).json()
        if r.get("error"):
            raise Exception(r["error"])
        return r["result"]

    def hash_id(self, *parts):
        h = hashlib.sha1()
        for p in parts:
            h.update((p or "").encode("utf-8"))
        return h.hexdigest()[:12]

    def ensure_basic_model(self) -> str:
        name = "FinnishWord"

        try:
            self.invoke(
                "createModel",
                modelName=name,
                inOrderFields=[
                    "ID",
                    "Front",
                    "Back",
                    "Category",
                    "SubCategory",
                    "CategoryColor",
                    "SubCategoryColor",
                    "Example",
                    "TranslatedExample"
                ],
                css=CARD_CSS,
                cardTemplates=[
                    {
                        "Name": "Card 1",
                        "Front": FRONT_TEMPLATE,
                        "Back": BACK_TEMPLATE,
                    }
                ],
                isCloze=False
            )

        except:
            self.invoke(
                "updateModelTemplates",
                model={
                    "name": name,
                    "templates": {
                        "Card 1": {
                            "Front": FRONT_TEMPLATE,
                            "Back": BACK_TEMPLATE,
                        }
                    },
                },
            )
            self.invoke("updateModelStyling", model={"name": name, "css": CARD_CSS})

        return name

    def ensure_deck(self, deck_path: str|list[str]):
        if isinstance(deck_path, list):
            deck_path = DECK_PATH_SEP.join(deck)

        available_decks = self.invoke("deckNamesAndIds")
        if deck_path not in available_decks:
            self.invoke("createDeck", deck=deck_path)


    def find_note(self, note_id):
        return self.invoke("findNotes", query=f'"{note_id}"')

    def upsert_basic(self, category: str, subdeck_path: str|list[str], model: str, fields: dict[str, str], dry_run: bool):

        logger = logging.getLogger("common")

        # Get ID from the information
        nid = self.hash_id(fields["Front"], fields["Back"])

        found = self.find_note(nid)
        fields["ID"] = nid


        if found:
            self.logger.info(f"update: {nid}")
            if not dry_run:
                self.invoke("updateNoteFields", note={"id": found[0], "fields": fields})
        else:


            deck = [category]
            if isinstance(subdeck_path, str):
                deck.append(subdeck_path)
            else:
                deck += subdeck_path
            deck = DECK_PATH_SEP.join(deck)
            logger.info(f"add: {nid} into {deck}")
            if not dry_run:
                self.ensure_deck(deck)
                self.invoke(
                    "addNote", note={"deckName": deck, "modelName": model, "fields": fields, "tags": fields["tags"]}
                )
