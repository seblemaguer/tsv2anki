# TSV2Anki

Helper to convert a TSV file to a anki deck.

For now it is restricted to Finnish (because I learn Finnish), but it is planned to be expanded.


## How to install

Clone the repository, and simply run

```sh
pip install .
```

## How to use

You need to have a TSV file formatted like the following example:

```tsv
Kategoria	Suomi	Englanti	Esimerkki
luonto	luonto	nature	Luonnon kauneus on henkeäsalpaavaa.
luonto	Suo	swamp	Me patikoimme **suolla** koko iltapäivän.
luonto	Tunturi	fell	Kiipesimme Lapin korkeimmalle **tunturille**.
```

Then you can run the following command

```sh
tsv2anki <input.tsv> <output_dir>
```

The deck package will be in `output_dir/to_import.apkg`.
