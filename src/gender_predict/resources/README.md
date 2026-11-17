# Data files shipped with the package

## han_zh.json
Table for Chinese names written in characters (see `gender_predict/han.py`): for each character, the log-odds
female/male of the given names that contain it, and the list of frequent compound family names.

- Source: Comprehensive Chinese Name Corpus (CCNC), https://github.com/jaaack-wang/ccnc, licence GPL-3.0
  (3,563,784 names with gender). The table is a derived work and is distributed under the same licence as this package.
- Traditional characters are copies of the value of their simplified form (mapping of the `zhconv` package, used only
  when the table is built).
- Built with `scripts/build_han_table.py` of the project's laboratory (not part of this repository).

## names_by_country.json.gz
Table of first names by country (see `gender_predict/country.py`): for each ISO 3166-1 alpha-2 country, the normalised
first names borne by at least 3 persons, with the number of persons and of women.

- Sources, open data only: Wikidata (CC0); registers of local administrators of the Italian Ministry of the Interior;
  Répertoire national des élus, France (Licence Ouverte 2.0); candidates of the Brazilian Tribunal Superior Eleitoral
  (CC BY); Olympic athletes (Oonk 2026, CC0). No IMDb row enters this table.
- The persons of the project's evaluation sets, and their homonyms, are not counted.
- Built with `scripts/build_name_tables.py` of the project's laboratory from the training file of November 2026.

## han_ja.json.gz
Tables for Japanese names in original script (see `gender_predict/han.py`): log-odds female/male of the last and of the
last-but-one character of the full name; and the evidence that a name written only in Han characters is Japanese
rather than Chinese (length, first character, first two characters, last character).

- Sources: Japanese labels of 38,458 Wikidata persons (CC0), harvested on 18 November 2026, none of them in the
  project's evaluation sets; for the Chinese side of the language table, CCNC (GPL-3.0).
- Built with `scripts/build_kanji_table.py` of the project's laboratory.

## hangul_ko.json.gz
Table for Korean names in hangul (see `gender_predict/han.py`): log-odds female/male of the last and of the last-but-one
syllable of the given name.

- Source: Korean labels of 18,233 Wikidata persons (CC0), harvested on 18 November 2026, none of them in the project's
  evaluation sets.
- Built with `scripts/build_hangul_table.py` of the project's laboratory.
