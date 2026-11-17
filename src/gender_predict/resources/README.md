# Data files shipped with the package

## han_zh.json
Table for Chinese names written in characters (see `gender_predict/han.py`): for each character, the log-odds
female/male of the given names that contain it, and the list of frequent compound family names.

- Source: Comprehensive Chinese Name Corpus (CCNC), https://github.com/jaaack-wang/ccnc, licence GPL-3.0
  (3,563,784 names with gender). The table is a derived work and is distributed under the same licence as this package.
- Traditional characters are copies of the value of their simplified form (mapping of the `zhconv` package, used only
  when the table is built).
- Built with `scripts/build_han_table.py` of the project's laboratory (not part of this repository).
