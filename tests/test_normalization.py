"""
Tests for name normalisation, version 2 (gender_predict.normalization).

The cases are the battery of the preprocessing audit of November 2026: each raw name with the given-name and
family-name strings the model must receive, or the status returned when the name cannot be represented.

Run with:  pytest tests/  (or  python tests/test_normalization.py)
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gender_predict.normalization import fit_field, normalize_name  # noqa: E402

OK = [
    # letters with no Unicode decomposition
    ("Søren Jørgensen", "soren", "jorgensen"), ("Łukasz Wałęsa", "lukasz", "walesa"), ("Đặng Thị Hà", "dang", "thi ha"),
    ("Işıl Yıldız", "isil", "yildiz"), ("Richard Strauß", "richard", "strauss"), ("Ægir Þórsson", "aegir", "thorsson"),
    ("Björk Guðmundsdóttir", "bjork", "gudmundsdottir"), ("Khwankǣo Watcharōthai", "khwankaeo", "watcharothai"), ("Ọlá Adébáyọ̀", "ola", "adebayo"),
    # common accents, hyphens, spaces, case
    ("José Müller", "jose", "muller"), ("François Nuñez", "francois", "nunez"), ("Jean-Pierre Dupont", "jean-pierre", "dupont"),
    ("  Maria   Rossi ", "maria", "rossi"), ("MARIO ROSSI", "mario", "rossi"), ("María José de la Cruz", "maria", "jose de la cruz"),
    ("Nguyễn Thị Hà", "nguyen", "thi ha"), ("Madonna", "madonna", ""),
    # apostrophes: kept inside a word, in every typographic variant; dropped at the end of a word (accent)
    ("D'Angelo Russell", "d'angelo", "russell"), ("Giovanni D’Angelo", "giovanni", "d'angelo"), ("Sean O`Brien", "sean", "o'brien"),
    ("NICOLO' ROSSI", "nicolo", "rossi"), ("Anne-Marie O’Neill-Smith", "anne-marie", "o'neill-smith"),
    # titles, suffixes, comma order, initials, nicknames, stray characters
    ("Dr. Anna Bianchi", "anna", "bianchi"), ("Prof.ssa Maria Verdi", "maria", "verdi"), ("Sig.ra Lucia Neri", "lucia", "neri"),
    ("Mr John Smith", "john", "smith"), ("Ing Yoe Tan", "ing", "yoe tan"), ("Rossi, Mario", "mario", "rossi"),
    ("Martin Luther King, Jr.", "martin", "luther king"), ("John Smith Jr", "john", "smith"),
    ("J. Robert Oppenheimer", "robert", "oppenheimer"), ("John F. Kennedy", "john", "f kennedy"),
    ("Riccardo 'Solar Red' Ferguson", "riccardo", "ferguson"), ('Dwayne "The Rock" Johnson', "dwayne", "johnson"),
    ("Mario Rossi (attore)", "mario", "rossi"), ("Mario Rossi 2", "mario", "rossi"),
    # supported non-Latin scripts
    ("Анна Каренина", "anna", "karenina"), ("김민석", "gimminseog", ""),
    # Han characters: family name first, pinyin "given family" for the network
    ("王芳", "fang", "wang"), ("欧阳娜娜", "nana", "ouyang"), ("張學友", "xueyou", "zhang"), ("王 小明", "xiaoming", "wang"),
    # Greek (ELOT 743 style)
    ("Γιώργος Παπαδόπουλος", "giorgos", "papadopoulos"), ("Γεώργιος Athanasiadis", "georgios", "athanasiadis"),
    ("Ευάγγελος Βενιζέλος", "evangelos", "venizelos"), ("Μπάμπης Αγγελόπουλος", "bampis", "angelopoulos"),
    ("Ελευθερία Αρβανιτάκη", "eleftheria", "arvanitaki"), ("ΜΑΡΙΑ ΚΑΛΛΑΣ", "maria", "kallas"), ("Χρήστος Ψωμιάδης", "christos", "psomiadis"),
]
NOT_OK = [
    ("", "empty"), ("   ", "empty"), (None, "empty"), ("12345", "empty"), ("J. Smith", "initial_only"),
    ("דוד Levi", "unsupported_script"), ("محمد علي", "unsupported_script"), ("דוד לוי", "unsupported_script"),
    ("Արամ Խաչատրյան", "unsupported_script"), ("สมชาย ใจดี", "unsupported_script"), ("राज कुमार", "unsupported_script"),
]


def test_expected_strings():
    bad = [(raw, normalize_name(raw)[:4]) for raw, first, rest in OK if normalize_name(raw)[1:4] != (first, rest, "ok")]
    assert not bad, bad


def test_undetermined():
    bad = [(raw, normalize_name(raw)) for raw, status in NOT_OK if normalize_name(raw).status != status or normalize_name(raw).text]
    assert not bad, bad


def test_partial_when_given_name_is_readable():
    n = normalize_name("Marina לוי")
    assert (n.first, n.rest, n.status) == ("marina", "", "partial")
    n = normalize_name("Anna Maria محمد")
    assert (n.first, n.rest, n.status) == ("anna", "maria", "partial")
    n = normalize_name("Marina Συμεού")   # Greek is romanised, so nothing is lost
    assert (n.first, n.rest, n.status) == ("marina", "symeou", "ok")


def test_initials_drop_mode():
    assert normalize_name("J. Smith").status == "initial_only"
    n = normalize_name("S. Radhakrishnan", "drop")
    assert (n.first, n.rest, n.status) == ("radhakrishnan", "", "initials_dropped")
    n = normalize_name("T. N. Seshan", "drop")
    assert (n.first, n.rest, n.status) == ("seshan", "", "initials_dropped")
    assert normalize_name("J.", "drop").status == "initial_only" and normalize_name("G.B.", "drop").status == "initial_only"
    assert normalize_name("Maria Rossi", "drop") == normalize_name("Maria Rossi")


def test_han_characters():
    from gender_predict.han import han_logodds, split_han_name

    assert normalize_name("王芳").han == "芳" and normalize_name("王芳").script == "ZH"
    assert normalize_name("Fang Wang").han == ""
    assert split_han_name("欧阳娜娜") == ("欧阳", "娜娜") and split_han_name("王芳 (演员)") == ("王", "芳") and split_han_name("Wang 芳") is None
    assert han_logodds("芳") > 1 and han_logodds("强") < -1 and han_logodds("偉") == han_logodds("伟") < -1   # traditional = simplified


def test_idempotent_and_alphabet():
    allowed = set("abcdefghijklmnopqrstuvwxyz' -")
    for raw, _, _ in OK:
        n = normalize_name(raw)
        assert set(n.text) <= allowed, (raw, n.text)
        assert normalize_name(n.text).text == n.text, (raw, n.text)


def test_family_name_keeps_its_end():
    assert fit_field("giuseppe alfonso giungi", 20, "tail") == "alfonso giungi"
    assert fit_field("nunes de magalhaes magnavita de oliveira", 20, "tail") == "de oliveira"
    assert fit_field("rossi", 20, "tail") == "rossi"
    assert len(fit_field("wolfeschlegelsteinhausenbergerdorff", 20, "tail")) == 20


def test_preprocessor_v2_matches_normalize_name():
    from gender_predict.data.improved_preprocessing import ImprovedNamePreprocessor

    p = ImprovedNamePreprocessor(normalization=2)
    d = p.preprocess_name("Alessandro Giuseppe Alfonso Maria Søren Giungi")
    assert (d["first_raw"], d["last_raw"]) == ("alessandro", "maria soren giungi")
    assert p.split_full_name("Łukasz Wałęsa") == ("lukasz", "walesa")
    assert 1 not in d["first_name"] and 1 not in d["last_name"]  # no <UNK>
    old = ImprovedNamePreprocessor()  # default: historical behaviour untouched
    assert old.preprocess_name("Søren Jørgensen")["first_raw"] == "søren"


def test_augmenter_v2():
    from gender_predict.data.augmentation import NameAugmenter

    random.seed(0)
    a = NameAugmenter(0.25, version=2)
    out = [a.augment("maria rossi") for _ in range(2000)]
    assert sum(o != "maria rossi" for o in out) > 1500      # always applied: the caller's draw is the only one
    assert not any(c.isdigit() or c.isupper() for o in out for c in o)




def test_single_string_preprocessor_and_model():
    import torch
    from gender_predict.data.improved_preprocessing import ImprovedNamePreprocessor
    from gender_predict.models import create_model

    p = ImprovedNamePreprocessor(normalization=2, single_string=True, max_name_length=48)
    d = p.preprocess_name("Maria Do Carmo Seffair Lins De Albuquerque Pereira Da Silva Santos")
    text = "".join(p.idx_to_char[i] for i in d["first_name"] if i)
    assert len(d["first_name"]) == 48 and text.startswith("maria ") and text.endswith("da silva santos"), text
    assert "".join(p.idx_to_char[i] for i in p.preprocess_name("Nguyễn Thị Hà")["first_name"] if i) == "nguyen thi ha"
    for aux in (True, False):
        m = create_model("v3s", vocab_size=p.vocab_size, suffix_vocab_size=50, embedding_dim=16, hidden_size=16, n_layers=1,
                         num_attention_heads=8, aux_features=aux).eval()
        x = torch.tensor([d["first_name"], d["first_name"]])
        out = m(x, torch.zeros(2, 20, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 4))
        assert out.shape == (2,)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
