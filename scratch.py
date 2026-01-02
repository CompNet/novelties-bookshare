# -*- eval: (code-cells-mode); -*-
# %%
from novelties_bookshare.experiments.data import (
    normalize_,
    EDITION_SETS,
    iter_book_chapters,
)
from novelties_bookshare.experiments.metrics import errors, errors_nb
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import decrypt_tokens
from more_itertools import flatten

F1818 = list(iter_book_chapters(EDITION_SETS["Frankenstein"]["F-1818"]))
normalize_(F1818)
F1823 = list(iter_book_chapters(EDITION_SETS["Frankenstein"]["F-1823"]))
normalize_(F1823)
F1831 = list(iter_book_chapters(EDITION_SETS["Frankenstein"]["F-1831"]))
normalize_(F1831)

eF1818 = [encrypt_tokens(chapter) for chapter in F1818]
eF1823 = [encrypt_tokens(chapter) for chapter in F1823]
eF1831 = [encrypt_tokens(chapter) for chapter in F1831]

dF1818 = decrypt_tokens(eF1818, F1823, hash_len=64)
print(errors_nb(list(flatten(F1818)), dF1818))
F1818_error_dict = errors(list(flatten(F1818)), dF1818)

dF1831 = decrypt_tokens(list(flatten(eF1831)), list(flatten(F1818)), hash_len=64)
print(errors_nb(list(flatten(F1831)), dF1831))
F1931_error_dict = errors(list(flatten(F1831)), dF1831)

# %%
from novelties_bookshare.decrypt import make_plugin_propagate

dF1831 = decrypt_tokens(
    list(flatten(eF1831)),
    list(flatten(F1818)),
    hash_len=64,
    decryption_plugins=[make_plugin_propagate()],
)
print(errors_nb(list(flatten(F1831)), dF1831))


# %%
from tests.test_encrypt_decrypt import test_substitution

ref_tokens = "A B C D E E".split()
user_tokens = "A B C D E X".split()
tags = "B-PER O O O B-PER I-PER".split()
pred_tokens = decrypt_tokens(encrypt_tokens(ref_tokens), tags, user_tokens)
assert pred_tokens == "A B C D E [UNK]".split()

# %%
from typing import *
import pathlib as pl
from novelties_bookshare.conll import load_conll2002_bio
from tqdm import tqdm


NOVELTIES_PATH = pl.Path("~/Dev/Novelties").expanduser()
corpus = [
    NOVELTIES_PATH / "corpus" / "1984",
    NOVELTIES_PATH / "corpus" / "Bel_Ami",
    NOVELTIES_PATH / "corpus" / "Brave_New_World",
    NOVELTIES_PATH / "corpus" / "Eugenie_Grandet/en",
    NOVELTIES_PATH / "corpus" / "Germinal/en",
    NOVELTIES_PATH / "corpus" / "Madame_Bovary/en",
    NOVELTIES_PATH / "corpus" / "Moby_Dick",
    NOVELTIES_PATH / "corpus" / "The_Black_Company",
    NOVELTIES_PATH / "corpus" / "The_Blade_Itself",
    NOVELTIES_PATH / "corpus" / "The_Colour_Of_Magic",
    NOVELTIES_PATH / "corpus" / "The_Hunchback_of_Notre-Dame/en",
    NOVELTIES_PATH / "corpus" / "The_Light_Fantastic",
    NOVELTIES_PATH / "corpus" / "The_Red_And_The_Black",
    NOVELTIES_PATH / "corpus" / "The_Three_Musketeers/en",
]


def load_book(path: pl.Path) -> tuple[list[str], list[str]]:
    tokens = []
    tags = []
    for path in tqdm(sorted(path.glob("*.conll")), ascii=True):
        chapter_tokens, chapter_tags = load_conll2002_bio(str(path))
        tokens += chapter_tokens
        tags += chapter_tags
    return tokens, tags


# %%
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import (
    decrypt_tokens,
    make_plugin_mlm,
    make_plugin_split,
    make_plugin_propagate,
    make_plugin_case,
)
from novelties_bookshare.experiments.data import load_book
from novelties_bookshare.experiments.metrics import errors_nb

gold_tokens, gold_tags = load_book(
    "./data/editions_diff/Moby_Dick/Novelties", chapter_limit=None
)
user_tokens, _ = load_book("./data/editions_diff/Moby_Dick/PG15", chapter_limit=None)


def normalize_(tokens: list[str], replacements: list[tuple[list[str], str]]):
    for i, token in enumerate(tokens):
        for repl_source, repl_target in replacements:
            if token in repl_source:
                tokens[i] = repl_target


normalize_(gold_tokens, [(["``", "''", "“", "”"], '"')])
normalize_(gold_tokens, [(["‘", "’"], "'")])
normalize_(gold_tokens, [(["…"], "...")])
normalize_(gold_tokens, [(["—"], "-")])
normalize_(user_tokens, [(["``", "''", "“", "”"], '"')])
normalize_(user_tokens, [(["‘", "’"], "'")])
normalize_(user_tokens, [(["…"], "...")])
normalize_(user_tokens, [(["—"], "-")])

encrypted = encrypt_tokens(gold_tokens)
print("decrypting with split...")
decrypted_with_split = decrypt_tokens(
    encrypted,
    gold_tags,
    user_tokens,
    decryption_plugins=[
        make_plugin_split(max_token_len=24, max_splits_nb=4),
    ],
)
print("decrypting with pipe...")
decrypted_with_pipe = decrypt_tokens(
    encrypted,
    gold_tags,
    user_tokens,
    decryption_plugins=[
        make_plugin_case(),
        make_plugin_split(max_token_len=24, max_splits_nb=4),
        make_plugin_mlm("answerdotai/ModernBERT-base", window=16, device="cuda"),
        make_plugin_propagate(),
    ],
)

print(errors_nb(gold_tokens, decrypted_with_split))
print(errors_nb(gold_tokens, decrypted_with_pipe))

# %% Benchmarking aligning multiple chapters vs a single one
import time
from novelties_bookshare.experiments.data import iter_book_chapters, load_book
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import (
    decrypt_tokens,
    make_plugin_mlm,
    make_plugin_split,
    make_plugin_propagate,
    make_plugin_case,
)

gold_tokens = load_book("./data/editions_diff/Moby_Dick/Novelties")
user_tokens = load_book("./data/editions_diff/Moby_Dick/PG15")
encrypted = encrypt_tokens(gold_tokens)
t0 = time.process_time()
_ = decrypt_tokens(encrypted, user_tokens)
t1 = time.process_time()
print(t1 - t0)

encrypted = [
    encrypt_tokens(tokens)
    for tokens in iter_book_chapters("./data/editions_diff/Moby_Dick/Novelties/")
]
user_tokens = list(iter_book_chapters("./data/editions_diff/Moby_Dick/PG15"))
_ = decrypt_tokens(encrypted, user_tokens=user_tokens)
t2 = time.process_time()
print(t2 - t1)


# %%
from typing import *
import os, glob
from novelties_bookshare.encrypt import encrypt_token, encrypt_tokens
from novelties_bookshare.decrypt import (
    decrypt_tokens,
    plugin_mlm,
    plugin_split,
    make_plugin_mlm,
    make_plugin_propagate,
)
from novelties_bookshare.conll import load_conll2002_bio

tokens = []
tags = []
for path in sorted(
    glob.glob(os.path.expanduser("~/Dev/Novelties/corpus/1984/*.conll"))
):
    chapter_tokens, chapter_tags = load_conll2002_bio(path)
    tokens += chapter_tokens
    tags += chapter_tags

# TODO: dev
tokens = tokens[:1000]
tags = tags[:1000]

encrypted = encrypt_tokens(tokens)
decrypted = decrypt_tokens(encrypted, tags, tokens)
assert tokens == decrypted


# %% Perturbations
import random, copy
import numpy as np
from scrambledtext import ProbabilityDistributions, CorruptionEngine
from more_itertools import windowed

OCR_CORRUPTION_PROBS = ProbabilityDistributions.load_from_json(
    "./corruption_distribs.json"
)


def substitute(tokens: List[str], proportion: float) -> List[str]:
    assert 0 <= proportion <= 1.0

    subst_nb = int(proportion * len(tokens))
    indices = np.random.choice(list(range(len(tokens))), subst_nb, replace=False)

    noisy_tokens = copy.deepcopy(tokens)
    for i in indices:
        noisy_tokens[i] = "[ADD]"

    return noisy_tokens


def delete(tokens: List[str], proportion: float) -> List[str]:
    assert 0 <= proportion <= 1.0
    deletion_nb = int(proportion * len(tokens))
    indices = set(
        np.random.choice(list(range(len(tokens))), deletion_nb, replace=False)
    )
    return [tok for i, tok in enumerate(tokens) if not i in indices]


def add(tokens: List[str], proportion: float) -> List[str]:
    addition_nb = int(proportion * len(tokens))
    noisy_tokens = copy.deepcopy(tokens)
    for _ in range(addition_nb):
        noisy_tokens.insert(random.randint(0, len(noisy_tokens) - 1), "[ADD]")
    return noisy_tokens


def ocr_scramble(tokens: List[str], proportion: float) -> List[str]:
    if proportion == 0.0:
        return tokens

    engine = CorruptionEngine(
        OCR_CORRUPTION_PROBS.conditional,
        OCR_CORRUPTION_PROBS.substitutions,
        OCR_CORRUPTION_PROBS.insertions,
        target_wer=proportion,
        target_cer=proportion,
    )

    corrupted_text, _, _, _ = engine.corrupt_text(" ".join(tokens))
    corrupted_tokens = corrupted_text.split()
    return corrupted_tokens


def token_split(tokens: List[str], proportion: float) -> List[str]:
    assert 0 <= proportion <= 1.0

    split_nb = int(proportion * len(tokens))
    split_indices = set(
        np.random.choice(list(range(len(tokens))), split_nb, replace=False)
    )

    noisy_tokens = []
    for i, token in enumerate(tokens):
        if i in split_indices:
            if len(token) >= 2:
                split_idx = random.randint(1, len(token))
                noisy_tokens.append(token[:split_idx])
                noisy_tokens.append(token[split_idx:])
        else:
            noisy_tokens.append(token)

    return noisy_tokens


def token_merge(tokens: List[str], proportion: float) -> List[str]:
    assert 0 <= proportion <= 1.0
    merge_nb = int(proportion * len(tokens))
    merge_indices = set(
        np.random.choice(list(range(len(tokens) - 1)), merge_nb, replace=False)
    )

    noisy_tokens = []
    for i, tok in enumerate(tokens):
        if i - 1 in merge_indices:
            noisy_tokens[-1] += tok
        else:
            noisy_tokens.append(tok)

    return noisy_tokens


ex = "Lianna, princess of Fomalhaut".split()
print("example of operations: ")
print(f"initial example: {ex}")
print(f"sub: {substitute(ex, 0.8)}")
print(f"del: {delete(ex, 0.8)}")
print(f"add: {add(ex, 0.8)}")
print(f"ocr_scramble: {ocr_scramble(ex, 0.8)}")
print(f"token_split: {token_split(ex, 0.8)}")
print(f"token_merge: {token_merge(ex, 0.8)}")


# %% Comparison between strategies
from collections import defaultdict
import matplotlib.pyplot as plt
import functools as ft
from more_itertools import flatten
from tqdm import tqdm
from novelties_bookshare.decrypt import (
    decrypt_tokens,
    make_plugin_mlm,
    make_plugin_propagate,
    make_plugin_split,
)

# %%
x_sub, x_del, x_add, x_comb = (
    defaultdict(list),
    defaultdict(list),
    defaultdict(list),
    defaultdict(list),
)
y_sub, y_del, y_add, y_comb = (
    defaultdict(list),
    defaultdict(list),
    defaultdict(list),
    defaultdict(list),
)

decrypt_fns = [
    {"name": "naive", "fn": decrypt_tokens},
    {
        "name": "propagate",
        "fn": ft.partial(decrypt_tokens, decryption_plugins=[make_plugin_propagate()]),
    },
    {
        "name": "splice",
        "fn": ft.partial(
            decrypt_tokens,
            decryption_plugins=[make_plugin_split(max_token_len=24, max_splits_nb=4)],
        ),
    },
    {
        "name": "mlm",
        "fn": ft.partial(
            decrypt_tokens,
            decryption_plugins=[
                make_plugin_mlm("answerdotai/ModernBERT-base", window=16)
            ],
        ),
    },
    {
        "name": "splice->mlm->propagate",
        "fn": ft.partial(
            decrypt_tokens,
            decryption_plugins=[
                make_plugin_split(max_token_len=24, max_splits_nb=4),
                make_plugin_mlm("answerdotai/ModernBERT-base", window=16),
                make_plugin_propagate(),
            ],
        ),
    },
]

degradations = [
    # {
    #     "name": "add",
    #     "fn": [add],
    #     "x": {d["name"]: [] for d in decrypt_fns},
    #     "y": {d["name"]: [] for d in decrypt_fns},
    # },
    # {
    #     "name": "del",
    #     "fn": [delete],
    #     "x": {d["name"]: [] for d in decrypt_fns},
    #     "y": {d["name"]: [] for d in decrypt_fns},
    # },
    # {
    #     "name": "sub",
    #     "fn": [substitute],
    #     "x": {d["name"]: [] for d in decrypt_fns},
    #     "y": {d["name"]: [] for d in decrypt_fns},
    # },
    # {
    #     "name": "ocr",
    #     "fn": [ocr_scramble],
    #     "x": {d["name"]: [] for d in decrypt_fns},
    #     "y": {d["name"]: [] for d in decrypt_fns},
    # },
    {
        "name": "token_split",
        "fn": [token_split],
        "x": {d["name"]: [] for d in decrypt_fns},
        "y": {d["name"]: [] for d in decrypt_fns},
    },
    {
        "name": "token_merge",
        "fn": [token_merge],
        "x": {d["name"]: [] for d in decrypt_fns},
        "y": {d["name"]: [] for d in decrypt_fns},
    },
]


def test_degradation(
    tokens: list[str],
    encrypted: list[str],
    tags: list[str],
    decrypt_fn,
    degradation_fns: list,
) -> float:
    user_tokens = tokens.copy()
    for degt in degradation_fns:
        user_tokens = degt(user_tokens)
    decrypted = decrypt_fn(encrypted, tags, user_tokens)
    return sum(1 if d == t else 0 for d, t in zip(decrypted, tokens)) / len(tokens)


progress = tqdm(np.arange(0.0, 0.1, 0.01))
for p in progress:
    p = float(p)

    for decrypt in decrypt_fns:
        for i, degradation in enumerate(degradations):
            progress.set_description(f"{decrypt['name']} ({i + 1}/{len(degradations)})")
            degradation["x"][decrypt["name"]].append(p)
            degradation["y"][decrypt["name"]].append(
                test_degradation(
                    tokens,
                    encrypted,
                    tags,
                    decrypt["fn"],
                    [lambda tokens: d(tokens, p) for d in degradation["fn"]],
                )
            )

# %%
if len(degradations) == 1:
    fig, ax = plt.subplots()
    axs = [ax]
else:
    fig, axs = plt.subplots(1 + len(degradations) // 3, 2)
    try:
        axs = list(flatten(axs))
    except TypeError:
        pass

for ax, degradation in zip(axs, degradations):
    for decrypt in decrypt_fns:
        ax.plot(
            degradation["x"][decrypt["name"]],
            degradation["y"][decrypt["name"]],
            label=decrypt["name"],
        )
        ax.grid()
        ax.set_xlabel(degradation["name"])
        ax.set_ylabel("Percentage of recovered tokens")
        ax.set_ylim(
            min([y for d in degradations for values in d["y"].values() for y in values])
            - 0.05,
            1.05,
        )
        ax.legend()

fig.suptitle("Impact of errors")
plt.show()

# %%
from tqdm import tqdm
from novelties_bookshare.decrypt import decrypt_tokens

x = []
y = []
y_naive = []
for i in tqdm(range(1, 65)):
    x.append(i)
    encrypted = encrypt_tokens(tokens, hash_len=i)
    user_tokens = add(substitute(delete(tokens, 0.1), 0.1), 0.1)
    decrypted = decrypt_tokens(encrypted, tags, user_tokens, hash_len=i)
    percent = sum(1 if d == t else 0 for d, t in zip(decrypted, tokens)) / len(tokens)
    y.append(percent)
    decrypted_naive = decrypt_tokens_naive(encrypted, tags, user_tokens, hash_len=i)
    percent = sum(1 if d == t else 0 for d, t in zip(decrypted_naive, tokens)) / len(
        tokens
    )
    y_naive.append(percent)

plt.plot(x, y, label="advanced")
plt.plot(x, y_naive, label="naive")
plt.xlabel("Hash length")
plt.ylabel("Percentage of recovered tokens")
plt.suptitle("Influence of hash length (delete=0.1, substitute=0.1, add=0.1)")
plt.legend()
plt.show()


# %% splice test
import difflib
from novelties_bookshare.decrypt import (
    encrypt_tokens,
    decrypt_tokens,
    plugin_mlm,
    plugin_split,
    make_plugin_mlm,
)

tokens = "Lianna princesse de Fomalhaut".split()
tags = ["O"] * len(tokens)
user_tokens = "Lianna princ esse de Fomalhaut".split()
encrypted_tokens = encrypt_tokens(tokens)

encrypted_user_tokens = encrypt_tokens(user_tokens)
matcher = difflib.SequenceMatcher(None, encrypted_tokens, encrypted_user_tokens)
# SequenceMatcher(None, A, B) gives each opcode with the form
# (operation, i1, i2, j1, j2) with i1, i2 in A and j1, j2 in B
for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    print(f"{tag} {tokens[i1:i2]} {user_tokens[j1:j2]}")

print(decrypt_tokens(encrypted_tokens, tags, user_tokens))
print(
    decrypt_tokens(
        encrypted_tokens, tags, user_tokens, decryption_plugins=[plugin_split]
    )
)

# %% BERT test
tokens = "I am your father Luke ! , said Vader".split()
tags = ["O"] * len(tokens)
user_tokens = "I am your Luke ! , said Vader".split()
encrypted_tokens = encrypt_tokens(tokens)

encrypted_user_tokens = encrypt_tokens(user_tokens)
matcher = difflib.SequenceMatcher(None, encrypted_tokens, encrypted_user_tokens)
# SequenceMatcher(None, A, B) gives each opcode with the form
# (operation, i1, i2, j1, j2) with i1, i2 in A and j1, j2 in B
for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    print(f"{tag} {tokens[i1:i2]} {user_tokens[j1:j2]}")

print(decrypt_tokens(encrypted_tokens, tags, user_tokens))
print(
    decrypt_tokens(
        encrypted_tokens,
        tags,
        user_tokens,
        decryption_plugins=[make_plugin_mlm("answerdotai/ModernBERT-base", window=16)],
    )
)


# %% Comparison between strategies
import pathlib as pl
import functools as ft
from collections import defaultdict
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import (
    decrypt_tokens,
    make_plugin_mlm,
    make_plugin_propagate,
    make_plugin_split,
)
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def load_book(path: pl.Path) -> tuple[list[str], list[str]]:
    tokens = []
    tags = []
    for path in tqdm(sorted(path.glob("*.conll")), ascii=True):
        chapter_tokens, chapter_tags = load_conll2002_bio(str(path))
        tokens += chapter_tokens
        tags += chapter_tags
    return tokens, tags


novelties_tokens, novelties_tags = load_book(
    pl.Path("~/Dev/Novelties/corpus/Brave_New_World").expanduser()
)

wild_editions = {
    "HC98": load_book(pl.Path("./data/editions_diff/HC98"))[0],
    "HC04": load_book(pl.Path("./data/editions_diff/HC04"))[0],
    "HC06": load_book(pl.Path("./data/editions_diff/HC06"))[0],
    "RB06": load_book(pl.Path("./data/editions_diff/RB06"))[0],
}


def normalize_(tokens: list[str], replacements: list[tuple[list[str], str]]):
    for i, token in enumerate(tokens):
        for repl_source, repl_target in replacements:
            if token in repl_source:
                tokens[i] = repl_target


# OPTIONAL: preprocessing
normalize_(novelties_tokens, [(["``", "''"], '"')])
normalize_(novelties_tokens, [(["…"], "...")])
for tokens in wild_editions.values():
    normalize_(tokens, [(["``", "''", "“", "”"], '"')])
    normalize_(tokens, [(["‘", "’"], "'")])
    normalize_(tokens, [(["…"], "...")])
    normalize_(tokens, [(["—"], "-")])


novelties_encrypted_tokens = encrypt_tokens(novelties_tokens)

strategies = {
    "naive": None,
    "propagate": [make_plugin_propagate()],
    "splice": [make_plugin_split(max_token_len=24, max_splits_nb=4)],
    "bert": [make_plugin_mlm("answerdotai/ModernBERT-base", window=16)],
    "pipe": [
        make_plugin_propagate(),
        make_plugin_mlm("answerdotai/ModernBERT-base", window=16),
        make_plugin_split(max_token_len=24, max_splits_nb=4),
        make_plugin_propagate(),
    ],
}
# { strategy => { edition => errors_nb } }
errors_nb = {
    strat: {ed: 0 for ed in wild_editions.keys()} for strat in strategies.keys()
}
errors = {
    strat: {ed: defaultdict(int) for ed in wild_editions.keys()}
    for strat in strategies.keys()
}

progress = tqdm(total=len(wild_editions) * len(strategies), ascii=True)

for edition, user_tokens in wild_editions.items():
    for strat, strat_plugins in strategies.items():
        progress.set_description(f"{edition}.{strat}")

        decrypted_tokens = decrypt_tokens(
            novelties_encrypted_tokens,
            novelties_tags,
            user_tokens,
            hash_len=None,
            decryption_plugins=strat_plugins,
        )
        local_errors_nb = sum(
            1 if ref != pred else 0
            for ref, pred in zip(novelties_tokens, decrypted_tokens)
        )
        errors_nb[strat][edition] = local_errors_nb

        for ref, pred in zip(novelties_tokens, decrypted_tokens):
            if ref != pred:
                errors[strat][edition][ref] += 1

        progress.update()

# %%
bar_width = 0.15
offset = bar_width * (len(strategies) / 2 - 0.5)
fig, ax = plt.subplots()
x = np.arange(len(wild_editions))
for i, strat in enumerate(strategies):
    strat_x = x + i * bar_width - offset
    y = np.array(list(errors_nb[strat].values()))
    bars = ax.bar(strat_x, y, bar_width, label=strat)
    ax.bar_label(bars)
ax.set_xticks(x)
ax.set_xticklabels(wild_editions.keys())
ax.set_ylabel("Number of errors")
ax.legend()
plt.show()


# %%
import difflib
from novelties_bookshare.experiments.data import load_book
from novelties_bookshare.decrypt import (
    decrypt_tokens,
    make_plugin_cycle,
    make_plugin_case,
    make_plugin_propagate,
    make_plugin_mlm,
    make_plugin_split,
)
from novelties_bookshare.encrypt import encrypt_tokens


ref_tokens, ref_tags = load_book("/home/aethor/Dev/Novelties/corpus/Brave_New_World")
encrypted_tokens = encrypt_tokens(ref_tokens)
user_tokens, _ = load_book("./data/editions_diff/Brave_New_World/HC98")

naive_decrypted = decrypt_tokens(encrypted_tokens, ref_tags, user_tokens)
print(sum(1 if ref != pred else 0 for ref, pred in zip(ref_tokens, naive_decrypted)))

cycle_decrypted = decrypt_tokens(
    encrypted_tokens,
    ref_tags,
    user_tokens,
    decryption_plugins=[
        make_plugin_cycle(
            [
                make_plugin_propagate(),
                make_plugin_case(),
                make_plugin_split(max_token_len=24, max_splits_nb=4),
                make_plugin_mlm("answerdotai/ModernBERT-base", window=16),
            ],
            budget=None,
        )
    ],
)
print(sum(1 if ref != pred else 0 for ref, pred in zip(ref_tokens, cycle_decrypted)))

# %%
# A refined proof-of-concept for a word-level frequency attack.
# This version uses the NLTK Brown Corpus to generate a realistic
# frequency list of English words.

from novelties_bookshare.experiments.data import load_conll2002_bio
from collections import Counter
import nltk
from nltk.corpus import brown

nltk.download("brown")


def generate_frequency_list(corpus):
    """
    Generates a list of the most common words from a given corpus.

    Args:
        corpus: An NLTK corpus object.

    Returns:
        list: A list of the most common words, sorted by frequency.
    """
    # Get all words from the corpus and convert them to lowercase
    all_words = [word.lower() for word in corpus.words()]

    # Count the frequency of each word.
    word_frequencies = Counter(all_words)

    # Exclude common "stopwords" like 'the', 'a', 'is' to focus on more
    # meaningful words for a more advanced analysis, but for this basic
    # attack, we will include them.

    # Get the 100 most common words.
    most_common = [word for word, count in word_frequencies.most_common(100)]
    return most_common
    # return word_frequencies


def frequency_attack(ciphertext, known_words):
    """
    Performs a word-level frequency attack on a list of tokens.

    Args:
        ciphertext (list): A list of encrypted word tokens.
        known_words (list): A list of common words in the plaintext language,
                            sorted by frequency (most common first).

    Returns:
        tuple: A tuple containing the list of decrypted tokens and the
               substitution map.
    """
    # 1. Count the frequency of each token in the ciphertext.
    ciphertext_frequencies = Counter(ciphertext)

    # 2. Get the unique tokens from the ciphertext, sorted by their frequency.
    encrypted_tokens_sorted_by_frequency = [
        token for token, count in ciphertext_frequencies.most_common()
    ]

    # 3. Create a substitution map.
    substitution_map = {}
    for i, encrypted_token in enumerate(encrypted_tokens_sorted_by_frequency):
        if i < len(known_words):
            substitution_map[encrypted_token] = known_words[i]
        else:
            substitution_map[encrypted_token] = f"UNKNOWN_WORD_{i}"

    # 4. Decrypt the original ciphertext list using the substitution map.
    decrypted_tokens = [substitution_map.get(token, token) for token in ciphertext]

    return decrypted_tokens, substitution_map
    # Generate a frequency list from a real corpus.
    # Note: WordNet's own frequency counts are based on a very small, outdated
    # corpus and are often inaccurate, so using the Brown Corpus is a more
    # effective demonstration.


most_common_english_words = generate_frequency_list(brown)

# ref_tokens, ref_tags = load_book("/home/aethor/Dev/Novelties/corpus/Brave_New_World")
# encrypted_tokens = encrypt_tokens(ref_tokens)
# user_tokens, _ = load_book("./data/editions_diff/Brave_New_World/HC98")
ref_tokens, ref_tags = load_conll2002_bio(
    "./data/editions_diff/Brave_New_World/HC98/chapter_1.conll"
)
encrypted_tokens = encrypt_tokens(ref_tokens)
user_tokens, _ = load_conll2002_bio(
    "./data/editions_diff/Brave_New_World/HC98/chapter_1.conll"
)

# Perform the attack
decrypted_result, mapping = frequency_attack(
    encrypted_tokens, most_common_english_words
)

decrypted_tokens = decrypt_tokens(
    encrypted_tokens,
    ref_tags,
    decrypted_result,
    decryption_plugins=[
        make_plugin_cycle(
            [
                make_plugin_propagate(),
                make_plugin_case(),
                make_plugin_split(max_token_len=24, max_splits_nb=4),
                make_plugin_mlm("answerdotai/ModernBERT-base", window=16),
            ],
            budget=None,
        )
    ],
)

# Print the results
print("Corpus Used:", "Brown Corpus (via NLTK)")
print("Original Encrypted Tokens:")
# print(encrypted_tokens)
print("\n---")
print("Frequency-Based Substitution Map:")
# for key, value in mapping.items():
#     print(f"'{key}' -> '{value}'")
print("\n---")
print("Attempted Decrypted Result:")
print()
print(decrypted_result)
errors = sum(
    1 if ref != pred else 0
    for ref, pred in zip(
        encrypted_tokens,
        encrypt_tokens(decrypted_tokens, hash_len=64),
    )
)
print(errors)

# %%
import nltk
from nltk.util import ngrams
import math
from collections import Counter
from nltk.corpus import brown, wordnet

nltk.download("brown")
nltk.download("wordnet")


def freq(tokens: list) -> dict[Any, float]:
    counter = Counter(tokens)
    return {token: count / len(tokens) for token, count in counter.items()}


def entropy(dist: list[float]) -> float:
    return -sum(prob * math.log2(prob) for prob in dist)


def guesses_nb(dist: list[float]) -> float:
    return sum(i * prob for i, prob in enumerate(sorted(dist, key=lambda v: -v)))


def token_entropy(freq_corpus: list[str], ngram_len: int) -> float:
    return entropy(list(freq(list(ngrams(freq_corpus, ngram_len))).values()))


# %%
guesses_nb(freq(list(wordnet.words())).values())


# %%
import matplotlib.pyplot as plt

# 1984
tokens, _ = load_book(corpus[0])

x = list(range(1, 10))
y = []
for ngram_len in x:
    token_ngrams = list(ngrams(tokens, ngram_len))
    y.append(guesses_nb(list(freq(token_ngrams).values())))

plt.plot(x, y)
plt.title(
    "Nombre d'essais moyens nécéssaires pour deviner le premier ngram du texte en connaissant la distribution de probabilité des ngrams du texte (1984)"
)
plt.ylabel("Nombre d'essais")
plt.xlabel("Longueur du ngram")
plt.show()

# %%
attack_tokens = []
for other_book in corpus[1:]:
    attack_tokens += load_book(other_book)[0]

x = list(range(1, 10))
y = []
for ngram_len in x:
    attack_freq = freq(list(ngrams(attack_tokens, ngram_len)))
    token_freq = freq(list(ngrams(tokens, ngram_len)))
    y.append(
        len(set(token_freq.keys()) & set(attack_freq.keys()))
        / len(set(token_freq.keys()))
    )

plt.plot(x, y)
plt.title("Couverture des ngrams de 1984 dans le reste du corpus Novelties")
plt.ylabel("Couverture")
plt.xlabel("Longueur des ngrams")
plt.show()


# %%
def coverage(novel: list[str], freq_corpus: list[str], ngram_len: int) -> float:
    novel_chunks = set(ngrams(novel, ngram_len))
    freq_corpus_chunks = set(ngrams(freq_corpus, ngram_len))
    return len(novel_chunks & freq_corpus_chunks) / len(novel_chunks)


print(coverage(tokens, brown.words(), 1))
print(coverage(tokens, brown.words(), 2))
print(coverage(tokens, brown.words(), 3))
print(coverage(tokens, brown.words(), 4))
print(coverage(tokens, brown.words(), 5))

# %%
from novelties_bookshare.encrypt import encrypt_tokens
from collections import defaultdict
from statistics import mean
import matplotlib.pyplot as plt

tokens, _ = load_book(corpus[0])

x = list(range(1, 65))
y = []
for hash_len in x:
    hash2tokens = defaultdict(set)
    encrypted = encrypt_tokens(tokens, hash_len=hash_len)
    for e, token in zip(encrypted, tokens):
        hash2tokens[e].add(token)
    y.append(mean(len(v) - 1 for v in hash2tokens.values()))

plt.plot(x, y)
for xi, yi in zip(x, y):
    plt.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=9, color="red")
plt.ylabel("Mean number of collisions for a given hash")
plt.xlabel("hash len")
plt.grid()
plt.show()

# %%
hash_len = 2
tokens, _ = load_book(corpus[0])
encrypted = encrypt_tokens(tokens, hash_len=hash_len)

perfect_dist = freq(tokens)
freq_sorted_tokens = list(reversed(sorted(perfect_dist.keys(), key=perfect_dist.get)))
freq_sorted_hashs = encrypt_tokens(freq_sorted_tokens, hash_len=hash_len)

guessed_tokens = []
for e in tqdm(encrypted, ascii=True):
    # we guess the most frequent word
    for guess_token, guess_hash in zip(freq_sorted_tokens, freq_sorted_hashs):
        if guess_hash == e:
            guessed_tokens.append(guess_token)
            break

print(mean(1 if guess == token else 0 for guess, token in zip(guessed_tokens, tokens)))

# %%
import functools as ft
import operator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from novelties_bookshare.encrypt import encrypt_token, encrypt_tokens
from statistics import mean


@ft.lru_cache(maxsize=16)
def cached_generate(text: str, model, tokenizer):
    device = torch.device("cuda")
    encoded_text = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded_text, logits_to_keep=1)
    return outputs


def token_next_prob(
    text_so_far: str, token: str, model, tokenizer, max_depth: int = 4
) -> Optional[float]:
    subtokens = tokenizer.tokenize(token)
    subtoken_ids = tokenizer(token, add_special_tokens=False)["input_ids"]

    prob = None
    probs = []
    depth = 0

    while prob is None and depth < max_depth:
        subtoken_id = subtoken_ids[0]

        outputs = cached_generate(text_so_far, model, tokenizer)
        next_token_logits = outputs.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, -1)
        probs.append(next_token_probs[subtoken_id].item())

        depth += 1
        text_so_far += subtokens[0]
        subtokens = subtokens[1:]
        subtoken_ids = subtoken_ids[1:]

        if len(subtoken_ids) == 0:
            prob = ft.reduce(operator.mul, probs)

    return prob


def next_probs(
    text_so_far: str, tokens: set[str], model, tokenizer, max_depth: int = 4
) -> dict[str, float]:
    probs = {}
    for token in tokens:
        prob = token_next_prob(
            text_so_far, token, model, tokenizer, max_depth=max_depth
        )
        probs[token] = prob or 0
    return probs


def beam_search(
    prompt: str, guessed_tokens: list[set[str]], model, tokenizer, beams_nb: int
) -> list[str]:
    assert beams_nb > 0

    def beam_score(beam: tuple[list[str], list[float]]) -> float:
        eps = 1e-16
        return sum([math.log(p + eps) for p in beam[1]])

    def inv_beam_score(beam: tuple[list[str], list[float]]) -> float:
        return -beam_score(beam)

    probs = next_probs(prompt, guessed_tokens[0], model, tokenizer)
    # a list of beams, each of the form ([tokens], [scores])
    beams = [([token], [prob]) for token, prob in probs.items()]
    beams = sorted(beams, key=inv_beam_score)[:beams_nb]

    for possible_tokens in tqdm(guessed_tokens[1:], ascii=True):
        new_beams = []
        for tokens, prob_seq in beams:
            probs = next_probs(
                prompt + " " + " ".join(tokens), possible_tokens, model, tokenizer
            )
            local_new_beams = [
                (tokens + [token], prob_seq + [prob]) for token, prob in probs.items()
            ]
            local_new_beams = sorted(local_new_beams, key=inv_beam_score)[:beams_nb]
            new_beams += local_new_beams
        beams = sorted(new_beams, key=inv_beam_score)[:beams_nb]

    return max(beams, key=beam_score)[0]


model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = model.to(torch.device("cuda"))

hash_len = 2
tokens, _ = load_book(corpus[0])
encrypted = encrypt_tokens(tokens, hash_len=hash_len)

h_voc = {t: encrypt_token(t, hash_len=hash_len) for t in set(tokens)}

guessed_tokens = []
for e in tqdm(encrypted, ascii=True):
    guessed_tokens.append({t for t, h in h_voc.items() if h == e})

# baseline
token_freq = freq(tokens)
print(
    mean(
        1 if max(guess, key=token_freq.get) == token else 0
        for guess, token in zip(guessed_tokens, tokens)
    )
)

# let's be crazy and give some starting tokens for free
start_tokens = tokens[:100]
llm_guessed_tokens = beam_search(
    "Complete the following text from 1984: {}".format(" ".join(start_tokens)),
    guessed_tokens[len(start_tokens) :][:100],
    model,
    tokenizer,
    beams_nb=4,
)

print(
    mean(
        1 if guess == token else 0
        for guess, token in zip(llm_guessed_tokens, tokens[len(start_tokens) :])
    )
)


# %%
from novelties_bookshare.experiments.errors import token_split, token_merge
