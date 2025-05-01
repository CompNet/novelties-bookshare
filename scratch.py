# -*- eval: (code-cells-mode); -*-

# %%
from typing import *
import os, glob
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import decrypt_tokens
from novelties_bookshare.conll import load_conll2002_bio

tokens = []
tags = []
for path in sorted(
    glob.glob(os.path.expanduser("~/Dev/Novelties/corpus/1984/*.conll"))
):
    chapter_tokens, chapter_tags = load_conll2002_bio(path)
    tokens += chapter_tokens
    tags += chapter_tags

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
        noisy_tokens[i] = "[UNK]"

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
                noisy_tokens.append(token[split_idx:])
                noisy_tokens.append(token[:split_idx])
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
    for i, (tok1, tok2) in enumerate(windowed(tokens, 2)):
        if i in merge_indices:
            noisy_tokens.append(tok1 + tok2)
        else:
            noisy_tokens.append(tok1)
            if i == len(tokens) - 2:
                noisy_tokens.append(tok2)

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


# %%
import difflib


def decrypt_tokens_naive(
    encrypted_tokens: list,
    tags: List[str],
    user_tokens: List[str],
    hash_len: Optional[int] = None,
):
    assert len(encrypted_tokens) == len(tags)

    decrypted_tokens = ["[UNK]" for _ in encrypted_tokens]

    encrypted_user_tokens = encrypt_tokens(user_tokens, hash_len=hash_len)

    # loop over operations turning encrypted_tokens into
    # encrypted_user_tokens
    matcher = difflib.SequenceMatcher(None, encrypted_tokens, encrypted_user_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # NOTE: ignores 'insert', 'replace', 'delete'
        if tag == "equal":
            decrypted_tokens[i1:i2] = user_tokens[j1:j2]

    assert len(decrypted_tokens) == len(tags)
    return decrypted_tokens


# %%
from collections import defaultdict
import matplotlib.pyplot as plt
from more_itertools import flatten
from tqdm import tqdm

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
    {"name": "naive", "fn": decrypt_tokens_naive},
    {"name": "advanced", "fn": decrypt_tokens},
]

degradations = [
    {
        "name": "add",
        "fn": [add],
        "x": {d["name"]: [] for d in decrypt_fns},
        "y": {d["name"]: [] for d in decrypt_fns},
    },
    {
        "name": "del",
        "fn": [delete],
        "x": {d["name"]: [] for d in decrypt_fns},
        "y": {d["name"]: [] for d in decrypt_fns},
    },
    {
        "name": "sub",
        "fn": [substitute],
        "x": {d["name"]: [] for d in decrypt_fns},
        "y": {d["name"]: [] for d in decrypt_fns},
    },
    {
        "name": "ocr",
        "fn": [ocr_scramble],
        "x": {d["name"]: [] for d in decrypt_fns},
        "y": {d["name"]: [] for d in decrypt_fns},
    },
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
    tokens: List[str],
    encrypted: List[str],
    tags: List[str],
    decrypt_fn,
    degradation_fns: list,
) -> float:
    user_tokens = tokens
    for degt in degradation_fns:
        user_tokens = degt(user_tokens)
    decrypted = decrypt_fn(encrypted, tags, user_tokens)
    return sum(1 if d == t else 0 for d, t in zip(decrypted, tokens)) / len(tokens)


for p in tqdm(np.arange(0.0, 0.3, 0.01)):

    p = float(p)

    for decrypt in decrypt_fns:

        for degradation in degradations:

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


fig, axs = plt.subplots(1 + len(degradations) // 3, 2)
axs = list(flatten(axs))

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
        ax.set_ylim(-0.05, 1.05)
        ax.legend()

fig.suptitle("Impact of errors")
plt.show()

# %%
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


# %%
tokens = "La princesse de Fomalhaut".split()
tags = ["O"] * len(tokens)
user_tokens = "La de Fomalhaut princesse".split()
encrypted = encrypt_tokens(tokens)
decrypt_tokens(encrypted, tags, user_tokens)
