# -*- eval: (code-cells-mode); -*-

# %%
from typing import *
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import decrypt_tokens
from novelties_bookshare.conll import load_conll2002_bio

tokens, tags = load_conll2002_bio("./tests/scratch/chapter_1.conll")

encrypted = encrypt_tokens(tokens)
decrypted = decrypt_tokens(encrypted, tags, tokens)
assert tokens == decrypted


# %% Perturbations
import random, copy
import numpy as np
from scrambledtext import ProbabilityDistributions, CorruptionEngine

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


def ocr_scramble(
    tokens: List[str], target_wer: float = 0.1, target_cer: float = 0.1
) -> List[str]:
    engine = CorruptionEngine(
        OCR_CORRUPTION_PROBS.conditional,
        OCR_CORRUPTION_PROBS.substitutions,
        OCR_CORRUPTION_PROBS.insertions,
        target_wer=0.2,
        target_cer=0.1,
    )

    corrupted_text, _, _, _ = engine.corrupt_text(" ".join(tokens))

    return corrupted_text.split()


ex = "Lianna, princess of Fomalhaut".split()
print("example of operations: ")
print(f"initial example: {ex}")
print(f"sub: {substitute(ex, 0.8)}")
print(f"del: {delete(ex, 0.8)}")
print(f"add: {add(ex, 0.8)}")
print(f"ocr_scramble: {ocr_scramble(ex)}")


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
import scienceplots
from tqdm import tqdm

plt.style.use("science")

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


def test_degradation(
    tokens: List[str],
    encrypted: List[str],
    tags: List[str],
    decrypt_fn,
    degradation_fns,
) -> float:
    user_tokens = tokens
    for degt in degradation_fns:
        user_tokens = degt(user_tokens)
    decrypted = decrypt_fn(encrypted, tags, user_tokens)
    return sum(1 if d == t else 0 for d, t in zip(decrypted, tokens)) / len(tokens)


for p in tqdm(np.arange(0.0, 1.0, 0.1)):
    p = float(p)

    for decrypt in decrypt_fns:

        # sub
        x_sub[decrypt["name"]].append(p)
        y_sub[decrypt["name"]].append(
            test_degradation(
                tokens, encrypted, tags, decrypt["fn"], [lambda t: substitute(t, p)]
            )
        )

        # del
        x_del[decrypt["name"]].append(p)
        y_del[decrypt["name"]].append(
            test_degradation(
                tokens, encrypted, tags, decrypt["fn"], [lambda t: delete(t, p)]
            )
        )

        # add
        x_add[decrypt["name"]].append(p)
        y_add[decrypt["name"]].append(
            test_degradation(
                tokens, encrypted, tags, decrypt["fn"], [lambda t: add(t, p)]
            )
        )

        # combined
        x_comb[decrypt["name"]].append(p)
        y_comb[decrypt["name"]].append(
            test_degradation(
                tokens,
                encrypted,
                tags,
                decrypt["fn"],
                [
                    lambda t: delete(t, p / 3.0),
                    lambda t: substitute(t, p / 3.0),
                    lambda t: add(t, p / 3.0),
                ],
            )
        )


fig, axs = plt.subplots(2, 2)

for decrypt in decrypt_fns:
    name = decrypt["name"]
    axs[0][0].plot(x_sub[name], y_sub[name], label=name)
axs[0][0].grid()
axs[0][0].set_xlabel("Percentage of substitutions")
axs[0][0].set_ylabel("Percentage of recovered tokens")
axs[0][0].set_ylim(-0.05, 1.05)
axs[0][0].legend()

for decrypt in decrypt_fns:
    name = decrypt["name"]
    axs[0][1].plot(x_del[name], y_del[name], label=name)
axs[0][1].grid()
axs[0][1].set_xlabel("Percentage of deletions")
axs[0][1].set_ylabel("Percentage of recovered tokens")
axs[0][1].set_ylim(-0.05, 1.05)
axs[0][1].legend()

for decrypt in decrypt_fns:
    name = decrypt["name"]
    axs[1][0].plot(x_add[name], y_add[name], label=name)
axs[1][0].grid()
axs[1][0].set_xlabel("Percentage of additions")
axs[1][0].set_ylabel("Percentage of recovered tokens")
axs[1][0].set_ylim(-0.05, 1.05)
axs[1][0].legend()

for decrypt in decrypt_fns:
    name = decrypt["name"]
    axs[1][1].plot(x_comb[name], y_comb[name], label=name)
axs[1][1].grid()
axs[1][1].set_xlabel("Percentage of combined add/del/sub")
axs[1][1].set_ylabel("Percentage of recovered tokens")
axs[1][1].set_ylim(-0.05, 1.05)
axs[1][1].legend()

fig.suptitle("Impact of sub/del/add")
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
