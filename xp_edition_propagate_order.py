from typing import Optional, Literal
import time
from more_itertools import flatten
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from tqdm import tqdm
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import (
    decrypt_tokens,
    make_plugin_mlm,
    make_plugin_propagate,
    make_plugin_case,
    make_plugin_split,
)
from novelties_bookshare.experiments.data import (
    iter_book_chapters,
    normalize_,
    EDITION_SETS,
)
from novelties_bookshare.experiments.metrics import record_decryption_metrics_

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))


@ex.config
def config():
    novel: str
    hash_len: int = 64
    chapter_limit: Optional[int] = None
    device: Literal["auto", "cuda", "cpu"] = "auto"


@ex.automain
def main(
    _run: Run,
    novel: str,
    hash_len: int,
    chapter_limit: Optional[int],
    device: Literal["auto", "cuda", "cpu"],
):
    print_config(_run)
    assert novel in EDITION_SETS
    assert hash_len > 0 and hash_len <= 64

    reference_edition = list(EDITION_SETS[novel].keys())[0]
    reference_chapters = list(
        iter_book_chapters(
            EDITION_SETS[novel][reference_edition], chapter_limit=chapter_limit
        )
    )

    wild_editions = {
        key: list(iter_book_chapters(path, chapter_limit=chapter_limit))
        for key, path in EDITION_SETS[novel].items()
        if key != reference_edition
    }

    normalize_(reference_chapters)
    for chapters in wild_editions.values():
        normalize_(chapters)

    reference_encrypted = [
        encrypt_tokens(chapter, hash_len=hash_len) for chapter in reference_chapters
    ]

    pipelines = [
        (
            "first",
            [
                make_plugin_propagate(),
                make_plugin_case(),
                make_plugin_split(max_token_len=24, max_splits_nb=4),
                make_plugin_mlm(
                    "answerdotai/ModernBERT-base", window=16, device=device
                ),
            ],
        ),
        (
            "last",
            [
                make_plugin_case(),
                make_plugin_split(max_token_len=24, max_splits_nb=4),
                make_plugin_mlm(
                    "answerdotai/ModernBERT-base", window=16, device=device
                ),
                make_plugin_propagate(),
            ],
        ),
        (
            "first+last",
            [
                make_plugin_propagate(),
                make_plugin_case(),
                make_plugin_split(max_token_len=24, max_splits_nb=4),
                make_plugin_mlm(
                    "answerdotai/ModernBERT-base", window=16, device=device
                ),
                make_plugin_propagate(),
            ],
        ),
        (
            "interleave",
            [
                make_plugin_propagate(),
                make_plugin_case(),
                make_plugin_propagate(),
                make_plugin_split(max_token_len=24, max_splits_nb=4),
                make_plugin_propagate(),
                make_plugin_mlm(
                    "answerdotai/ModernBERT-base", window=16, device=device
                ),
                make_plugin_propagate(),
            ],
        ),
    ]

    progress = tqdm(total=len(wild_editions) * len(pipelines), ascii=True)

    for edition, user_tokens in wild_editions.items():
        for pipe_title, pipe in pipelines:
            progress.set_description(f"{edition}.p={pipe_title}")

            t0 = time.process_time()
            decrypted_tokens = decrypt_tokens(
                reference_encrypted,
                user_tokens,
                hash_len=hash_len,
                decryption_plugins=pipe,
            )
            t1 = time.process_time()

            reference_tokens = list(flatten(reference_chapters))
            setup_name = f"p={pipe_title}.e={novel},{edition}"
            record_decryption_metrics_(
                _run,
                setup_name,
                reference_tokens,
                decrypted_tokens,
                t1 - t0,
            )

            progress.update()
