import os
import pytest

from prepare import choose_random_ids, minimmizing_triples
from utils.files import read_cached_array
from utils.settings import settings


@pytest.fixture(scope="module")
def minimized_triples_ids():
    full_files = settings.PREPROCESSED_FILES
    full_triples = read_cached_array(full_files.TRIPLES_TRAIN)
    triples_ids = [t[0] for t in full_triples]
    total = len(triples_ids)
    n = max(1, int(total * 0.0001))
    return choose_random_ids(triples_ids, n)


@pytest.fixture(scope="module")
def run_minimizing(minimized_triples_ids):
    min_files = settings.MINIMIZED_FILES
    raw_fp = settings.RAW_FILES.TRIPLES_TRAIN
    minimmizing_triples(minimized_triples_ids, raw_fp, min_files)
    return min_files


def test_output_file_exists(run_minimizing):
    assert os.path.exists(run_minimizing.TRIPLES_TRAIN), (
        f"Output file not found: {run_minimizing.TRIPLES_TRAIN}"
    )


def test_triple_count_matches(minimized_triples_ids, run_minimizing):
    saved = read_cached_array(run_minimizing.TRIPLES_TRAIN)
    saved_heads = set(t[0] for t in saved)
    assert saved_heads.issubset(minimized_triples_ids), (
        "Saved triples contain heads not in minimized_triples_ids"
    )
    assert len(saved) > 0, "No triples were saved"


def test_triples_are_valid_tuples(run_minimizing):
    saved = read_cached_array(run_minimizing.TRIPLES_TRAIN)
    for i, triple in enumerate(saved):
        assert len(triple) == 3, f"Triple at index {i} does not have 3 elements: {triple}"
        head, relation, tail = triple
        assert isinstance(head, str), f"head at index {i} is not a str: {type(head)}"
        assert isinstance(relation, str), f"relation at index {i} is not a str: {type(relation)}"
        assert isinstance(tail, str), f"tail at index {i} is not a str: {type(tail)}"
