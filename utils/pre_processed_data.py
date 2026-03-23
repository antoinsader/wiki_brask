from collections import defaultdict
import os
import re
import nltk
import torch
from tqdm import tqdm

from utils.files import cache_array, read_cached_array, read_tensor
from utils.settings import _MinimizedFiles, _PreprocessedFiles, settings


class RawDataLoader:
    """Parses raw .txt files into dicts and caches them as .pkl files.
    On subsequent calls, loads directly from the .pkl cache."""

    def __init__(self, raw_files, preprocessed_files:_PreprocessedFiles, minimized_files: _MinimizedFiles):
        self.raw = raw_files
        self.pkl :_PreprocessedFiles= preprocessed_files
        self.min: _MinimizedFiles = minimized_files

    # ------------------------------------------------------------------ #
    # Private parsers (raw .txt → dict)
    # ------------------------------------------------------------------ #

    def _parse_triples(self, raw_fp) -> dict:
        result = []
        with open(raw_fp, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Parsing triples"):
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    head, relation, tail = parts
                    result.append((head, relation, tail))
        return result

    def _parse_descriptions(self, raw_fp) -> dict:
        result = {}
        with open(raw_fp, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="parsing descriptions"):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    result[parts[0]] = parts[1]
        return result

    def _parse_aliases(self, raw_fp) -> dict:
        result = defaultdict(list)
        with open(raw_fp, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in tqdm(lines, desc="parsing aliases"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                entity_id = parts[0]
                result[entity_id] = parts[1:]   # first entry is canonical name, rest are aliases
        return result

    def _parse_relations(self, raw_fp) -> dict:
        result = defaultdict(list)
        with open(raw_fp, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in tqdm(lines, desc="parsing relations"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                result[parts[0]] = parts[1:]
        return result

    # ------------------------------------------------------------------ #
    # Public getters (load pkl if exists, else parse raw and save pkl)
    # ------------------------------------------------------------------ #

    def _get(self, pkl_fp, raw_fp, parser):
        if os.path.isfile(pkl_fp):
            return read_cached_array(pkl_fp)
        if raw_fp is None or parser is None:
            raise ValueError(f"No raw file or parser provided for {pkl_fp}, and cache not found.")
        print(f"Cache not found, parsing from raw: {raw_fp}")
        data = parser(raw_fp)
        cache_array(data, pkl_fp)
        return data

    def _get_minimized(self, pkl_fp):
        if not os.path.isfile(pkl_fp):
            raise FileNotFoundError(f"Minimized cache not found: {pkl_fp}. Run minimize.py first.")
        return read_cached_array(pkl_fp)

    def _get_minimized_tensor(self, tensor_fp, mmap=False):
        if not os.path.isfile(tensor_fp):
            raise FileNotFoundError(f"Minimized cache not found: {tensor_fp}. Run minimize.py first.")
        return read_tensor(tensor_fp, mmap=mmap)


    def get_triples_train(self, minimized=False) -> dict:
        if minimized:
            return self._get_minimized(self.min.TRIPLES_TRAIN)
        return self._get(self.pkl.TRIPLES_TRAIN, self.raw.TRIPLES_TRAIN, self._parse_triples)

    def get_triples_valid(self) -> dict:
        return self._get(self.pkl.TRIPLES_VALID, self.raw.TRIPLES_VALID, self._parse_triples)

    def get_triples_test(self) -> dict:
        return self._get(self.pkl.TRIPLES_TEST, self.raw.TRIPLES_TEST, self._parse_triples)

    def get_descriptions(self, minimized=False) -> dict:
        if minimized:
            return self._get_minimized(self.min.DESCRIPTIONS)
        return self._get(self.pkl.DESCRIPTIONS, self.raw.DESCRIPTIONS, self._parse_descriptions)


    def get_aliases(self, minimized=False) -> dict:
        if minimized:
            return self._get_minimized(self.min.ALIASES)
        return self._get(self.pkl.ALIASES, self.raw.ALIASES, self._parse_aliases)

    def get_relations(self, minimized=False) -> dict:
        if minimized:
            return self._get_minimized(self.min.RELATIONS)
        return self._get(self.pkl.RELATIONS, self.raw.RELATIONS, self._parse_relations)

    def get_silver_spans(self, minimized=False) -> dict:
        if minimized:
            return self._get_minimized(self.min.SILVER_SPANS)
        return self._get(self.pkl.SILVER_SPANS, None, None)

    def get_golden_triples(self) -> torch.Tensor:
        return self._get_minimized(self.min.GOLD_TRIPLES)

    def get_description_embeddings_all(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        description_embs_all = self._get_minimized_tensor(self.min.DESCRIPTION_EMBEDDINGS_ALL, True)
        description_embs_ids = self._get_minimized(self.min.DESCRIPTION_EMBEDDINGS_IDS)
        description_embs_masks = self._get_minimized_tensor(self.min.DESCRIPTION_EMBEDDING_ALL_MASKS)
        return description_embs_all, description_embs_ids, description_embs_masks

    def get_description_embeddings_mean(self) -> torch.Tensor:
        return self._get_minimized_tensor(self.min.DESCRIPTION_EMBEDDINGS_MEAN)

    def cache_all(self):
        """Parse and cache every dataset. Skips files already cached."""
        self.get_triples_train()
        self.get_triples_valid()
        self.get_triples_test()
        self.get_descriptions()
        self.get_aliases()
        self.get_relations()


# Module-level singletons — safe to import anywhere, no side effects on load.
data_loader = RawDataLoader(settings.RAW_FILES, settings.PREPROCESSED_FILES, settings.MINIMIZED_FILES)


def check_minimized_files():
    min_files = settings.MINIMIZED_FILES
    missing = [p for p in [min_files.DESCRIPTIONS, min_files.ALIASES, min_files.TRIPLES_TRAIN, min_files.RELATIONS] if not os.path.isfile(p)]
    if missing:
        print("Minimized files not found. Run minimize.py first.")
        for p in missing:
            print(f"  Missing: {p}")
        return False
    return True

def check_preprocessed_files():
    files = settings.PREPROCESSED_FILES
    missing = [p for p in [files.DESCRIPTIONS, files.ALIASES, files.TRIPLES_TRAIN, files.RELATIONS] if not os.path.isfile(p)]
    if missing:
        print("Preprocessed files not found. Run any script with choosing no for minimization to generate them.")
        for p in missing:
            print(f"  Missing: {p}")
        return False
    return True

def check_files(use_minimized=False):
    if use_minimized:
        return check_minimized_files()

    return check_preprocessed_files()

