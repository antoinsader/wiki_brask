from dataclasses import dataclass, field
import os


@dataclass
class _RAW_FILES:
    raw_dir: str

    ALIASES: str = "wikidata5m_entity.txt"
    RELATIONS: str = "wikidata5m_relation.txt"
    DESCRIPTIONS: str = "wikidata5m_text.txt"
    TRIPLES_TEST: str = "wikidata5m_transductive_test.txt"
    TRIPLES_VALID: str = "wikidata5m_transductive_valid.txt"
    TRIPLES_TRAIN: str = "wikidata5m_transductive_train.txt"

    def __post_init__(self):
        for attr in ["ALIASES", "RELATIONS", "DESCRIPTIONS", "TRIPLES_TEST", "TRIPLES_VALID", "TRIPLES_TRAIN"]:
            filename = getattr(self, attr)
            setattr(self, attr, os.path.join(self.raw_dir, filename))

    def validate(self):
        """Raise if any raw file is missing. Call explicitly when you need the raw files."""
        for attr in ["ALIASES", "RELATIONS", "DESCRIPTIONS", "TRIPLES_TEST", "TRIPLES_VALID", "TRIPLES_TRAIN"]:
            path = getattr(self, attr)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Raw file not found: {path}")


@dataclass
class _PreprocessedFiles:
    preprocessed_dir: str

    ALIASES: str = "aliases.pkl"
    RELATIONS: str = "relations.pkl"
    DESCRIPTIONS: str = "descriptions.pkl"
    TRIPLES_TEST: str = "triples_test.pkl"
    TRIPLES_VALID: str = "triples_valid.pkl"
    TRIPLES_TRAIN: str = "triples_train.pkl"

    RELATIONS_EMBEDDINGS: str = "relation_embeddings.npz"
    TRANSE_MODEL_RESULTS: str = "transe_rel_embs.npz"
    SILVER_SPANS: str = "silver_spans.pkl"

    REL2IDX: str = "rel2idx.pkl"

    def __post_init__(self):
        for attr in ["ALIASES", 
                     "RELATIONS",
                     "DESCRIPTIONS",
                     "TRIPLES_TEST",
                     "TRIPLES_VALID",
                     "TRIPLES_TRAIN",
                     "RELATIONS_EMBEDDINGS",
                     "TRANSE_MODEL_RESULTS",
                     "SILVER_SPANS",
                     "REL2IDX"]:
            filename = getattr(self, attr)
            setattr(self, attr, os.path.join(self.preprocessed_dir, filename))


@dataclass
class _HELPERS_FILES:
    helpers_dir: str

    STRANGE_CHARS: str = "strange_chars.pkl"
    STOP_WORDS: str = "stop_words.pkl"

    def __post_init__(self):
        for attr in ["STRANGE_CHARS", "STOP_WORDS"]:
            filename = getattr(self, attr)
            setattr(self, attr, os.path.join(self.helpers_dir, filename))


@dataclass
class _MinimizedFiles:
    minimized_dir: str

    ALIASES: str = "aliases.pkl"
    RELATIONS: str = "relations.pkl"
    DESCRIPTIONS: str = "descriptions.pkl"
    TRIPLES_TRAIN: str = "triples_train.pkl"
    RELATIONS_EMBEDDINGS: str = "relation_embeddings.npy"
    TRANSE_MODEL_RESULTS: str = "transe_rel_embs.npy"
    SILVER_SPANS: str = "silver_spans.pkl"
    GOLD_TRIPLES: str = "golden_triples.pkl"

    DESCRIPTION_EMBEDDINGS_ALL: str = "description_embeddings_all.npy"
    DESCRIPTION_EMBEDDINGS_MEAN: str = "description_embeddings_mean.npy"
    DESCRIPTION_EMBEDDING_ALL_MASKS : str = "description_embeddings_all_masks.npy"
    DESCRIPTION_EMBEDDINGS_IDS : str = "description_embeddings_ids.pkl"

    REL2IDX: str = "rel2idx.pkl"
    MINIMIZE_META: str = "minimize_meta.json"

    def __post_init__(self):
        for attr in ["ALIASES",
                     "RELATIONS",
                     "DESCRIPTIONS",
                     "TRIPLES_TRAIN",
                     "RELATIONS_EMBEDDINGS",
                     "TRANSE_MODEL_RESULTS",
                     "SILVER_SPANS",
                     "DESCRIPTION_EMBEDDINGS_ALL",
                     "DESCRIPTION_EMBEDDINGS_MEAN",
                        "DESCRIPTION_EMBEDDING_ALL_MASKS",
                     "DESCRIPTION_EMBEDDINGS_IDS",
                     "GOLD_TRIPLES",
                     "REL2IDX",
                     "MINIMIZE_META"]:
            filename = getattr(self, attr)
            setattr(self, attr, os.path.join(self.minimized_dir, filename))


@dataclass
class _FOLDERS:
    RAW_DIR: str = "./data/raw/"
    PREPROCESSED_DIR: str = "./data/preprocessed/"
    HELPERS_DIR: str = "./data/helpers/"
    MINIMIZED_DIR: str = "./data/minimized/"

    def __post_init__(self):
        for folder in self.__dict__.values():
            os.makedirs(folder, exist_ok=True)


@dataclass
class _SETTINGS:
    MINIMIZE_FACTOR: float = 0.001

    FOLDERS: _FOLDERS = field(default_factory=_FOLDERS)
    RAW_FILES: _RAW_FILES = field(init=False)
    PREPROCESSED_FILES: _PreprocessedFiles = field(init=False)
    HELPERS_FILES: _HELPERS_FILES = field(init=False)
    MINIMIZED_FILES: _MinimizedFiles = field(init=False)

    def __post_init__(self):
        self.RAW_FILES = _RAW_FILES(raw_dir=self.FOLDERS.RAW_DIR)
        self.PREPROCESSED_FILES = _PreprocessedFiles(preprocessed_dir=self.FOLDERS.PREPROCESSED_DIR)
        self.HELPERS_FILES = _HELPERS_FILES(helpers_dir=self.FOLDERS.HELPERS_DIR)
        self.MINIMIZED_FILES = _MinimizedFiles(minimized_dir=self.FOLDERS.MINIMIZED_DIR)


settings = _SETTINGS()
