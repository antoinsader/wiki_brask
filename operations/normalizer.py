
import re
import unicodedata
from tqdm import tqdm


class Normalizer():
    def __init__(self, strange_chars, lowercasing=False ):
        self.strange_chars = strange_chars
        self.lowercasing = lowercasing
        self._spacer = re.compile(r'\s+')
        self._valid_word = re.compile(r"^[A-Za-z0-9.,!?;:'\"()-]+$")
        

    def __call__(self, text: str) -> str:
        """Replace strange chars, lowercasing(optional), remove extra spaces"""
        
        for pattern, repl in self.strange_chars:
            text = pattern.sub(repl, text)

        # Process word by word
        text = ' '.join(w for w in text.split() if self._valid_word.match(w))
        text = unicodedata.normalize('NFKC', text)
        if self.lowercasing:
            text = text.lower()

        return self._spacer.sub(' ', text).strip()


