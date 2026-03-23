from tqdm import tqdm
import sys
import re

import os



def get_strange_chars():
    return [(re.compile(p, re.IGNORECASE), r) for p, r in {
            r'[€"©]': "",
            r"[áăắặằẳẵǎâấậầẩẫäǟȧǡạȁàảȃāąåǻḁãǽǣ]": "a",
            r"[ḃḅḇ]": "b",
            r"[ćčçḉĉċ]": "c",
            r"[ďḑḓḋḍḏ]": "d",
            r"[éĕěȩḝêếệềểễḙëėẹȅèẻȇēḗḕęẽḛé]": "e",
            r"[ḟ]": "f",
            r"[ǵğǧģĝġḡ]": "g",
            r"[ḫȟḩĥḧḣḥẖ]": "h",
            r"[íĭǐîïḯi̇ịȉìỉȋīįĩḭ]": "i",
            r"[ǰĵ]": "j",
            r"[ḱǩķḳḵ]": "k",
            r"[ĺľļḽḷḹḻ]": "l",
            r"[ḿṁṃ]": "m",
            r"[ńňņṋṅṇǹṉñ]": "n",
            r"[óŏǒôốộồổỗöȫȯȱọőȍòỏơớợờởỡȏōṓṑǫǭõṍṏȭǿøɔ]": "o",
            r"[ṕṗ]": "p",
            r"[ŕřŗṙṛṝȑȓṟ]": "r",
            r"[śṥšṧşŝșṡẛṣṩ]": "s",
            r"[ťţṱțẗṫṭṯ]": "t",
            r"[úŭǔûṷüǘǚǜǖṳụűȕùủưứựừửữȗūṻųůũṹṵ]": "u",
            r"[ṿṽ]": "v",
            r"[ẃŵẅẇẉẁẘ]": "w",
            r"[ẍẋ]": "x",
            r"[ýŷÿẏỵỳỷȳẙỹy]": "y",
            r"[źžẑżẓẕʐ]": "z",
            r"[&]": "and",
        }.items()]

def get_stop_words() -> set:
    import nltk 
    nltk.download("stopwords")
    stop_words = set(nltk.corpus.stopwords.words("english"))
    return stop_words




def ask_factor(prompt):
    while True:
        try:
            val = float(input(prompt).strip())
            if 0 < val <= 1:
                return val
            print(f"Must be between 0 and 1")
        except ValueError:
            print("Invalid input- choose a decimal number between 0 and 1")


def timed_input(prompt: str, timeout: int = 10, default: str = "y") -> str:
    """Cross-platform timed input.
    Returns user input if provided within timeout, otherwise returns default.
    """
    print(f"{prompt} (auto-answer '{default}' in {timeout}s): ", end="", flush=True)

    if os.name == "nt":
        # Windows
        import msvcrt
        import time
        start = time.time()
        chars = []
        while True:
            if msvcrt.kbhit():
                c = msvcrt.getwche()
                if c in ("\r", "\n"):
                    print()
                    return "".join(chars).strip() or default
                chars.append(c)
            if time.time() - start > timeout:
                print()
                return default

    else:
        pp = 1
        # Linux / macOS
        import select
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            answer = sys.stdin.readline().strip()
            return answer if answer else default
        else:
            print()  # newline after the prompt
            return default


def create_aliases_patterns_map(aliases : dict) -> dict[str, re.Pattern]:
    """Creates a map of alias strings to regex patterns that match them."""
    patterns_map = {}
    for als_lst in tqdm(aliases.values(), desc="creating aliases patterns map"):
        for als_str in als_lst:
            escaped = re.escape(als_str)
            flexible = escaped.replace(r'\ ', r'\s+')
            pattern = rf"(?<!\w){flexible}(?!\w)"
            patterns_map[als_str] = re.compile(pattern, flags=re.IGNORECASE)
    del aliases
    return patterns_map
