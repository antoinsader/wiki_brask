from tqdm import tqdm
import sys
import time
import re


import math


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
            r"[íĭǐîïḯi̇ịȉìỉȋīįĩḭı]": "i",
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

def timed_input(prompt, timeout=15, default="y"):
    """Prompt the user with a countdown. Returns default if no answer within timeout."""
    import msvcrt
   
    print(prompt)
    chars = []
    end_time = time.time() + timeout

    while True:
        remaining = end_time - time.time()
        if remaining <= 0:
            sys.stdout.write(f"\r  → No input detected, proceeding with '{default}'...        \n")
            sys.stdout.flush()
            return default

        secs = math.ceil(remaining)
        line = f"  [auto-'{default}' in {secs}s] > {''.join(chars)}"
        sys.stdout.write(f"\r{line}  ")
        sys.stdout.flush()

        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ('\r', '\n'):
                sys.stdout.write("\n")
                sys.stdout.flush()
                user_input = ''.join(chars).strip().lower()
                return user_input if user_input else default
            elif ch == '\x08':  # backspace
                if chars:
                    chars.pop()
            elif ch == '\x03':  # Ctrl-C
                raise KeyboardInterrupt
            else:
                chars.append(ch)

        time.sleep(0.05)


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
