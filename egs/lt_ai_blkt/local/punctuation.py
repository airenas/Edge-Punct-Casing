COMMA = ","
DOT = "."
QUESTION = "?"
EXCLAMATION = "!"
PUNCTUATION = [COMMA, DOT, QUESTION, EXCLAMATION]
PUNCTUATION_MAP = {
    "": 0,
    COMMA: 1,
    DOT: 2,
    QUESTION: 3,
    EXCLAMATION: 4,
}


def get_punctuation_id(s: str):
    return PUNCTUATION_MAP.get(s, 0)
