NO_PUNCT = ""
COMMA = ","
PERIOD = "."
QUESTION = "?"
EXCLAMATION = "!"
SEMICOLON = ";"
DASH = "—"
COLON = ":"
PUNCTUATION = [NO_PUNCT, COMMA, PERIOD, QUESTION, EXCLAMATION, SEMICOLON, DASH, COLON]
PUNCTUATION_MAP = {
    "": 0,
    COMMA: 1,
    PERIOD: 2,
    QUESTION: 3,
    EXCLAMATION: 4,
    COLON: 5,
    DASH: 6,
    SEMICOLON: 7,
}


def get_punctuation_id(s: str):
    return PUNCTUATION_MAP.get(s, 0)

PUNCTUATION_ID_MAP = {v:k for k,v in PUNCTUATION_MAP.items()}

# punct_id = {0:"NO_PUNCT",
#              1:"COMMA",
#              2:"PERIOD",
#              3:"QUESTION",
#             }