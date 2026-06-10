def first_upper(s):
    if len(s) == 0:
        return False
    if s[0].isupper():
        for c in s[1:]:
            if c.isupper():
                return False
        return True
    return False


def all_upper(s):
    if len(s) == 0:
        return False
    for c in s:
        if not c.isupper():
            return False
    return True


def mixed_upper(s):
    if len(s) == 0:
        return False
    has_uppercase = False
    has_lowercase = False
    if s[0].islower():
        has_lowercase = True
    for c in s[1:]:
        if c.isupper():
            has_uppercase = True
        elif c.islower():
            has_lowercase = True
        if has_uppercase and has_lowercase:
            return True
    return False


UPPER = 1
LOWER = 0
CAP = 2
MIX_CASE = 3


def get_case_id(s: str):
    if first_upper(s):
        return CAP
    elif all_upper(s):
        return UPPER
    elif mixed_upper(s):
        return MIX_CASE
    return LOWER


CASE_ID_MAP = {LOWER: "LOWER",
               UPPER: "UPPER",
               CAP: "CAP",
               MIX_CASE: "MIX_CASE",
               }
