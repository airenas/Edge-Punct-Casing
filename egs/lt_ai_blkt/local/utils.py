def several_upper(s):
    # Check if the string contains at least two uppercases alphabetic character
    return sum(1 for c in s if c.isupper()) >= 2

def has_upper(s):
    for c in s:
        if c.isupper():
            return True
    return False

class Word:
    def __init__(self, string):
        strs = string.split("(=", 1)
        if len(strs) == 2:
            self.word = strs[0]
            self.mi = strs[1][:-1]
        else:
            self.word = string
            self.mi = ""

    def to_str(self):
        if self.mi:
            return f"{self.word}(={self.mi})"
        else:
            return self.word
