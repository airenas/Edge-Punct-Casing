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
            strs_mi = strs[1].split(")", 1)
            if len(strs_mi) == 2:
                self.mi = strs_mi[0]
                self.word, self.punct = split_word_punctuation(self.word + strs_mi[1]) # add punctuation to word
        else:
            self.word, self.punct = split_word_punctuation(string)
            self.mi = ""

    def to_str(self):
        if self.mi:
            return f"{self.word}{self.punct}(={self.mi})"
        else:
            return f"{self.word}{self.punct}"


def split_word_punctuation(word):
    w = ""
    for c in word:
        if c.isalpha():
            w += c
        else:
            return w, word[len(w):]
    return w, ""
