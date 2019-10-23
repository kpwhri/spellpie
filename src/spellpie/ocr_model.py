"""
OCR model. When doing OCR, much of the incoming data is likely
    to be fairly replete with garbage. This model will limit the
    scope of the OCR (not to the sentence level), and ignore lines
    which appear to lack any content at all (e.g., no in-dictionary
    words).
"""
import collections
import re

Word = collections.namedtuple('Word', 'word start end is_word')


class Words:

    def __init__(self):
        self.words = []
        self.curr = None

    def append(self, obj):
        self.words.append(obj)

    def __iter__(self):
        for i, word in enumerate(self.words):
            self.curr = i
            yield word
        self.curr = None

    def next_word(self):
        """get next word while iterating through Words object"""
        if self.curr is None:
            raise ValueError('No current word. Use for loop to create this context.')
        if self.curr >= len(self.words) - 1:
            return '<EOS>'
        return self.words[self.curr + 1]

    def prev_word(self):
        """get previous word while iterating through Words object"""
        if self.curr is None:
            raise ValueError('No current word. Use for loop to create this context.')
        if self.curr <= 0:
            return '<BOS>'
        return self.words[self.curr - 1]

    def next_next_word(self):
        """get next word while iterating through Words object"""
        if self.curr is None:
            raise ValueError('No current word. Use for loop to create this context.')
        if self.curr >= len(self.words) - 2:
            return '<EOS>'
        return self.words[self.curr + 2]

    def prev_prev_word(self):
        """get previous word while iterating through Words object"""
        if self.curr is None:
            raise ValueError('No current word. Use for loop to create this context.')
        if self.curr <= 0:
            return '<BOS>'
        return self.words[self.curr - 2]


def ocr_spell_correct_line(lm, line, cutoff=3):
    newline = []
    pat = re.compile(r'[A-Za-z]+')
    words = Words()
    apparent_word_ctr = 0
    for m in pat.finditer(line):
        word = m.group()
        apparent_word = len(word) > cutoff and word in lm
        if apparent_word:
            apparent_word_ctr += 1
        words.append(Word(word, m.start(), m.end(), apparent_word))
    idx = 0
    for word in words:
        if word in lm or len(word) <= cutoff:
            newline.append(line[idx:word.end])
            idx = word.end
    newline.append(line[idx:])
    return ''.join(newline)
