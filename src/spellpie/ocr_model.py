"""
OCR model. When doing OCR, much of the incoming data is likely
    to be fairly replete with garbage. This model will limit the
    scope of the OCR (not to the sentence level), and ignore lines
    which appear to lack any content at all (e.g., no in-dictionary
    words).
"""
import collections
import re

from spellpie.lm import TrigramLanguageModel

Word = collections.namedtuple('Word', 'word start end is_word orig_word')


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
            return Word('<EOS>', None, None, None, None)
        return self.words[self.curr + 1]

    def prev_word(self):
        """get previous word while iterating through Words object"""
        if self.curr is None:
            raise ValueError('No current word. Use for loop to create this context.')
        if self.curr <= 0:
            return Word('<BOS>', None, None, None, None)
        return self.words[self.curr - 1]

    def next_next_word(self):
        """get next word while iterating through Words object"""
        if self.curr is None:
            raise ValueError('No current word. Use for loop to create this context.')
        if self.curr >= len(self.words) - 2:
            return Word('<EOS>', None, None, None, None)
        return self.words[self.curr + 2]

    def prev_prev_word(self):
        """get previous word while iterating through Words object"""
        if self.curr is None:
            raise ValueError('No current word. Use for loop to create this context.')
        if self.curr <= 0:
            return Word('<BOS>', None, None, None, None)
        return self.words[self.curr - 2]


class OcrSpellCorrector:

    def __init__(self):
        self.changes = []

    def spell_correct_line(self, lm: TrigramLanguageModel, line, cutoff=3):
        newline = []
        pat = re.compile(r'[A-Za-z]+')
        words = Words()
        apparent_word_ctr = 0
        for m in pat.finditer(line):
            word = m.group()
            apparent_word = len(word) > cutoff and word in lm
            if apparent_word:
                apparent_word_ctr += 1
            words.append(Word(word.lower(), m.start(), m.end(), apparent_word, word))
        idx = 0
        for word in words:
            if word.word in lm or len(word.word) <= cutoff:
                newline.append(line[idx:word.end])
            else:  # out-of-vocab long word
                ppw = words.prev_prev_word().word
                pw = words.prev_word().word
                nw = words.next_word().word
                nnw = words.next_next_word().word
                best_candidate = None
                best_score = 0
                for cand, diff in ((word.word, 0), ) + tuple(lm.generate_candidates(word.word)):
                    score = lm.sum((cand,), (pw, cand), (ppw, pw, cand),
                                   (pw, cand, nw), (cand, nw), (cand, nw, nnw))
                    if not best_candidate or score > best_score:
                        best_candidate = cand
                        best_score = score
                newline.append(line[idx:word.start])
                newword = ''.join(x if x.lower() == y.lower() else y for x, y in zip(word.orig_word, best_candidate))
                self.changes.append((word.orig_word, newword, line[word.start - 100:word.end + 100]))
                newline.append(newword)
            idx = word.end
        newline.append(line[idx:])
        return ''.join(newline)
