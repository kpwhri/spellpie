"""
OCR model. When doing OCR, much of the incoming data is likely
    to be fairly replete with garbage. This model will limit the
    scope of the OCR (not to the sentence level), and ignore lines
    which appear to lack any content at all (e.g., no in-dictionary
    words).
"""
import collections

import regex as re
from itertools import zip_longest

from spellpie.algos.base import SpellCorrector
from spellpie.lm import TrigramLanguageModel
from spellpie.noise.ocr_channel import OcrNoisyChannel

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


def isascii(s):
    try:
        s.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False
    # return len(s) == len(s.encode())


class OcrSpellCorrector(SpellCorrector):

    def __init__(self):
        super().__init__()
        self.noisy_channel = OcrNoisyChannel()

    def spell_correct_line(self, lm: TrigramLanguageModel, line, cutoff=3, tag=None, **kwargs):
        newline = []
        pat = re.compile(r'[\p{Letter}]+')
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
            if word.word in lm:
                newline.append(line[idx:word.end])
            elif len(word.word) <= cutoff:
                # handle unicode
                if word.word in lm or isascii(word.word):
                    newline.append(line[idx:word.end])
                else:
                    self.get_best_candidate(lm, word, idx, words, newline, line, tag=tag)
            else:  # out-of-vocab long word
                self.get_best_candidate(lm, word, idx, words, newline, line, tag=tag)
            idx = word.end
        newline.append(line[idx:])
        return ''.join(newline)

    def get_best_candidate(self, lm, word, idx, words, newline, line, tag=None):
        ppw = words.prev_prev_word().word
        pw = words.prev_word().word
        nw = words.next_word().word
        nnw = words.next_next_word().word
        best_candidate = None
        best_score = 0
        for cand, diff in ((word.word, 0),) + tuple(lm.generate_candidates(word.word,
                                                                           noisy_channel=self.noisy_channel)):
            score = lm.sum(cand, (pw, cand), (ppw, pw, cand),
                           (pw, cand, nw), (cand, nw), (cand, nw, nnw))
            if not best_candidate or score > best_score:
                best_candidate = cand
                best_score = score
        newline.append(line[idx:word.start])
        newword = ''.join(x if x.lower() == y.lower() else y
                          for x, y in zip_longest(word.orig_word, best_candidate, fillvalue=''))
        if best_candidate != word.word:
            self.changes.append((word.orig_word, newword, line[max(word.start - 100, 0):word.end + 100], tag))
        newline.append(newword)


def main(model_path, input_path, output_path):
    """
    Process a single file. This is mainly for testing/debugging.
    :param model_path:
    :param input_path:
    :param output_path:
    :return:
    """
    from spellpie.lm import TrigramLanguageModel
    lm = TrigramLanguageModel.frompickle(model_path)
    osc = OcrSpellCorrector()
    with open(input_path, encoding='utf8') as fh, \
            open(output_path, 'w', encoding='utf8') as out:
        for i, line in enumerate(fh):
            new_line = osc.spell_correct_line(lm, line, tag='sample')
            out.write(new_line)

    with open('changes.tsv', 'w', encoding='utf8') as out:
        out.write('orig_word\tnew_word\tcontext\ttag\n')
        for orig_word, new_word, context, tag in osc.changes:
            out.write(f'{orig_word}\t{new_word}\t{" ".join(context.split())}\t{tag}\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-m', '--model-path', dest='model_path',
                        help='Path to model pickle file')
    parser.add_argument('-i', '--input-path', dest='input_path',
                        help='Path to model pickle file')
    parser.add_argument('-o', '--output-path', dest='output_path',
                        help='Path to model pickle file')
    args = parser.parse_args()

    main(args.model_path, args.input_path, args.output_path)
