import os
import re
from collections import defaultdict

from spellpie.algos.base import SpellCorrector
from spellpie.lm import TrigramLanguageModel, load_model


class History:

    def __init__(self, lm):
        self.history = defaultdict(dict)  # index is state, list of ([ngram], score)
        self.index = 0
        self.lm = lm

    def append(self, item, prob, index=None):
        if not index:
            index = self.index
        self.history[index][item] = prob

    @classmethod
    def fromdict(cls, data):
        m = cls(None)
        m.data = data['data']
        m.smoothed_prob = data['smoothed_prob']
        return m

    def next(self, index=None):
        if index:
            self.index = index
        self.index += 1

    def best_candidates(self):
        return [x[0] for x in sorted(self, key=lambda x: -x[1])]

    def best_candidate(self):
        return self.best_candidates()[0]

    def __bool__(self):
        return self.index > 0

    def __iter__(self):
        # can't iterate current index
        for item, prob in self.history[self.index - 1].items():
            yield item, prob

    def __getitem__(self, item):
        try:
            return self.history[self.index - 1][item]
        except KeyError:
            pass
        prob = 0
        for i, word in enumerate(item):
            prob += self.lm[word]
            if i > 0:
                prob += self.lm[(item[-1], word)]
        return prob


class ViterbiSpellCorrector(SpellCorrector):

    SMALL_AMOUNT = 0.1
    COMBINATION_PENALTY = 6.2  # between 3-6

    def spell_correct_line(self, lm: TrigramLanguageModel, line, **kwargs):
        return self.viterbi(lm, line, **kwargs).best_candidate()

    def calculate_next_step(self, wordlist, lm, history: History, history_index=None,
                            first_value_increment=None,
                            penalty=0.0,
                            require_word_exists=False):
        if first_value_increment is None:
            first_value_increment = self.SMALL_AMOUNT
        for ci, (candidate, level) in enumerate(wordlist):
            if not candidate:
                continue
            if require_word_exists and candidate not in lm:
                continue
            candidate_prob = lm[candidate] + penalty
            # if ci == 0:
            #     candidate_prob += first_value_increment  # prefer the current word to equal probability options
            if not history:  # populate history
                if history_index is None:  # initialization
                    history.append((candidate,), candidate_prob)
                else:  # combined words, add penalty
                    curr_prob = candidate_prob + lm.smoothed_prob(candidate) + (penalty * level)
                    history.append((candidate,), curr_prob, index=history_index)
            else:
                best_path = None
                best_prob = 0
                for curr_path, prob in history:
                    curr_prob = prob + lm[(curr_path[-1], candidate)] + candidate_prob + (penalty * level)
                    if not best_path or curr_prob > best_prob:
                        best_path = curr_path
                        best_prob = curr_prob
                history.append(best_path + (candidate,), best_prob, index=history_index)

    def viterbi(self, lm: TrigramLanguageModel, sentence: tuple, combine_neighbors=True, penalty=None):
        if penalty is None:
            penalty = self.COMBINATION_PENALTY
        history = History(lm)
        for i, word in enumerate(sentence):
            word = word.lower().strip()
            if not word:
                continue
            if word in lm:
                penalty = lm.borrowed_prob(word)
            else:
                penalty = - self.SMALL_AMOUNT
            self.calculate_next_step([(word, 0)] + list(lm.generate_candidates(word)), lm, history, penalty=penalty)
            if combine_neighbors and i + 1 < len(sentence):  # candidates from combining words together
                cword = word + sentence[i + 1]
                self.calculate_next_step(list(lm.generate_candidates(cword)),  # includes cword in output
                                         lm, history,
                                         first_value_increment=0,
                                         penalty=lm.borrowed_prob(cword) + penalty,  # still count as 2 words
                                         history_index=history.index + 1)
            history.next()
        return history


def correct_sentence(sentence, lm, **kwargs):
    return ViterbiSpellCorrector().spell_correct_line(sentence, lm, **kwargs)


def correct_file(file, lm, encoding='utf8', **kwargs):
    with open(file, encoding=encoding) as fh:
        for line in fh:
            if not line.strip():
                continue
            sentence = re.split(r'\W+', line.strip())
            new_sent = correct_sentence(sentence, lm, **kwargs)
            yield ' '.join(new_sent) + '\n'


def spellcorrect(model, files, outdir='.', **kwargs):
    os.makedirs(outdir, exist_ok=True)
    model = load_model(model)
    for file in files:
        basename = os.path.basename(file)
        with open(os.path.join(outdir, basename), 'w', encoding='utf8') as out:
            for upd_line in correct_file(file, model, **kwargs):
                out.write(upd_line)


def main():
    import argparse

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@!')
    parser.add_argument('model',
                        help='Spelling model to use')
    parser.add_argument('files', nargs='+',
                        help='List files or directories to spell-correct.')
    parser.add_argument('--outdir', default='.',
                        help='Directory to place output files.')
    args = parser.parse_args()
    spellcorrect(args.model, args.files, args.outdir)


if __name__ == '__main__':
    main()
