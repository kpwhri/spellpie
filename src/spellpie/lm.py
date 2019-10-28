import os
import pickle
import json
import re
import statistics
from collections import defaultdict

import math

from spellpie.noise.ocr_channel import NoisyChannel

SMALL_AMOUNT = 0.1
COMBINATION_PENALTY = 6.2  # between 3-6


def build_spelling_model(it, split_pat=r'[^a-z]+'):
    """

    :param split_pat: pattern to separate words in given sentences
    :param it: iterator of text segments (sentences, documents, etc.)
    :return:
    """
    unigrams = defaultdict(int)
    bigrams = defaultdict(int)
    trigrams = defaultdict(int)
    pat = re.compile(split_pat)
    for sentence in it:
        words = pat.split(sentence.lower())
        for i in range(len(words)):
            unigrams[words[i]] += 1
            if i > 0:
                bigrams[(words[i - 1], words[i])] += 1
            if i > 1:
                trigrams[(words[i - 2], words[i - 1], words[i])] += 1
    lm = TrigramLanguageModel(unigrams, bigrams, trigrams)
    return lm


class TrigramLanguageModel:

    def __init__(self, unigram=None, bigram=None, trigram=None):
        """Frequencies"""
        self.unigram = SmoothedLanguageModel(unigram)
        self.bigram = SmoothedLanguageModel(bigram)
        self.trigram = SmoothedLanguageModel(trigram)

    def smoothed_prob(self, item):
        return self._get_lm(item).smoothed_prob

    def borrowed_prob(self, item):
        return self._get_lm(item).borrowed_prob

    def generate_candidates(self, word, require_word_exists=True,
                            noisy_channel: NoisyChannel = None, n_edits=2):
        edits = set()
        if noisy_channel:
            for w in noisy_channel.transform(word):
                edits |= set(self.edits(w, require_word_exists=require_word_exists, n_edits=n_edits))
        edits |= set(self.edits(word, require_word_exists=require_word_exists, n_edits=n_edits))
        return edits

    def _edits(self, word):
        """All edits that are one edit away from `word`."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits(self, word, require_word_exists=True, n_edits=2):
        """All edits that are two edits away from `word`."""
        if len(word) <= 2:
            return
        for e1 in self._edits(word):
            if require_word_exists:
                if e1 in self.unigram:
                    yield e1, 1
            else:
                yield e1, 1
            if len(word) <= 3 or n_edits < 2:
                continue
            for e2 in self._edits(e1):
                if require_word_exists:
                    if e2 in self.unigram:
                        yield e2, 2
                else:
                    yield e2, 2

    def _get_lm(self, item):
        if isinstance(item, tuple):
            if len(item) == 1:
                raise ValueError('Unigram value should not be a tuple.')
            if len(item) == 2:
                return self.bigram
            if len(item) == 3:
                return self.trigram
        return self.unigram

    def sum(self, *items):
        return sum(self[item] for item in items)

    def __getitem__(self, item):
        return self._get_lm(item)[item]

    def __contains__(self, item):
        return item in self._get_lm(item)

    def tojson(self, path=None):
        def pack_key(key):
            if isinstance(key, tuple):
                return '_'.join(key)
            return key

        data = {
            f'{i}': {'data': {pack_key(d): v for d, v in x.data.items()},
                     'smoothed_prob': x.smoothed_prob}
            for i, x in enumerate((self.unigram, self.bigram, self.trigram), start=1)
        }
        if path:
            with open(path, 'w') as fh:
                json.dump(data, fh)
        else:
            return json.dumps(data)

    def topickle(self, path=None):
        data = {
            f'{i}': {'data': x.data, 'smoothed_prob': x.smoothed_prob}
            for i, x in enumerate((self.unigram, self.bigram, self.trigram), start=1)
        }
        if path:
            with open(path, 'wb') as fh:
                pickle.dump(data, fh)
        else:
            return pickle.dumps(data)

    @classmethod
    def fromjson(cls, path):
        def unpack_key(key):
            if '_' in key:
                return tuple('_'.split(key))
            return key

        with open(path, 'r') as fh:
            data = json.load(fh)
        for ngram in data:
            data[ngram]['data'] = {unpack_key(k): v for k, v in data[ngram]['data'].items()}
        return cls.fromdict(data)

    @classmethod
    def fromdict(cls, data):
        m = cls()
        m.unigram = SmoothedLanguageModel.fromdict(data['1'])
        m.bigram = SmoothedLanguageModel.fromdict(data['2'])
        m.trigram = SmoothedLanguageModel.fromdict(data['3'])
        return m

    @classmethod
    def frompickle(cls, path):
        with open(path, 'rb') as fh:
            data = pickle.load(fh)
        return cls.fromdict(data)


class SmoothedLanguageModel:

    def __init__(self, d, rate=0.5):
        self.data = {}
        self.smoothed_prob = 0
        self.borrowed_prob = 0
        if d:
            denom = sum(d.values()) + len(d) + 1  # +1 smoothing
            self.smoothed_prob = math.log(rate) - math.log(denom)
            for word, freq in d.items():
                self.data[word] = math.log(freq + rate) - math.log(denom)
            self.borrowed_prob = statistics.mean(self.data.values())

    def __getitem__(self, item):
        try:
            return self.data[item]
        except KeyError:
            return self.smoothed_prob

    def __contains__(self, item):
        return item in self.data

    @classmethod
    def fromdict(cls, data):
        m = cls(None)
        m.data = data['data']
        m.smoothed_prob = data['smoothed_prob']
        return m


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


def calculate_next_step(wordlist, lm, history: History, history_index=None,
                        first_value_increment=SMALL_AMOUNT,
                        penalty=0.0,
                        require_word_exists=False):
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


def viterbi(sentence, lm, combine_neighbors=True, penalty=COMBINATION_PENALTY):
    history = History(lm)
    for i, word in enumerate(sentence):
        word = word.lower().strip()
        if not word:
            continue
        if word in lm:
            penalty = lm.borrowed_prob(word)
        else:
            penalty = - SMALL_AMOUNT
        calculate_next_step([(word, 0)] + list(lm.generate_candidates(word)), lm, history, penalty=penalty)
        if combine_neighbors and i + 1 < len(sentence):  # candidates from combining words together
            cword = word + sentence[i + 1]
            calculate_next_step(list(lm.generate_candidates(cword)),  # includes cword in output
                                lm, history,
                                first_value_increment=0,
                                penalty=lm.borrowed_prob(cword) + penalty,  # still count as 2 words
                                history_index=history.index + 1)
        history.next()
    return history


def correct_sentence(sentence, lm, **kwargs):
    return viterbi(sentence, lm, **kwargs).best_candidate()


def correct_file(file, lm, encoding='utf8', **kwargs):
    with open(file, encoding=encoding) as fh:
        for line in fh:
            if not line.strip():
                continue
            sentence = re.split(r'\W+', line.strip())
            new_sent = correct_sentence(sentence, lm, **kwargs)
            yield ' '.join(new_sent) + '\n'


def load_model(model_file):
    if model_file.endswith('.pkl'):
        return TrigramLanguageModel.frompickle(model_file)
    elif model_file.endswith('.json'):
        return TrigramLanguageModel.fromjson(model_file)


def determine_penalty_cutoff(sample_sentences=None, pass_sentences=None, fail_sentences=None,
                             start=2, end=8, incr=0.1):
    sample_sentences = ['I like cheese', 'I like potatoes', 'I eat meat', 'I eat tomatoes']
    input_sentences = ['I like cheese', 'I eat potatos', 'pot atoes', 'drink soda']
    output_sentences = ['I like cheese', 'I eat potatoes', 'potatoes', 'drink soda']
    lm = build_spelling_model(sample_sentences)
    val = start
    passed = defaultdict(list)
    while val < end:
        n_passed = 0
        results = []
        for i_s, o_s in zip(input_sentences, output_sentences):
            res = viterbi(i_s.split(), lm, penalty=val)[0][0]
            results.append(res)
            if res == o_s.lower().split():
                n_passed += 1
        print(val, results)
        if n_passed > 0:
            passed[n_passed].append(val)
        val += incr
        if passed and n_passed == 0:
            break
    m = max(passed.keys())
    print(m, passed[m])


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
