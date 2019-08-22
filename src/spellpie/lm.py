import pickle
import json
import re
from collections import defaultdict

import math


def build_spelling_model(it):
    """

    :param it: iterator of text segments (sentences, documents, etc.)
    :return:
    """
    unigrams = defaultdict(int)
    bigrams = defaultdict(int)
    trigrams = defaultdict(int)
    for sentence in it:
        words = re.sub(r'[\W_]+', ' ', sentence.lower()).split()
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

    def generate_candidates(self, word):
        return self._edits(word)

    def _edits(self, word):
        """All edits that are one edit away from `word`."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits(self, word):
        """All edits that are two edits away from `word`."""
        return (e2 for e1 in self._edits(word) for e2 in self._edits(e1))

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) == 1:
                return self.unigram[item]
            if len(item) == 2:
                return self.bigram[item]
            if len(item) == 3:
                return self.trigram[item]
        return self.unigram[item]

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
    def fromjson(cls, fh):
        def unpack_key(key):
            if '_' in key:
                return tuple('_'.split(key))
            return key

        with open(fh, 'r') as fh:
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
    def frompickle(cls, fh):
        with open(fh, 'rb') as fh:
            data = pickle.load(fh)
        return cls.fromdict(data)


class SmoothedLanguageModel:

    def __init__(self, d):
        self.data = {}
        self.smoothed_prob = 0
        if d:
            denom = sum(d.values()) + len(d) + 1  # +1 smoothing
            self.smoothed_prob = math.log(1) - math.log(denom)
            for word, freq in d.items():
                self.data[word] = math.log(freq + 1) - math.log(denom)

    def __getitem__(self, item):
        try:
            return self.data[item]
        except KeyError:
            return self.smoothed_prob

    @classmethod
    def fromdict(cls, data):
        m = cls(None)
        m.data = data['data']
        m.smoothed_prob = data['smoothed_prob']
        return m


def viterbi(sentence, lm):
    history = []
    for i, word in enumerate(sentence):
        new_history = []
        for candidate in lm.generate_candidates(word) | {word}:
            if not candidate:
                continue
            candidate_prob = lm[candidate]
            if i == 0:  # populate history
                new_history.append(([candidate], candidate_prob))
            else:
                best_path = None
                best_prob = 0
                for curr_path, prob in history:
                    curr_prob = prob + lm[(curr_path[-1], candidate)] + candidate_prob
                    if not best_path or curr_prob > best_prob:
                        best_path = curr_path
                        best_prob = curr_prob
                new_history.append((best_path + [candidate], best_prob))
        history = new_history
    return sorted(history, key=lambda x: -x[1])
