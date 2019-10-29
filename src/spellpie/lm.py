import pickle
import json
import re
import statistics
from collections import defaultdict

import math

from spellpie.noise.ocr_channel import NoisyChannel


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


def load_model(model_file):
    if model_file.endswith('.pkl'):
        return TrigramLanguageModel.frompickle(model_file)
    elif model_file.endswith('.json'):
        return TrigramLanguageModel.fromjson(model_file)
