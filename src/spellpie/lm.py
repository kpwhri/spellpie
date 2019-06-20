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
                bigrams[(words[i-1], words[i])] += 1
            if i > 1:
                trigrams[(words[i-2], words[i-1], words[i])] += 1


class TrigramLanguageModel:

    def __init__(self, unigram, bigram, trigram):
        self.unigram = SmoothedLanguageModel(unigram)
        self.bigram = SmoothedLanguageModel(bigram)
        self.trigram = SmoothedLanguageModel(trigram)

    def generate_candidates(self, word):
        self._edits(word)

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


class SmoothedLanguageModel:

    def __init__(self, d):
        self.data = {}
        denom = sum(d.values()) + len(d) + 1  # +1 smoothing
        self.smoothed_prob = math.log(1) - math.log(denom)
        for word, freq in d.items():
            self.data[word] = math.log(freq + 1) - math.log(denom)

    def __getitem__(self, item):
        try:
            return self.data[item]
        except KeyError:
            return self.smoothed_prob


