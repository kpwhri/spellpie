import re
from collections import defaultdict

import math

SMALL_AMOUNT = 0.000001
COMBINATION_PENALTY = 6.2  # between 3-6


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

    def __init__(self, unigram, bigram, trigram):
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


def calculate_next_step(wordlist, lm, history,
                        first_value_increment=SMALL_AMOUNT,
                        penalty=0.0):
    new_history = []
    for ci, candidate in enumerate(wordlist):
        if not candidate:
            continue
        candidate_prob = lm[candidate] - penalty
        if ci == 0:
            candidate_prob += SMALL_AMOUNT  # prefer the current word to equal probability options
        if not history:  # populate history
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
    return new_history


def viterbi(sentence, lm, combine_neighbors=True, penalty=COMBINATION_PENALTY):
    history = []
    future_history = defaultdict(list)
    for i, word in enumerate(sentence):
        new_history = calculate_next_step([word] + list(lm.generate_candidates(word)), lm, history)
        if combine_neighbors and i + 1 < len(sentence):  # candidates from combining words together
            cword = word + sentence[i + 1]
            future_history[i + 2] = calculate_next_step([cword] + list(lm.generate_candidates(cword)), lm, history,
                                                        first_value_increment=0, penalty=penalty)
        history = new_history
        history += future_history[i+1]
    return sorted(history, key=lambda x: -x[1])


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
