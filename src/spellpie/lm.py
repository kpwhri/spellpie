import re
import statistics
from collections import defaultdict

import math

SMALL_AMOUNT = 0.1
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

    def smoothed_prob(self, item):
        return self._get_lm(item).smoothed_prob

    def borrowed_prob(self, item):
        return self._get_lm(item).borrowed_prob

    def generate_candidates(self, word, require_word_exists=True):
        return set(self.edits(word, require_word_exists=require_word_exists))

    def _edits(self, word):
        """All edits that are one edit away from `word`."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits(self, word, require_word_exists=True):
        """All edits that are two edits away from `word`."""
        for e1 in self._edits(word):
            if require_word_exists:
                if e1 in self.unigram:
                    yield e1, 1
            else:
                yield e1, 1
            for e2 in self._edits(e1):
                if require_word_exists:
                    if e2 in self.unigram:
                        yield e2, 2
                else:
                    yield e2, 2

    def _get_lm(self, item):
        if isinstance(item, tuple):
            if len(item) == 1:
                return self.unigram
            if len(item) == 2:
                return self.bigram
            if len(item) == 3:
                return self.trigram
        return self.unigram

    def __getitem__(self, item):
        return self._get_lm(item)[item]

    def __contains__(self, item):
        return item in self._get_lm(item)


class SmoothedLanguageModel:

    def __init__(self, d, rate=0.5):
        self.data = {}
        denom = sum(d.values()) + len(d) + rate  # +1 smoothing
        self.smoothed_prob = math.log(rate) - math.log(denom)
        for word, freq in d.items():
            self.data[word] = math.log(freq + 0.5) - math.log(denom)
        self.borrowed_prob = statistics.mean(self.data.values())

    def __getitem__(self, item):
        try:
            return self.data[item]
        except KeyError:
            return self.smoothed_prob

    def __contains__(self, item):
        return item in self.data


class History:

    def __init__(self, lm):
        self.history = defaultdict(dict)  # index is state, list of ([ngram], score)
        self.index = 0
        self.lm = lm

    def append(self, item, prob, index=None):
        if not index:
            index = self.index
        self.history[index][item] = prob

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
        if ci == 0:
            candidate_prob += first_value_increment  # prefer the current word to equal probability options
        if not history:  # populate history
            if history_index == 0:  # initialization
                history.append((candidate,), candidate_prob, index=history_index)
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
        word = word.lower()
        calculate_next_step([(word, 0)] + list(lm.generate_candidates(word)), lm, history, penalty=-SMALL_AMOUNT)
        if combine_neighbors and i + 1 < len(sentence):  # candidates from combining words together
            cword = word + sentence[i + 1]
            calculate_next_step(list(lm.generate_candidates(cword)),  # includes cword in output
                                lm, history,
                                first_value_increment=0,
                                penalty=lm.borrowed_prob(cword),  # still count as 2 words
                                history_index=history.index + 1)
        history.next()
    return history


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
