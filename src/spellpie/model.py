import os
import pickle
from collections import defaultdict
from nltk.tokenize import word_tokenize

from spellpie.db_handler import DbHandler


class SpellingModel(object):
    def __init__(self, ignore_case=True):
        self.ignore_case = ignore_case
        self.words = defaultdict(int)
        self.prebuilt = False
        self.model = defaultdict(int)

    def import_wordlist_from_file(self, fp):
        with open(fp, 'r') as to_read:
            for line in to_read:
                self.words[self._case(line.strip())] = 1
        self.prebuilt = True

    def import_wordlist(self, table=None):
        """Import built-in word lists"""
        for word in DbHandler().load_wordlist(table):
            self.words[self._case(word)] = 1
        self.prebuilt = True

    def train(self, training_dir):
        for fn in os.listdir(training_dir):
            with open(os.path.join(training_dir, fn), 'r') as fh:
                for line in fh:
                    for token in word_tokenize(line):
                        token = self._case(token)
                        if not self.prebuilt or token in self.words:
                            self.words[token] += 1
        self._calculate_model()

    def _calculate_model(self):
        total = sum(self.words.values())
        for word in self.words:
            self.model[word] = float(self.words[word]) / total

    def _case(self, word):
        word = word.strip()
        return word.lower() if self.ignore_case else word

    def get_probability(self, word):
        if not self.model:
            raise ValueError('Missing model')
        return self.model[self._case(word)]

    def get_best_candidate(self, word_candidates):
        return max((self.get_probability(w), w) for w in word_candidates)[1]

    def in_model(self, word):
        return word in self.model

    def export_model(self, fp, readable=False):
        if readable:
            raise ValueError('Not yet supported.')
        else:
            with open(fp, 'wb') as fh:
                pickle.dump(self.model, fh, pickle.HIGHEST_PROTOCOL)

    def import_model(self, fp, readable=False):
        if readable:
            raise ValueError('Not yet supported.')
        else:
            with open(fp, 'rb') as fh:
                self.model = pickle.load(fh)
