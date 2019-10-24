class NoisyChannel:

    def __init__(self):
        self.singles = None
        self.doubles = None

    def transform(self, term):
        for i, letter in enumerate(term):
            if letter in self.singles:
                for cand in self.singles[letter]:
                    yield term[:i] + cand + term[i + 1:]
            double = term[i - 1:i + 1]
            if i > 0 and double in self.doubles:
                for cand in self.doubles[double]:
                    yield term[:i - 1] + cand + term[i + 1:]

    # def transform(self, term, max_changes=None):
    #     terms = {term: 0}
    #     for obs, cands in self.matrix:
    #         if obs not in term:
    #             continue
    #         for t, changes in list(terms.items()):
    #             for cand in cands:
    #                 new_term = t.replace(obs, cand)
    #                 if new_term in terms:
    #                     continue
    #                 if max_changes and changes + 1 <= max_changes:
    #                     terms[new_term] = changes + 1
    #                     yield new_term, changes + 1


class OcrNoisyChannel(NoisyChannel):

    def __init__(self):
        super().__init__()
        self.singles = {
            'I': ('l',),
            'i': ('l',),
            'l': ('i', 'I'),
            'y': ('v',),
            'v': ('y',),
        }
        self.doubles = {
            'ii': ('n',)
        }
