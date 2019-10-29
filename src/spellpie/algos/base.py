class SpellCorrector:

    def __init__(self):
        self.changes = []  # orig_word, newword, context, tag/label
        self.noisy_channel = None

    @property
    def num_corrections(self):
        return len(self.changes)
