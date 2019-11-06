from spellpie.algos.store.word import Word


class Words:

    def __init__(self):
        self.words = []
        self.curr = None

    def append(self, obj):
        self.words.append(obj)

    def __iter__(self):
        for i, word in enumerate(self.words):
            self.curr = i
            yield word
        self.curr = None

    def next_word(self):
        """get next word while iterating through Words object"""
        if self.curr is None:
            raise ValueError('No current word. Use for loop to create this context.')
        if self.curr >= len(self.words) - 1:
            return Word('<EOS>', None, None, None, None)
        return self.words[self.curr + 1]

    def prev_word(self):
        """get previous word while iterating through Words object"""
        if self.curr is None:
            raise ValueError('No current word. Use for loop to create this context.')
        if self.curr <= 0:
            return Word('<BOS>', None, None, None, None)
        return self.words[self.curr - 1]

    def next_next_word(self):
        """get next word while iterating through Words object"""
        if self.curr is None:
            raise ValueError('No current word. Use for loop to create this context.')
        if self.curr >= len(self.words) - 2:
            return Word('<EOS>', None, None, None, None)
        return self.words[self.curr + 2]

    def prev_prev_word(self):
        """get previous word while iterating through Words object"""
        if self.curr is None:
            raise ValueError('No current word. Use for loop to create this context.')
        if self.curr <= 0:
            return Word('<BOS>', None, None, None, None)
        return self.words[self.curr - 2]
