from dataclasses import dataclass
from typing import List

from spellpie.algos.store import Word


@dataclass
class Change:
    orig_word: str
    new_word: str
    index: int  # character index (in line)
    line_index: int  # character index of start of line
    context: str
    tag: str

    @property
    def header(self):
        return 'orig_word', 'new_word', 'index', 'context', 'tag'

    @property
    def absolute_index(self):
        return self.index + self.line_index

    @property
    def clean_context(self):
        return ' '.join(self.context.split())

    def to_list(self):
        return (self.orig_word, self.new_word, self.absolute_index,
                self.clean_context, self.tag)


class SpellCorrector:

    def __init__(self):
        self.changes: List[Change] = []
        self.noisy_channel = None

    @property
    def num_corrections(self):
        return len(self.changes)

    def add_change(self, word: Word, line, newword: str, newline,
                   tag=None, line_idx=0):
        self.changes.append(Change(
            orig_word=word.word,
            new_word=newword,
            index=word.start,
            line_index=line_idx + len(newline),
            context=line[max(word.start - 100, 0):word.end + 100],
            tag=tag
        ))

    def export_changes(self, include_header=True):
        for i, change in enumerate(self.changes):
            if i == 0 and include_header:
                yield change.header
            yield change.to_list()
