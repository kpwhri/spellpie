import pytest

from spellpie.lm import build_spelling_model
from spellpie.ocr_model import ocr_spell_correct_line


@pytest.fixture()
def simple_lm():
    return build_spelling_model([
        'immature granulocytes',
    ])


@pytest.mark.parametrize(('line', ), [
    (r'(ous 0 ==> oes .+ . ale e=', )
])
def test_garbage_pattern(simple_lm, line):
    """These patterns should not be altered"""
    assert ocr_spell_correct_line(simple_lm, line) == line


@pytest.mark.parametrize(('line', 'exp'), [
    (r'immature Granuiocvtes% 0: cl e % BRH', r'immature Granulocytes% 0: cl e % BRH')
])
def test_good_pattern(simple_lm, line, exp):
    assert ocr_spell_correct_line(simple_lm, line) == exp
