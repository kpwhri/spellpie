import pytest

from spellpie.lm import build_spelling_model
from spellpie.ocr_model import OcrSpellCorrector


@pytest.fixture()
def simple_lm():
    return build_spelling_model([
        'immature granulocytes',
        'everything is complete',
    ])


@pytest.fixture()
def cspy_ocr():
    return build_spelling_model([
        'terminal ileum: normal',
        'colonoscopy was performed without difficulty',
        'patient tolerated the procedure well',
        'the quality of the bowel prep was good',
        'the quality of the bowel preparation was poor',
        'bowel prep was fair',
        'prior to the procedure',
        'physical was performed',
        'history of present illness',
    ])


@pytest.fixture()
def ocr_spell_corrector():
    return OcrSpellCorrector()


@pytest.mark.parametrize(('line',), [
    (r'(ous 0 ==> oes .+ . ale e=',),
    (r'ip . P9. . . 12.0000 12000, mae on rn ien ani iit mim BRH ..". — comte.',),
])
def test_garbage_pattern(simple_lm, ocr_spell_corrector, line):
    """These patterns should not be altered"""
    assert ocr_spell_corrector.spell_correct_line(simple_lm, line) == line


@pytest.mark.parametrize(('line', 'exp'), [
    (r'immature Granuiocvtes% 0: cl e % BRH',
     r'immature Granulocytes% 0: cl e % BRH'),
    (r'Electronically signed by: Hippocrates Kos, MD on 04/16/-375 0520 Status: Completec',
     r'Electronically signed by: Hippocrates Kos, MD on 04/16/-375 0520 Status: Complete')
])
def test_good_pattern(simple_lm, ocr_spell_corrector, line, exp):
    assert ocr_spell_corrector.spell_correct_line(simple_lm, line) == exp


@pytest.mark.parametrize(('line', 'exp'), [
    (r'DTerminaI leum: Normal', r'terminal ileum: Normal'),
    (r'EDTerminaI leum: Normal', r'terminal ileum: Normal'),  # requires noisy channel model
    (r'Prior to the procedure, a History and Phyéiﬁal wés performed',
     r'Prior to the procedure, a History and Physical was performed'),
])
def test_good_pattern_cspy(cspy_ocr, ocr_spell_corrector, line, exp):
    assert ocr_spell_corrector.spell_correct_line(cspy_ocr, line) == exp
