from spellpie.ocr_model import ocr_spell_correct_line


def test_garbage_pattern():
    """These patterns should not be altered"""
    line = r'(ous 0 ==> oes .+ . ale e='
    assert ocr_spell_correct_line(line) == line
