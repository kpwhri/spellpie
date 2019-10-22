"""
OCR model. When doing OCR, much of the incoming data is likely
    to be fairly replete with garbage. This model will limit the
    scope of the OCR (not to the sentence level), and ignore lines
    which appear to lack any content at all (e.g., no in-dictionary
    words).
"""

def ocr_spell_correct_line(line):
    return line
