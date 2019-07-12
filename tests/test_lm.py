import pytest

from spellpie.lm import build_spelling_model, viterbi


@pytest.fixture()
def simple_lm():
    return build_spelling_model(['I like cheese', 'I like potatoes', 'I eat meat'])


@pytest.mark.parametrize(('input_sent', 'expected_sent'), [
    ('I eat potatos', 'i eat potatoes'),
])
def test_simple_lm(simple_lm, input_sent, expected_sent):
    output_sent = viterbi(input_sent.split(), simple_lm)
    assert output_sent[0][0] == expected_sent.split()
