import pytest

from spellpie.lm import build_spelling_model, viterbi


@pytest.fixture()
def simple_lm():
    return build_spelling_model(['I like cheese', 'I like potatoes', 'I eat meat', 'I eat tomatoes'])


def get_probability(history, actual, expected):
    return f'Actual: {actual}, {history[actual]}; Expected: {expected}, {history[expected]}'


def compare(history, expected_sent):
    expected = tuple(expected_sent.split())
    actual = history.best_candidate()
    assert actual == expected, get_probability(history, actual, expected)


@pytest.mark.parametrize(('input_sent', 'expected_sent'), [
    ('I eat potatos', 'i eat potatoes'),
])
def test_simple_lm(simple_lm, input_sent, expected_sent):
    history = viterbi(input_sent.split(), simple_lm)
    compare(history, expected_sent)


@pytest.mark.parametrize(('input_sent', 'expected_sent'), [
    ('eatpotatoes', 'eatpotatoes'),
    ('drink soda', 'drink soda'),
])
def test_simple_lm_prefer_input(simple_lm, input_sent, expected_sent):
    history = viterbi(input_sent.split(), simple_lm)
    compare(history, expected_sent)


@pytest.mark.parametrize(('input_sent', 'expected_sent'), [
    ('pot atoes', 'potatoes'),
    ('eat chee se eat', 'eat cheese eat'),
])
def test_simple_lm_separated(simple_lm, input_sent, expected_sent):
    history = viterbi(input_sent.split(), simple_lm)
    compare(history, expected_sent)
