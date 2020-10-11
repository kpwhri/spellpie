# spellpie
Exploratory spell correction for post-processing optical character recognition.

## Introduction
Probabilistic spell correction was used in order to improve the quality of the scanned records. The current version relies on a Laplacian smoothed language model (i.e., +1 smoothing) and incorporates unigram, bigram, and trigrams involving the target word. Due to the time complexity of generating potential candidate spellings, only a single pass is used.

The intention will be to rely on two passes:

1. Generate spell corrections for the corpus as well as a dataset of changes that were applied
		a. This will be considered part of the corpus
		b. The changes can be reviewed (as well as any other spelling errors encountered) and the algorithm will be updated and re-run
2. Re-generate dataset using improved model (time permitting)

This be aware that this takes a while and so is really only practical on a small number of records.

## How it Works
Train a language model on text similar to the target OCR'd text.

1. Tokenization is done at the word-level, splitting on all non-letters (including unicode)
2. If a word containing > 3 characters has not been encountered before (i.e., not in dictionary), it is a candidate for spell correction
3. Generate potential spell corrections using edit distance of 2, in addition to a custom-developed noisy channel model for OCR
4. Select the highest probability candidate (defaulting to prior token)

## Post-processing

Generate a table with the transformations, and see if there are any commons words/phrases that have been wrongly replaced. This can also be used to automatically apply the most important transformations only.
