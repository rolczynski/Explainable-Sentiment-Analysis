import os
import logging
import pathlib
import itertools
from typing import List
from typing import Tuple
from collections import defaultdict

import numpy as np
import aspect_based_sentiment_analysis as absa
from sklearn.metrics import confusion_matrix
from aspect_based_sentiment_analysis import Example
from aspect_based_sentiment_analysis import PatternRecognizer
from aspect_based_sentiment_analysis import PredictedExample
from joblib import Memory

import extension
import utils

logger = logging.getLogger('analysis.recognition-key-token')
HERE = pathlib.Path(__file__).parent
memory = Memory(HERE / 'outputs')
PRETRAINED_LM_NAMES = {
    'restaurant': 'absa/classifier-rest-0.2',
    'laptop': 'absa/classifier-lapt-0.2'
}


def mask(example: PredictedExample, index: int = None) -> Example:
    # Make sure that the pipeline `NLP` has prepared the basic tokenizer.
    NLP.tokenizer.basic_tokenizer.never_split = {NLP.tokenizer.mask_token}
    tokens = list(example.text_tokens)  # Break the reference to the object.
    if index is None:
        best = 0    # The patterns have already been sorted.
        chosen_pattern = example.review.patterns[best]
        index = np.argmax(chosen_pattern.weights)
    tokens[index] = NLP.tokenizer.mask_token
    new_example = Example(text=' '.join(tokens), aspect=example.aspect)
    return new_example


def ground_truth_examples(domain: str, test: bool):
    dataset = absa.load_examples('semeval', domain, test)
    for i, example in enumerate(dataset):
        yield i, -1, example    # Predict without a mask.
        [tokenized_example] = NLP.tokenize([example])
        n = len(tokenized_example.text_tokens)
        for index in range(n):
            new_example = mask(tokenized_example, index)
            yield i, index, new_example


@memory.cache
def _ground_truth(domain: str, test: bool) -> np.ndarray:
    results = []
    examples = ground_truth_examples(domain, test)
    batches = absa.utils.batches(examples, batch_size=32)
    for batch in batches:
        indices, mask_index, batch_examples = zip(*batch)
        predictions = NLP.transform(batch_examples)
        y_hat = [e.sentiment.value for e in predictions]
        results.extend(zip(indices, mask_index, y_hat))
    return np.array(results)


def ground_truth(domain: str, test: bool) -> float:
    results = _ground_truth(domain, test)
    d = defaultdict(set)
    for i, mask_i, y_hat in results:
        d[i].add(y_hat)
    is_key_token = [len(classes) > 1 for classes in d.values()]
    return sum(is_key_token) / len(is_key_token)


@memory.cache
def _evaluate(
        recognizer: PatternRecognizer,
        domain: str,
        test: bool) -> np.ndarray:
    # It's not a generator because we cache function results.
    # The global variable `NLP` helps to avoid serializing the complex input.
    NLP.professor.pattern_recognizer = recognizer
    results = []
    dataset = absa.load_examples('semeval', domain, test)
    batches = absa.utils.batches(dataset, batch_size=32)
    for batch in batches:
        predictions = NLP.transform(batch)
        predictions = list(predictions)  # Needed to the comparison.

        new_batch = [mask(prediction) for prediction in predictions]
        new_predictions = NLP.transform(new_batch)

        y_ref = [e.sentiment.value for e in predictions]
        y_new = [e.sentiment.value for e in new_predictions]
        results.extend(zip(y_ref, y_new))
    return np.array(results)


def evaluate(
        recognizer: PatternRecognizer,
        domain: str,
        test: bool) -> List[Tuple[float, np.ndarray]]:
    results = _evaluate(recognizer, domain, test)
    y_ref, y_new = results.T
    matrix = confusion_matrix(y_ref, y_new)
    acc = 1 - accuracy(matrix)  # The aim is to change a prediction.
    return acc, matrix


if __name__ == '__main__':
    np.random.seed(0)
    os.chdir(HERE)
    utils.setup_logger(HERE / 'logs' / 'recognition-key-token.log')
    accuracy = lambda m: np.diagonal(m).sum() / m.sum()

    logger.info('Begin Evaluation: the Key Token Recognition')
    for TEST, DOMAIN in itertools.product([False, True], ['restaurant', 'laptop']):
        NAME = PRETRAINED_LM_NAMES[DOMAIN]
        NLP = absa.load(NAME)
        max_score = ground_truth(DOMAIN, TEST)

        random = extension.RandomPatternRecognizer()
        attention = extension.AttentionPatternRecognizer(max_patterns=5)
        gradient = extension.GradientPatternRecognizer(max_patterns=5)
        basic = absa.BasicPatternRecognizer(max_patterns=5)
        recognizers = [random, attention, gradient, basic]
        RESULTS = [evaluate(recognizer, DOMAIN, TEST) for recognizer in recognizers]

        logger.info(f'{DOMAIN.upper()} {"TEST" if TEST else "TRAIN"} DOMAIN\n'
                    f'Examples that have at least one key token: {max_score:.4f}\n\n'
                    f'Random Pattern Recognizer\n'
                    f'Acc.: {RESULTS[0][0]/max_score:.4f} ({RESULTS[0][0]:.4f})\n'
                    f'Confusion Matrix (y_ref, y_new):\n{RESULTS[0][1]}\n\n'
                    f'Attention Pattern Recognizer\n'
                    f'Acc.: {RESULTS[1][0]/max_score:.4f} ({RESULTS[1][0]:.4f})\n'
                    f'Confusion Matrix (y_ref, y_new):\n{RESULTS[1][1]}\n\n'
                    f'Gradient Pattern Recognizer\n'
                    f'Acc.: {RESULTS[2][0]/max_score:.4f} ({RESULTS[2][0]:.4f})\n'
                    f'Confusion Matrix (y_ref, y_new):\n{RESULTS[2][1]}\n\n'
                    f'Basic Pattern Recognizer\n'
                    f'Acc.: {RESULTS[3][0]/max_score:.4f} ({RESULTS[3][0]:.4f})\n'
                    f'Confusion Matrix (y_ref, y_new):\n{RESULTS[3][1]}\n\n')
