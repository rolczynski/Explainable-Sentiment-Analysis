import os
import logging
import pathlib
from functools import partial

from sklearn.metrics import confusion_matrix
import numpy as np
import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import Example
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
    nlp.tokenizer.basic_tokenizer.never_split = {nlp.tokenizer.mask_token}
    tokens = list(example.text_tokens)  # Break the reference to the object.
    if index is None:
        best = 0    # The patterns have already been sorted.
        chosen_pattern = example.review.patterns[best]
        index = np.argmax(chosen_pattern.weights)
    tokens[index] = nlp.tokenizer.mask_token
    new_example = Example(text=' '.join(tokens), aspect=example.aspect)
    return new_example


def ground_truth_examples(domain: str):
    dataset = absa.load_examples('semeval', domain, test=False)
    for i, example in enumerate(dataset):
        yield i, -1, example    # Predict without a mask.
        [tokenized_example] = nlp.tokenize([example])
        n = len(tokenized_example.text_tokens)
        for index in range(n):
            new_example = mask(tokenized_example, index)
            yield i, index, new_example


@memory.cache
def ground_truth(domain: str) -> np.ndarray:
    results = []
    examples = ground_truth_examples(domain)
    batches = absa.utils.batches(examples, batch_size=32)
    for batch in batches:
        indices, mask_index, batch_examples = zip(*batch)
        predictions = nlp.transform(batch_examples)
        y_hat = [e.sentiment.value for e in predictions]
        results.extend(zip(indices, mask_index, y_hat))
    return np.array(results)


@memory.cache
def evaluate(recognizer: absa.PatternRecognizer, domain: str) -> np.ndarray:
    nlp.professor.pattern_recognizer = recognizer
    results = []
    dataset = absa.load_examples('semeval', domain, test=False)
    batches = absa.utils.batches(dataset, batch_size=32)
    for batch in batches:
        predictions = nlp.transform(batch)
        predictions = list(predictions)  # Needed to the comparison.

        new_batch = [mask(prediction) for prediction in predictions]
        new_predictions = nlp.transform(new_batch)

        y_ref = [e.sentiment.value for e in predictions]
        y_new = [e.sentiment.value for e in new_predictions]
        results.extend(zip(y_ref, y_new))
    # It is not a generator because we cache function results.
    return np.array(results)


if __name__ == '__main__':
    np.random.seed(0)
    os.chdir(HERE)
    utils.setup_logger(HERE / 'logs' / 'recognition-key-token.log')
    accuracy = lambda m: np.diagonal(m).sum() / m.sum()

    logger.info('Begin Evaluation: the Key Token Recognition')
    for dataset_domain in ['restaurant', 'laptop']:
        lm_name = PRETRAINED_LM_NAMES[dataset_domain]
        nlp = absa.load(lm_name)

        reference = ground_truth(dataset_domain)

        random = extension.RandomPatternRecognizer()
        attention = extension.AttentionPatternRecognizer(max_patterns=5)
        gradient = extension.GradientPatternRecognizer(max_patterns=5)
        basic = absa.BasicPatternRecognizer(max_patterns=5)
        recognizers = [random, attention, gradient, basic]

        evaluate_domain = partial(evaluate, domain=dataset_domain)
        results = map(evaluate_domain, recognizers)
        matrices = map(confusion_matrix, results)
        accuracies = map(accuracy, matrices)

        list(accuracies)
        logger.info(f'{dataset_domain.upper()} DOMAIN\n')
