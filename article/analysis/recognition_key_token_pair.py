import logging
import pathlib
from typing import Dict, List
from typing import Tuple
from collections import defaultdict

import numpy as np
import aspect_based_sentiment_analysis as absa
from sklearn.metrics import confusion_matrix
from aspect_based_sentiment_analysis import PatternRecognizer
from aspect_based_sentiment_analysis import Pipeline
from joblib import Memory

from . import extension
from . import utils
from . import plots
from .recognition_key_token import mask_tokens
from .recognition_key_token import key_token_mask


logger = logging.getLogger('analysis.recognition-key-token-pair')
HERE = pathlib.Path(__file__).parent
memory = Memory(HERE / 'outputs')


def mask_examples(nlp: Pipeline, domain: str, part_parts: Tuple[int, int]):
    dataset = absa.load_examples('semeval', domain, test=True)

    # Filter out examples that contain at least one key token.
    mask = key_token_mask(nlp, domain, is_test=True)
    dataset = [e for is_key_token, e in zip(mask, dataset) if not is_key_token]

    # Split a dataset because it's better to cache more freq.
    part, parts = part_parts
    chunks = utils.split(dataset, n=parts)
    dataset_chunk = chunks[part]

    for i, example in enumerate(dataset_chunk):
        yield i, -1, -1, example    # Predict without a mask.

        [tokenized_example] = nlp.tokenize([example])
        n = len(tokenized_example.text_tokens)

        x, y = np.triu_indices(n, k=1)  # Exclude the diagonal.
        for ij in zip(x, y):
            new_example = mask_tokens(nlp, tokenized_example, indices=ij)
            yield i, *ij, new_example


@memory.cache(ignore=['nlp'])
def _key_token_pair_mask(
        nlp: Pipeline,
        domain: str,
        part_parts: Tuple[int, int]
) -> np.ndarray:
    partial_results = []
    examples = mask_examples(nlp, domain, part_parts)
    batches = absa.utils.batches(examples, batch_size=32)
    for batch in batches:
        indices, *masked_tokens_ij, batch_examples = zip(*batch)
        predictions = nlp.transform(batch_examples)
        y_hat = [e.sentiment.value for e in predictions]
        partial_results.extend(zip(indices, *masked_tokens_ij, y_hat))
    return np.array(partial_results)


def key_token_pair_mask(nlp: Pipeline, domain: str, parts=10) -> np.ndarray:
    d = defaultdict(set)
    for part in range(parts):
        partial_results = _key_token_pair_mask(nlp, domain, (part, parts))
        n = len(d)
        for i, *ij, y_hat in partial_results:
            i += n  # We have parts so the index needs to be shifted.
            d[i].add(y_hat)
    mask = [len(classes) > 1 for classes in d.values()]
    return np.array(mask)


@memory.cache(ignore=['nlp'])
# The pattern recognizer name is used to distinguish function calls (caching).
def _evaluate(nlp: Pipeline, domain: str, name: str) -> np.ndarray:
    results = []
    dataset = absa.load_examples('semeval', domain, test=True)

    # Filter out examples that contain at least one key token.
    mask = key_token_mask(nlp, domain, is_test=True)
    dataset = [e for is_key_token, e in zip(mask, dataset) if not is_key_token]

    batches = absa.utils.batches(dataset, batch_size=32)
    for batch in batches:
        predictions = nlp.transform(batch)
        predictions = list(predictions)  # Keep in memory to append at the end.

        new_batch = [mask_tokens(nlp, prediction, k=2) for prediction in predictions]
        new_predictions = nlp.transform(new_batch)

        y_ref = [e.sentiment.value for e in predictions]
        y_new = [e.sentiment.value for e in new_predictions]
        results.extend(zip(y_ref, y_new))
    # It's not a generator because we cache function results.
    return np.array(results)


def evaluate(
        recognizer: PatternRecognizer,
        domain: str,
        name: str
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    results = _evaluate(recognizer, domain, name)
    y_ref, y_new = results.T
    matrix = confusion_matrix(y_ref, y_new)
    acc = 1 - utils.accuracy(matrix)  # The aim is to change a prediction.
    return acc, matrix, y_ref != y_new


def experiment(models: Dict[str, str], save_confusion_matrix: bool = True):
    utils.setup_logger(HERE / 'logs' / 'recognition-key-token-pair.log')
    logger.info('Begin Evaluation: the Key Token Pair Recognition')

    for domain in ['restaurant', 'laptop']:
        name = models[domain]
        nlp = absa.load(name)

        is_pair = key_token_pair_mask(nlp, domain)
        max_score = sum(is_pair) / len(is_pair)

        random = extension.RandomPatternRecognizer()
        attention = extension.AttentionPatternRecognizer(max_patterns=5)
        gradient = extension.GradientPatternRecognizer(max_patterns=5)
        basic = absa.BasicPatternRecognizer(max_patterns=5)
        recognizers = [random, attention, gradient, basic]

        results = []
        for recognizer in recognizers:
            nlp.professor.pattern_recognizer = recognizer
            result = evaluate(nlp, domain, repr(recognizer))
            results.append(result)

        if save_confusion_matrix:
            plots.confusion_matrix.save_figure(results, domain)

        logger.info(
            f'{domain.upper()} DOMAIN\n'
            f'Examples that have at least one key token pair: {max_score:.4f}\n\n'
            f'Random Pattern Recognizer\n'
            f'Acc.: {results[0][0]/max_score:.4f} ({results[0][0]:.4f})\n'
            f'Confusion Matrix (y_ref, y_new):\n{results[0][1]}\n\n'
            f'Attention Pattern Recognizer\n'
            f'Acc.: {results[1][0]/max_score:.4f} ({results[1][0]:.4f})\n'
            f'Confusion Matrix (y_ref, y_new):\n{results[1][1]}\n\n'
            f'Gradient Pattern Recognizer\n'
            f'Acc.: {results[2][0]/max_score:.4f} ({results[2][0]:.4f})\n'
            f'Confusion Matrix (y_ref, y_new):\n{results[2][1]}\n\n'
            f'Basic Pattern Recognizer\n'
            f'Acc.: {results[3][0]/max_score:.4f} ({results[3][0]:.4f})\n'
            f'Confusion Matrix (y_ref, y_new):\n{results[3][1]}\n\n')
