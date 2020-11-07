import logging
import pathlib
from typing import Dict, Iterable, List
from typing import Tuple
from collections import defaultdict

import numpy as np
import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import PredictedExample
from aspect_based_sentiment_analysis import Sentiment
from aspect_based_sentiment_analysis import Pipeline
from sklearn.metrics import confusion_matrix
from joblib import Memory

from . import extension
from . import utils
from .recognition_key_token import mask_tokens
from .recognition_key_token import retrieve_labels \
    as key_token_labels
from .recognition_key_token_pair import retrieve_labels \
    as key_token_pair_labels
from .recognition_key_token_triplet import retrieve_negative_labels \
    as key_token_triplet_negative_labels

logger = logging.getLogger('analysis.recognition-minimal-key-set')
HERE = pathlib.Path(__file__).parent
memory = Memory(HERE / 'outputs')


def mask_examples(
        nlp: Pipeline,
        predictions: Iterable[PredictedExample],
        max_k: int
):
    for i, example in enumerate(predictions):
        s = example.sentiment.value
        for k in range(1, max_k+1):
            masked_example = mask_tokens(nlp, example, k=k)
            yield i, s, k, masked_example


@memory.cache(ignore=['nlp'])
# The pattern recognizer name is used to distinguish function calls (caching).
def _evaluate(nlp: Pipeline, domain: str, name: str, max_k: int) -> np.ndarray:
    partial_results = []
    dataset = absa.load_examples('semeval', domain, test=True)
    batches = absa.utils.batches(dataset, batch_size=32)
    for batch_i, batch in enumerate(batches):
        predictions = nlp.transform(batch)
        predictions = list(predictions)  # Keep in memory.

        i = np.arange(len(predictions)) + batch_i * 32
        s = [e.sentiment.value for e in predictions]
        k = np.zeros_like(i)
        partial_results.extend(zip(i, s, s, k))

        masked_examples = mask_examples(nlp, predictions, max_k)
        masked_batches = absa.utils.batches(masked_examples, batch_size=32)
        for masked_batch in masked_batches:
            i, s, k, masked_batch_examples = zip(*masked_batch)
            i = np.array(i) + batch_i * 32
            masked_predictions = nlp.transform(masked_batch_examples)
            new_s = [e.sentiment.value for e in masked_predictions]
            partial_results.extend(zip(i, s, new_s, k))
    return np.array(partial_results)


def retrieve_max_scores(nlp: Pipeline, domain: str) -> np.ndarray:
    max_scores = []
    y_ref, _, mask_1 = key_token_labels(nlp, domain, is_test=True)
    *_, mask_2 = key_token_pair_labels(nlp, domain, parts=10)
    *_, mask_3 = key_token_triplet_negative_labels(nlp, domain, parts=5)

    negative = y_ref == Sentiment.negative
    max_scores.append(sum(mask_1[negative]))
    mask_2 = (mask_2.astype(int) - mask_1.astype(int)).astype(bool)
    max_scores.append(sum(mask_2[negative]))
    max_scores.append(sum(mask_3))  # The mask 3 has only negatives.

    max_scores = np.array(max_scores) / sum(negative)
    max_scores = np.round(max_scores, decimals=3)
    max_scores = np.append(max_scores, 1-sum(max_scores))
    return max_scores


def filter_partial_results(partial_results):
    grouped = defaultdict(list)
    for i, s, s_new, k in partial_results:
        grouped[(i, s)].append([s_new, k])
    results = []
    for (i, s), group in grouped.items():
        s_new, k = zip(*group)
        is_changed = s_new != s
        if any(is_changed):
            i = np.argmax(is_changed)
            record = i, s, s_new[i], k[i]
        else:
            record = i, s, s, max(k)+1
        results.append(record)
    return np.array(results)


def evaluate(
        nlp: Pipeline,
        domain: str,
        name: str,
        max_k: int
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    partial_results = _evaluate(nlp, domain, name, max_k)
    results = filter_partial_results(partial_results)
    i, s, s_new, k = results.T
    matrix = confusion_matrix(s, s_new)

    negative = s == Sentiment.negative
    hist = np.bincount(k[negative])
    max_scores = hist[1:4] / sum(negative)
    max_scores = np.round(max_scores, decimals=3)
    max_scores = np.append(max_scores, 1-sum(max_scores))
    return max_scores, matrix


def experiment(models: Dict[str, str], max_k: int = 5):
    utils.setup_logger(HERE / 'logs' / 'recognition-minimal-key-set.log')
    logger.info('Begin Evaluation: the Minimal Key Set Recognition')

    for domain in ['restaurant', 'laptop']:
        name = models[domain]
        nlp = absa.load(name)
        max_scores = retrieve_max_scores(nlp, domain)

        random = extension.RandomPatternRecognizer()
        attention = extension.AttentionPatternRecognizer(max_patterns=5)
        gradient = extension.GradientPatternRecognizer(max_patterns=5)
        basic = absa.BasicPatternRecognizer(max_patterns=5)
        recognizers = [random, attention, gradient, basic]

        results = []
        for recognizer in recognizers:
            nlp.professor.pattern_recognizer = recognizer
            result = evaluate(nlp, domain, repr(recognizer), max_k)
            results.append(result)

        logger.info(
            f'{domain.upper()} DOMAIN\n'
            f'Max scores: \n{max_scores}\n\n'
            f'Random Pattern Recognizer: \n{results[0][0]}\n'
            f'Confusion Matrix (y_ref, y_new):\n{results[0][1]}\n\n'
            f'Attention Pattern Recognizer: \n{results[1][0]}\n'
            f'Confusion Matrix (y_ref, y_new):\n{results[1][1]}\n\n'
            f'Gradient Pattern Recognizer: \n{results[2][0]}\n'
            f'Confusion Matrix (y_ref, y_new):\n{results[2][1]}\n\n'
            f'Basic Pattern Recognizer: \n{results[3][0]}\n'
            f'Confusion Matrix (y_ref, y_new):\n{results[3][1]}\n\n')
