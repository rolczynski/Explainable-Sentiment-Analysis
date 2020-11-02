import logging
import pathlib
from typing import Dict, Iterable, List
from typing import Tuple
from collections import defaultdict

import numpy as np
import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import PatternRecognizer
from aspect_based_sentiment_analysis import PredictedExample
from aspect_based_sentiment_analysis import Pipeline
from joblib import Memory

from . import extension
from . import utils
from . import plots
from .recognition_key_token import mask_tokens
from .recognition_key_token import key_token_mask
from .recognition_key_token_pair import key_token_pair_mask

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
    results = []
    dataset = absa.load_examples('semeval', domain, test=True)
    batches = absa.utils.batches(dataset, batch_size=32)
    for batch_i, batch in enumerate(batches):
        predictions = nlp.transform(batch)
        predictions = list(predictions)  # Keep in memory.

        i = np.arange(len(predictions)) + batch_i * 32
        s = [e.sentiment.value for e in predictions]
        k = np.zeros_like(i)
        results.extend(zip(i, s, s, k))

        masked_examples = mask_examples(nlp, predictions, max_k)
        masked_batches = absa.utils.batches(masked_examples, batch_size=32)
        for masked_batch in masked_batches:
            i, s, k, masked_batch_examples = zip(*masked_batch)
            i = np.array(i) + batch_i * 32
            masked_predictions = nlp.transform(masked_batch_examples)
            new_s = [e.sentiment.value for e in masked_predictions]
            results.extend(zip(i, s, new_s, k))
    return np.array(results)


def evaluate(
        recognizer: PatternRecognizer,
        domain: str,
        name: str,
        max_k: int
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    partial_results = _evaluate(recognizer, domain, name, max_k)
    grouped = defaultdict(list)
    for i, s, s_new, k in partial_results:
        if s != s_new:
            grouped[i].append([i, s, s_new, k])

    minimal_key_sets = []
    for i, group in grouped.items():
        *_, k = zip(*group)
        min_k_index = np.argmin(k)
        minimal_key_sets.append(group[min_k_index])
    i, s, s_new, k = np.array(minimal_key_sets).T
    return np.bincount(k)


def experiment(models: Dict[str, str], max_k: int = 5):
    utils.setup_logger(HERE / 'logs' / 'recognition-minimal-key-set.log')
    logger.info('Begin Evaluation: the Minimal Key Set Recognition')

    for domain in ['restaurant', 'laptop']:
        name = models[domain]
        nlp = absa.load(name)

        token_mask = key_token_mask(nlp, domain, is_test=True)
        max_score_key_token = sum(token_mask) / len(token_mask)
        pair_mask = key_token_pair_mask(nlp, domain)
        max_score_key_token_pair = sum(pair_mask) / len(pair_mask)

        random = extension.RandomPatternRecognizer()
        attention = extension.AttentionPatternRecognizer(max_patterns=5)
        gradient = extension.GradientPatternRecognizer(max_patterns=5)
        basic = absa.BasicPatternRecognizer(max_patterns=5)
        recognizers = [basic, random, attention, gradient, ]

        results = []
        for recognizer in recognizers:
            nlp.professor.pattern_recognizer = recognizer
            result = evaluate(nlp, domain, repr(recognizer), max_k)
            results.append(result)

        logger.info(
            f'{domain.upper()} DOMAIN\n')
