import logging
import pathlib
from typing import Dict
from typing import Tuple
from collections import defaultdict

import numpy as np
import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import Pipeline
from aspect_based_sentiment_analysis import Sentiment
from joblib import Memory

from . import utils
from .recognition_key_token import mask_tokens
from .recognition_key_token import key_token_mask
from .recognition_key_token_pair import key_token_pair_mask


logger = logging.getLogger('analysis.recognition-key-token-triplet')
HERE = pathlib.Path(__file__).parent
memory = Memory(HERE / 'outputs')


def mask_examples(nlp: Pipeline, domain: str, part_parts: Tuple[int, int], q: int):
    dataset = absa.load_examples('semeval', domain, test=True)

    # Filter out examples that contain a key token or a pair of key tokens.
    mask_1 = key_token_mask(nlp, domain, is_test=True)
    mask_2 = key_token_pair_mask(nlp, domain)
    mask = mask_1 | mask_2
    dataset = [e for e, is_used in zip(dataset, mask) if not is_used]

    dataset = list(filter(lambda e: e.sentiment == Sentiment.negative, dataset))
    text_lengths = [len(e.text) for e in dataset]
    threshold = np.percentile(text_lengths, q)
    dataset = filter(lambda e: len(e.text) <= threshold, dataset)
    dataset = list(dataset)

    # Split a dataset because it's better to cache more freq.
    part, parts = part_parts
    chunks = utils.split(dataset, n=parts)
    dataset_chunk = chunks[part]

    for i, example in enumerate(dataset_chunk):
        yield i, -1, -1, -1, example    # Predict without a mask.

        [tokenized_example] = nlp.tokenize([example])
        n = len(tokenized_example.text_tokens)

        ij = np.zeros(shape=[n, n])
        ij[np.triu_indices(n, k=1)] = 1  # The j shifted by 1 from i.
        ij = ij.reshape([n, n, 1]).astype(bool)

        jk = np.zeros(shape=[n, n])
        jk[np.triu_indices(n, k=1)] = 1  # The k shifted by 1 from j.
        jk = jk.reshape([1, n, n]).astype(bool)

        matrix_ijk = np.where(ij & jk)
        for ijk in zip(*matrix_ijk):
            new_example = mask_tokens(nlp, tokenized_example, indices=ijk)
            yield i, *ijk, new_example


@memory.cache(ignore=['nlp'])
def _key_token_triplet_mask(
        nlp: Pipeline,
        domain: str,
        part_parts: Tuple[int, int],
        q: int
) -> np.ndarray:
    partial_results = []
    examples = mask_examples(nlp, domain, part_parts, q)
    batches = absa.utils.batches(examples, batch_size=32)
    for batch in batches:
        indices, *masked_tokens_ijk, batch_examples = zip(*batch)
        predictions = nlp.transform(batch_examples)
        y_hat = [e.sentiment.value for e in predictions]
        partial_results.extend(zip(indices, *masked_tokens_ijk, y_hat))
    return np.array(partial_results)


def key_token_triplet_mask(
        nlp: Pipeline,
        domain: str,
        parts=5,
        q: int = 90  # The text length percentile threshold.
) -> np.ndarray:
    d = defaultdict(set)
    for part in range(parts):
        partial_results = _key_token_triplet_mask(nlp, domain, (part, parts), q)
        n = len(d)
        for i, *ijk, y_hat in partial_results:
            i += n  # We have parts so the index needs to be shifted.
            d[i].add(y_hat)
    mask = [len(classes) > 1 for classes in d.values()]
    return np.array(mask)


def experiment(models: Dict[str, str]):
    utils.setup_logger(HERE / 'logs' / 'recognition-key-token-pair.log')
    logger.info('Begin Evaluation: the Key Token Pair Recognition')

    for domain in ['restaurant', 'laptop']:
        name = models[domain]
        nlp = absa.load(name)

        is_pair = key_token_triplet_mask(nlp, domain)
        max_score = sum(is_pair) / len(is_pair)

        logger.info(
            f'{domain.upper()} DOMAIN\n'
            f'Examples that have at least one triplet '
            f'of key tokens: {max_score:.4f}\n\n')
