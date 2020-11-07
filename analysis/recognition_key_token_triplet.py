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
from sklearn.metrics import confusion_matrix

from . import utils
from .recognition_key_token import mask_tokens
from .recognition_key_token import retrieve_labels as key_token_labels
from .recognition_key_token_pair import retrieve_labels as key_token_pair_labels


logger = logging.getLogger('analysis.recognition-key-token-triplet')
HERE = pathlib.Path(__file__).parent
memory = Memory(HERE / 'outputs')


def mask_examples(
        nlp: Pipeline,
        domain: str,
        part_parts: Tuple[int, int]
):
    dataset = absa.load_examples('semeval', domain, test=True)
    # Filter out examples that contain a key token or a pair of key tokens,
    # and that are other than negative.
    y_ref, _, mask_1 = key_token_labels(nlp, domain, is_test=True)
    y_ref, _, mask_2 = key_token_pair_labels(nlp, domain, parts=10)
    mask = ~(mask_1 | mask_2) & y_ref == Sentiment.negative.value
    dataset = [e for e, is_correct in zip(dataset, mask) if is_correct]

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
def _retrieve_negative_labels(
        nlp: Pipeline,
        domain: str,
        part_parts: Tuple[int, int]
) -> np.ndarray:
    partial_results = []
    examples = mask_examples(nlp, domain, part_parts)
    batches = absa.utils.batches(examples, batch_size=32)
    for batch in batches:
        indices, *masked_tokens_ijk, batch_examples = zip(*batch)
        predictions = nlp.transform(batch_examples)
        y_hat = [e.sentiment.value for e in predictions]
        partial_results.extend(zip(indices, *masked_tokens_ijk, y_hat))
    return np.array(partial_results)


def retrieve_negative_labels(
        nlp: Pipeline,
        domain: str,
        parts: int = 5
) -> np.ndarray:
    results = []
    d = defaultdict(list)
    for part in range(parts):
        partial_results = _retrieve_negative_labels(nlp, domain, (part, parts))
        results.extend(partial_results)

        n = len(d)
        for i, *ijk, y_hat in partial_results:
            i += n  # We have parts so the index needs to be shifted.
            d[i].append(y_hat)
    results = np.array(results)

    example_indices, mask_i, mask_j, mask_k, y_new = results.T
    # The index -1 means predictions without masking.
    y_ref = y_new[mask_i == -1]

    y_new = []
    for ref, classes in zip(y_ref, d.values()):
        classes = np.array(classes)
        available = np.where(classes != ref)
        available_classes = classes[available].tolist()

        if not available_classes:
            y_new.append(ref)
            continue

        new = np.bincount(available_classes).argmax()
        y_new.append(new)
    y_new = np.array(y_new)
    mask = y_ref != y_new
    return y_ref, y_new, mask


def experiment(models: Dict[str, str]):
    utils.setup_logger(HERE / 'logs' / 'recognition-key-token-triplet.log')
    logger.info('Begin Evaluation: the Key Token Triplet Recognition '
                '(Predicted Negative Examples)')

    for domain in ['restaurant', 'laptop']:
        name = models[domain]
        nlp = absa.load(name)

        y_ref, y_new, mask = retrieve_negative_labels(nlp, domain)
        max_score_matrix = confusion_matrix(y_ref, y_new)

        # Remember that we process only (predicted) negative examples,
        # therefore, we need to retrieve predicted negative labels.
        _, reference, _ = key_token_labels(nlp, domain, is_test=True)
        negative = reference == Sentiment.negative
        max_score = sum(mask) / sum(negative)

        logger.info(
            f'{domain.upper()} DOMAIN\n'
            f'Examples that have at least one key token triplet: '
            f'{max_score:.4f}\ny_ref means a prediction without a mask, '
            f'and y_new with a triplet mask.\n'
            f'Confusion Matrix (y_ref, y_new):\n{max_score_matrix}\n\n')
