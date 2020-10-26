import os
import logging
import pathlib
from typing import Iterable

import numpy as np
import transformers
import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import LabeledExample
from aspect_based_sentiment_analysis import Sentiment
from joblib import Memory

import utils

logger = logging.getLogger('analysis.aspect_without')
HERE = pathlib.Path(__file__).parent
memory = Memory(HERE / 'outputs')
PRETRAINED_MODEL_NAMES = {
    'restaurant': 'absa/bert-rest-0.2',
    'laptop': 'absa/bert-lapt-0.2'
}


def build_dataset(domain: str, unknown_token: str) -> Iterable[LabeledExample]:
    examples = absa.load_examples('semeval', domain, test=True)

    # Process only positive/negative examples
    condition = lambda e: e.sentiment in [Sentiment.positive, Sentiment.negative]
    dataset = filter(condition, examples)

    # Remove the information about an aspect
    # (change aspect to the special [UNK] token)
    convert = lambda e: LabeledExample(e.text, unknown_token, e.sentiment)
    dataset = map(convert, dataset)

    return dataset


@memory.cache
def evaluate_without_aspect(domain: str) -> np.ndarray:
    name = PRETRAINED_MODEL_NAMES[domain]
    tokenizer = transformers.BertTokenizer.from_pretrained(name)
    dataset = build_dataset(domain, unknown_token=tokenizer.unk_token)
    nlp = absa.load(name)

    metric = absa.training.ConfusionMatrix(num_classes=3)
    confusion_matrix = nlp.evaluate(dataset, metric, batch_size=32)
    confusion_matrix = confusion_matrix.numpy()
    return confusion_matrix


if __name__ == '__main__':
    os.chdir(HERE)
    utils.setup_logger(HERE / 'logs' / 'aspect-without.log')

    logger.info('Begin Evaluation: Predict With the Masked Aspect')
    for dataset_domain in ['restaurant', 'laptop']:
        matrix = evaluate_without_aspect(dataset_domain)
        accuracy = np.diagonal(matrix).sum() / matrix.sum()
        logger.info(f'{dataset_domain.upper()} DOMAIN\n'
                    f'Acc. {dataset_domain}: {accuracy:.4f}')
