import os
import logging
import itertools
import pathlib
import random
from typing import Iterable

import numpy as np
import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import LabeledExample
from aspect_based_sentiment_analysis import Sentiment
from joblib import Memory

import utils

logger = logging.getLogger('analysis.aspect_condition')
HERE = pathlib.Path(__file__).parent
memory = Memory(HERE / 'outputs')
PRETRAINED_MODEL_NAMES = {
    'restaurant': 'absa/bert-rest-0.2',
    'laptop': 'absa/bert-lapt-0.2'
}


def build_dataset(domain: str, seed: int) -> Iterable[LabeledExample]:
    examples = absa.load_examples('semeval', domain, test=True)

    # Process only positive/negative examples
    condition = lambda e: e.sentiment in [Sentiment.positive, Sentiment.negative]
    dataset = filter(condition, examples)

    random.seed(seed)
    wrong_aspects = ['car', 'plane', 'window', 'john', 'desk', 'fridge', 'sink']
    random.shuffle(wrong_aspects)
    wrong_aspect = itertools.cycle(wrong_aspects)

    # Map unrelated aspects (verified) and expect the neutral sentiment.
    convert = lambda e: LabeledExample(e.text, next(wrong_aspect), Sentiment.neutral)
    dataset = map(convert, dataset)

    return dataset


@memory.cache
def predict_with_wrong_aspect(domain: str, seed: int) -> np.ndarray:
    name = PRETRAINED_MODEL_NAMES[domain]
    dataset = build_dataset(domain, seed)
    nlp = absa.load(name)

    metric = absa.training.ConfusionMatrix(num_classes=3)
    confusion_matrix = nlp.evaluate(dataset, metric, batch_size=32)
    confusion_matrix = confusion_matrix.numpy()
    return confusion_matrix


if __name__ == '__main__':
    os.chdir(HERE)
    utils.setup_logger(HERE / 'logs' / 'aspect-condition.log')

    logger.info('Begin Evaluation: Predict with the Wrong Aspect')
    for dataset_domain in ['restaurant', 'laptop']:

        matrix = []
        for random_seed in range(7):
            matrix.append(predict_with_wrong_aspect(dataset_domain, random_seed))
        matrix = np.array(matrix)

        acc = lambda m: np.diagonal(m).sum() / m.sum()
        accuracies = [round(acc(m_i), 4) for m_i in matrix]
        mean_accuracy = np.mean(accuracies)
        mean_acu_accuracy = acc(np.sum(matrix, axis=0))
        logger.info(f'{dataset_domain.upper():10} DOMAIN\n'
                    f'Mean Accuracy: {mean_accuracy:.4f}\n'
                    f'Mean Accumulated Accuracy: {mean_acu_accuracy:.4f}\n'
                    f'Details: {accuracies}')
