import argparse
import os
import itertools
import logging
import pathlib
import random
from typing import Iterable

import numpy as np
import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import LabeledExample
from aspect_based_sentiment_analysis import Sentiment
from joblib import Memory

import utils

logger = logging.getLogger('analysis.aspect_separation')
HERE = pathlib.Path(__file__).parent
memory = Memory(HERE / 'outputs')
PRETRAINED_MODEL_NAMES = {
    'restaurant': 'absa/bert-rest-0.2',
    'laptop': 'absa/bert-lapt-0.2'
}
TEMPLATES = {
    Sentiment.positive: ['the {noun} is really great.',
                         'the {noun} is perfect.',
                         'the {noun} was awesome.',
                         'the {noun} was brilliant.',
                         'i love the {noun}.',
                         'i like the {noun} a lot.',
                         'i highly recommends the {noun}.'],
    Sentiment.negative: ['the {noun} is really bad.',
                         'the {noun} is awful.',
                         'the {noun} was poor.',
                         'the {noun} was ugly.',
                         'i hate the {noun}.',
                         'i dislike the {noun}.',
                         'i strongly dislike the {noun}.']
}


def build_dataset(
        domain: str,
        template: str,
        template_sent: Sentiment,
        seed: int
) -> Iterable[LabeledExample]:
    examples = absa.load_examples('semeval', domain, test=True)

    mapping = {
        Sentiment.negative: [Sentiment.neutral, Sentiment.positive],
        Sentiment.positive: [Sentiment.neutral, Sentiment.negative]
    }
    condition = lambda e: e.sentiment in mapping[template_sent]
    dataset = filter(condition, examples)

    random.seed(seed)
    nouns = ['car', 'plane', 'bottle', 'bag', 'desk', 'fridge', 'sink']

    convert = lambda e: LabeledExample(
        text=e.text + " " + template.format(noun=random.choice(nouns)),
        aspect=e.aspect,
        sentiment=e.sentiment)
    dataset = map(convert, dataset)

    return dataset


@memory.cache
def evaluate(domain: str):
    name = PRETRAINED_MODEL_NAMES[domain]
    dataset = absa.load_examples('semeval', domain, test=True)
    nlp = absa.load(name)

    metric = absa.training.ConfusionMatrix(num_classes=3)
    confusion_matrix = nlp.evaluate(dataset, metric, batch_size=32)
    confusion_matrix = confusion_matrix.numpy()
    return confusion_matrix


@memory.cache
def evaluate_with_enriched_text(
        domain: str,
        template: str,
        template_sent: Sentiment,
        seed: int
) -> np.ndarray:
    name = PRETRAINED_MODEL_NAMES[domain]
    dataset = build_dataset(domain, template, template_sent, seed)
    nlp = absa.load(name)

    metric = absa.training.ConfusionMatrix(num_classes=3)
    confusion_matrix = nlp.evaluate(dataset, metric, batch_size=32)
    confusion_matrix = confusion_matrix.numpy()
    return confusion_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', action='store', type=int, default=7,
                        help='The number of seeds to check.')
    args = parser.parse_args()
    os.chdir(HERE)
    utils.setup_logger(HERE / 'logs' / 'aspect-separation.log')

    logger.info('Begin Evaluation: Predict with the Enriched Text '
                '(added one emotional sentence).')

    domains = ['restaurant', 'laptop']
    ref_results = [evaluate(domain) for domain in domains]
    accuracy = lambda m: np.diagonal(m).sum() / m.sum()
    # Note that positive refers to the positive template therefore here
    # we select neutral and negative examples in the first case [0, 1].
    ref_acc_pos = np.array([accuracy(m[:, [0, 1]]) for m in ref_results])
    ref_acc_neg = np.array([accuracy(m[:, [0, 2]]) for m in ref_results])

    templates = [(t, s) for s,templates in TEMPLATES.items() for t in templates]
    seeds = range(args.seeds)
    space = itertools.product(domains, templates, seeds)
    results = [evaluate_with_enriched_text(domain, template, template_sent, seed)
               for domain, (template, template_sent), seed in space]
    acc = list(map(accuracy, results))

    i, j, k = len(domains), len(templates), len(seeds)
    acc = np.array(acc).reshape((i, j, k))

    acc_template_mean = acc.mean(axis=-1)
    acc_template_std = acc.std(axis=-1)
    positive = acc_template_mean[:, :j//2]
    negative = acc_template_mean[:, j//2:]

    acc_pos_mean, acc_pos_std = positive.mean(axis=-1), positive.std(axis=-1)
    acc_neg_mean, acc_neg_std = negative.mean(axis=-1), negative.std(axis=-1)

    acc_pos_improvement = (acc_pos_mean - ref_acc_pos) / ref_acc_pos
    acc_neg_improvement = (acc_neg_mean - ref_acc_neg) / ref_acc_neg

    details = lambda i: '\n'.join(f'{t:35} Acc. mean: {μ:.4f} std: {σ:.4f}'
        for (t, s), μ, σ in zip(templates, acc_template_mean[i], acc_template_std[i]))

    summary = lambda i: \
        (f'Added the Positive Sentence\n'
         f'Acc. mean: {acc_pos_mean[i]:.4f} std: {acc_pos_mean[i]:.4f}\n'
         f'Relative Improvement: {acc_pos_improvement[i]*100:.2f} %\n\n'
         f'Added the Negative Sentence\n'
         f'Acc. mean: {acc_neg_mean[i]:.4f} std: {acc_neg_mean[i]:.4f}\n'
         f'Relative Improvement: {acc_neg_improvement[i]*100:.2f} %\n\n'
         f'Details:\n{details(i)}\n')

    logger.info(f'RESTAURANT DOMAIN\n' + summary(i=0))
    logger.info(f'LAPTOP DOMAIN\n' + summary(i=1))
