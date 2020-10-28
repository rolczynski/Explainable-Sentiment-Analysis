import os
import logging
import pathlib
import random
from typing import Iterable
from typing import Tuple

import itertools
import numpy as np
import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import TokenizedExample
from aspect_based_sentiment_analysis import Output
from aspect_based_sentiment_analysis import LabeledExample
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegressionCV
from joblib import Memory

import aspect_condition
import utils

logger = logging.getLogger('analysis.aux_model_reference')
HERE = pathlib.Path(__file__).parent
memory = Memory(HERE / 'outputs')
PRETRAINED_LM_NAMES = {
    'restaurant': 'absa/classifier-rest-0.2',
    'laptop': 'absa/classifier-lapt-0.2'
}


def build_train_dataset(domain: str, seed: int = 0) -> Iterable[LabeledExample]:
    examples = absa.load_examples('semeval', domain, test=False)
    convert_pos = lambda e: LabeledExample(e.text, e.aspect, 1)
    pos_examples = map(convert_pos, examples)

    random.seed(seed)
    nouns = ['lamp', 'window', 'table', 'cap', 'backpack', 'key', 'chair']
    convert_neg = lambda e: LabeledExample(e.text, random.choice(nouns), 0)
    neg_examples = map(convert_neg, examples)
    dataset = itertools.chain(pos_examples, neg_examples)
    return dataset


@memory.cache
def build_train_observations(domain: str) -> np.ndarray:
    recognizer = absa.BasicReferenceRecognizer
    dataset = build_train_dataset(domain)
    batches = absa.utils.batches(dataset, batch_size=32)
    features = []
    for batch in batches:
        tokenized_examples = nlp.tokenize(batch)
        input_batch = nlp.encode(tokenized_examples)
        output_batch = nlp.predict(input_batch)
        for example, tokenized_example, output \
                in zip(batch, tokenized_examples, output_batch):
            masks = recognizer.text_aspect_subtoken_masks(tokenized_example)
            n = len(tokenized_example.subtokens)
            hidden_states = output.hidden_states[:, :n, :]  # TODO in recognizer
            similarity = recognizer.transform(hidden_states, *masks)
            features.append([similarity, example.sentiment])  # classes 0 or 1
    return np.array(features)


def train(domain: str) -> Tuple[float, float]:
    observations = build_train_observations(domain)
    x, y = observations.T
    X = x.reshape(-1, 1)
    clf = LogisticRegressionCV(cv=10, random_state=0)
    clf.fit(X, y)
    β_0 = clf.intercept_[0]
    β_1 = clf.coef_[0, 0]
    return β_0, β_1  # TODO support the logistic regression


@memory.cache
def evaluate(domain: str, seed: int) -> np.ndarray:
    # dataset = aspect_condition.build_dataset(domain, seed)
    dataset = absa.load_examples('semeval', domain, test=True)  # TODO
    metric = absa.training.ConfusionMatrix(num_classes=3)
    confusion_matrix = nlp.evaluate(dataset, metric, batch_size=32)
    confusion_matrix = confusion_matrix.numpy()
    return confusion_matrix


@dataclass
class BasicReferenceRecognizer(absa.BasicReferenceRecognizer):
    weights: Tuple[float, float] = None  # TODO

    def __call__(
            self,
            example: TokenizedExample,
            output: Output
    ) -> bool:
        β_0, β_1 = self.weights
        text_mask, aspect_mask = self.text_aspect_subtoken_masks(example)
        n = len(example.subtokens)
        hidden_states = output.hidden_states[:, :n, :]
        similarity = self.transform(hidden_states, text_mask, aspect_mask)
        is_reference = β_0 + β_1 * similarity > 0
        return is_reference


if __name__ == '__main__':
    os.chdir(HERE)
    utils.setup_logger(HERE / 'logs' / 'aux-model-reference.log')

    logger.info('Begin Evaluation: Predict with the Wrong Aspect')
    for dataset_domain in ['restaurant', 'laptop']:
        lm_name = PRETRAINED_LM_NAMES[dataset_domain]
        nlp = absa.load(lm_name)
        weights = train(dataset_domain)
        recognizer = BasicReferenceRecognizer(threshold=None, weights=weights)
        nlp.professor.reference_recognizer = recognizer
        evaluate(dataset_domain, seed=0)
        # TODO: impact on the original Semeval evaluation
