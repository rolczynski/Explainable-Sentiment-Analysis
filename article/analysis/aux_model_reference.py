import os
import logging
import pathlib
import random
from typing import Iterable
from typing import Tuple

import itertools
import numpy as np
import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import LabeledExample
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
            hidden_states = output.hidden_states[:, :n, :]
            similarity = recognizer.transform(hidden_states, *masks)
            features.append([similarity, example.sentiment])  # classes 0 or 1
    return np.array(features)


def train(domain: str) -> Tuple[float, float]:
    observations = build_train_observations(domain)
    x, y = observations.T
    X = x.reshape(-1, 1)
    # The class 1 has the advantage (the 1:10 ratio) because it's
    # important in this case (the purpose of the article) to maintain
    # the high performance of the Semeval test.
    clf = LogisticRegressionCV(cv=10, random_state=0, class_weight={0:1, 1:10})
    clf.fit(X, y)
    β_0 = clf.intercept_[0]
    β_1 = clf.coef_[0, 0]
    return β_0, β_1


@memory.cache
def evaluate(domain: str, nlp_modified: bool = False) -> np.ndarray:
    dataset = absa.load_examples('semeval', domain, test=True)
    metric = absa.training.ConfusionMatrix(num_classes=3)
    confusion_matrix = nlp.evaluate(dataset, metric, batch_size=32)
    confusion_matrix = confusion_matrix.numpy()
    return confusion_matrix


@memory.cache
def evaluate_condition(domain: str, nlp_modified: bool = False) -> np.ndarray:
    dataset = aspect_condition.build_dataset(domain, 0)  # default seed=0
    metric = absa.training.ConfusionMatrix(num_classes=3)
    confusion_matrix = nlp.evaluate(dataset, metric, batch_size=32)
    confusion_matrix = confusion_matrix.numpy()
    return confusion_matrix


if __name__ == '__main__':
    os.chdir(HERE)
    utils.setup_logger(HERE / 'logs' / 'aux-model-reference.log')

    logger.info('Begin Evaluation: Basic Pattern Recognizer')
    for dataset_domain in ['restaurant', 'laptop']:
        lm_name = PRETRAINED_LM_NAMES[dataset_domain]
        nlp = absa.load(lm_name)

        accuracy = lambda m: np.diagonal(m).sum() / m.sum()
        ref_matrix = evaluate(dataset_domain)
        ref_acc = accuracy(ref_matrix)
        ref_test_matrix = evaluate_condition(dataset_domain)
        ref_test_acc = accuracy(ref_test_matrix)

        weights = train(dataset_domain)
        recognizer = absa.BasicReferenceRecognizer(weights=weights)
        nlp.professor.reference_recognizer = recognizer
        matrix = evaluate(dataset_domain, nlp_modified=True)
        acc = accuracy(matrix)

        test_matrix = evaluate_condition(dataset_domain, nlp_modified=True)
        test_acc = accuracy(test_matrix)

        semeval_improvement = (acc - ref_acc) / ref_acc
        test_improvement = (test_acc - ref_test_acc) / ref_test_acc

        logger.info(f'{dataset_domain.upper()} DOMAIN\n'
                    # Semeval Results
                    f'Acc. Semeval: {ref_acc:.4f}\n'
                    f'Confusion Matrix (y, y_hat)\n{ref_matrix}\n'
                    f'Acc. Semeval with the ref. recognizer: {acc:.4f}\n'
                    f'Confusion Matrix (y, y_hat)\n{matrix}\n'
                    f'Relative Improvement: {semeval_improvement*100:.2f} %\n\n'
                    # Test B Results
                    f'Acc. Test B: {ref_test_acc:.4f}\n'
                    f'Confusion Matrix (y, y_hat)\n{ref_test_matrix}\n'
                    f'Acc. Test B with the ref. recognizer: {test_acc:.4f}\n'
                    f'Confusion Matrix (y, y_hat)\n{test_matrix}\n'
                    f'Relative Improvement: {test_improvement*100:.2f} %\n\n' 
                    # Details
                    f'(details) the ref. recognizer weights: {weights}\n')

        shortcut = {'restaurant': 'rest', 'laptop': 'lapt'}
        name = f'./basic_reference_recognizer-{shortcut[dataset_domain]}-0.1'
        recognizer.save_pretrained(name)
