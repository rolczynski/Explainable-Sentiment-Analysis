from dataclasses import dataclass
from typing import List, Tuple

import tensorflow as tf
import numpy as np

from aspect_based_sentiment_analysis import alignment
from aspect_based_sentiment_analysis import Output
from aspect_based_sentiment_analysis import Pattern
from aspect_based_sentiment_analysis import PatternRecognizer
from aspect_based_sentiment_analysis import TokenizedExample
from aspect_based_sentiment_analysis import BasicPatternRecognizer


@dataclass
class RandomPatternRecognizer(PatternRecognizer):

    def __call__(
            self,
            example: TokenizedExample,
            output: Output
    ) -> List[Pattern]:
        weights = np.random.normal(size=len(example.text_tokens))
        pattern = Pattern(1, example.text_tokens, weights)
        return [pattern]


@dataclass
class AttentionPatternRecognizer(BasicPatternRecognizer):
    add_diagonal: bool = True

    def transform(
            self,
            output: Output,
            text_mask: List[bool],
            token_subtoken_alignment: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = output.attentions   # Here is a change.
        x = tf.reduce_sum(x, axis=[0, 1], keepdims=True)
        x = alignment.merge_tensor(x, alignment=token_subtoken_alignment)
        x = x.numpy().squeeze(axis=(0, 1))

        w = x[0, text_mask]
        w /= np.max(w + 1e-9)

        patterns = x[text_mask, :][:, text_mask]
        if self.add_diagonal:
            max_values = np.max(patterns + 1e-9, axis=1)
            np.fill_diagonal(patterns, max_values)
            patterns /= max_values.reshape(-1, 1)
        if self.is_scaled:
            patterns *= w.reshape(-1, 1)
        if self.is_rounded:
            w = np.round(w, decimals=self.round_decimals)
            patterns = np.round(patterns, decimals=self.round_decimals)
        return w, patterns


@dataclass
class GradientPatternRecognizer(BasicPatternRecognizer):
    add_diagonal: bool = True

    def transform(
            self,
            output: Output,
            text_mask: List[bool],
            token_subtoken_alignment: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = output.attention_grads   # Here is a change.
        x = tf.reduce_sum(x, axis=[0, 1], keepdims=True)
        x = alignment.merge_tensor(x, alignment=token_subtoken_alignment)
        x = x.numpy().squeeze(axis=(0, 1))

        w = x[0, text_mask]
        w /= np.max(w + 1e-9)

        patterns = x[text_mask, :][:, text_mask]
        if self.add_diagonal:
            max_values = np.max(patterns + 1e-9, axis=1)
            np.fill_diagonal(patterns, max_values)
            patterns /= max_values.reshape(-1, 1)
        if self.is_scaled:
            patterns *= w.reshape(-1, 1)
        if self.is_rounded:
            w = np.round(w, decimals=self.round_decimals)
            patterns = np.round(patterns, decimals=self.round_decimals)
        return w, patterns
