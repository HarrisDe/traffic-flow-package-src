# energy_forecasting/deep/label_scaling.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import tensorflow as tf


def _infer_n_outputs(ds: tf.data.Dataset) -> int:
    _, yb = next(iter(ds.take(1)))
    return int(yb.shape[-1]) if yb.shape.rank and yb.shape.rank > 1 else 1


def _label_stats(ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    n_out = _infer_n_outputs(ds)
    s  = tf.zeros([n_out], tf.float32)
    ss = tf.zeros([n_out], tf.float32)
    n  = tf.zeros([], tf.float32)
    for _, y in ds:
        y = tf.cast(y, tf.float32)
        s  += tf.reduce_sum(y, axis=0)
        ss += tf.reduce_sum(tf.square(y), axis=0)
        n  += tf.cast(tf.shape(y)[0], tf.float32)
    mu = s / tf.maximum(n, 1.0)
    var = ss / tf.maximum(n, 1.0) - tf.square(mu)
    sd = tf.sqrt(tf.maximum(var, 1e-8))
    return mu.numpy(), sd.numpy()


@dataclass
class LabelScaler:
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def from_dataset(cls, train_ds: tf.data.Dataset) -> "LabelScaler":
        mu, sd = _label_stats(train_ds)
        return cls(mu=np.asarray(mu, dtype=np.float32),
                   sd=np.asarray(sd, dtype=np.float32))

    def scale_ds(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        mu = tf.constant(self.mu, tf.float32)
        sd = tf.constant(self.sd, tf.float32)
        def _map(x, y):
            y = tf.cast(y, tf.float32)
            return x, (y - mu) / sd
        return ds.map(_map)

    def unscale(self, y_scaled: np.ndarray) -> np.ndarray:
        return np.asarray(y_scaled, dtype=np.float32) * self.sd + self.mu