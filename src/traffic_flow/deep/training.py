# energy_forecasting/deep_tf/trainer.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence, List
import numpy as np
import tensorflow as tf




class TFTrainer:
    """
    Thin training wrapper around a compiled/uncompiled Keras model.

    - compile(loss="huber" | "mae" | "mse", optimizer="adam", metrics=("mae",))
    - fit(train_ds, val_ds, epochs, patience, ...) with EarlyStopping (+ LR schedule optional)
    - predict(test_ds) → np.ndarray (N, n_outputs)
    """

    def __init__(self, model: tf.keras.Model):
        self.model = model

    def compile(
        self,
        *,
        loss: str = "huber",           # "huber" is robust; "mae" or "mse" are fine too
        optimizer: str = "adam",
        metrics: Optional[Sequence] = None,  # e.g. ("mae", "mse", "huber")
        learning_rate: Optional[float] = None,
    ) -> None:
        if optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) if learning_rate else tf.keras.optimizers.Adam()
        else:
            opt = tf.keras.optimizers.get(optimizer)
            if learning_rate is not None:
                tf.keras.backend.set_value(opt.learning_rate, learning_rate)

        loss_fn = {
            "huber": tf.keras.losses.Huber(),
            "mae": "mae",
            "mse": "mse",
        }[loss]
        
        # Metrics default: show overall MAE if none given
        metrics_list: List = list(metrics) if metrics is not None else ["mae"]

        self.model.compile(optimizer=opt, loss=loss_fn, metrics=metrics_list)

    def fit(
        self,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset] = None,
        *,
        epochs: int = 50,
        patience: int = 5,
        reduce_lr: bool = True,
        reduce_lr_patience: int = 3,
        reduce_lr_factor: float = 0.5,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:

        cbs: list[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if val_ds is not None else "loss",
                patience=patience,
                restore_best_weights=True,
            )
        ]
        if reduce_lr and val_ds is not None:
            cbs.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=reduce_lr_factor,
                    patience=reduce_lr_patience,
                    verbose=1,
                    min_lr=1e-6,
                )
            )

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=cbs,
            verbose=verbose,
        )
        return history

    def predict(self, test_ds: tf.data.Dataset) -> np.ndarray:
        preds = self.model.predict(test_ds, verbose=0)
        return np.asarray(preds)