from __future__ import annotations
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


class LSTMBuilder:
    """
    Many-to-one LSTM for multi-horizon forecasting.

    Usage:
      model, builder = LSTMBuilder.from_dataset(
          train_dataset,
          units=64, n_layers=2, dropout=0.2, use_norm=True,
          adapt_batches=None  # or an int to limit how many batches to scan
      )
    """

    def __init__(
        self,
        *,
        seq_len: int,
        n_features: int,
        n_outputs: int,
        units: int = 64,
        n_layers: int = 2,
        dropout: float = 0.0,
        add_dense: bool = True,
        dense_units: int = 16,
        dense_activation: Optional[str] = "relu",
        use_norm: bool = True,
    ) -> None:
        self.seq_len = int(seq_len)
        self.n_features = int(n_features)
        self.n_outputs = int(n_outputs)
        self.units = int(units)
        self.n_layers = int(n_layers)
        self.add_dense = bool(add_dense)
        self.dense_units = int(dense_units)
        self.dense_activation = dense_activation
        self.dropout = float(dropout)
        self.use_norm = bool(use_norm)
        
        

        self.normalizer: Optional[layers.Normalization] = None
        self.model: Optional[tf.keras.Model] = None

    # ----------------------- convenience constructor -----------------------
    @classmethod
    def from_dataset(
        cls,
        ds: tf.data.Dataset,
        *,
        units: int = 64,
        n_layers: int = 2,
        dropout: float = 0.0,
        use_norm: bool = True,
        add_dense: bool = True,
        dense_units: int = 16,
        dense_activation: Optional[str] = "relu",
        adapt_batches: Optional[int] = None,   # None = scan full (finite) ds; int = limit
    ) -> Tuple[tf.keras.Model, "LSTMBuilder"]:
        """
        Infer (seq_len, n_features, n_outputs) from a dataset batch,
        build the model, and if use_norm=True, adapt the Normalization
        layer directly from the dataset (by flattening time into batch).

        IMPORTANT: 'ds' must be finite during adapt (no .repeat()).
        If you have .repeat() in your pipeline, pass adapt_batches to cap it.
        """
        seq_len, n_features, n_outputs = cls._infer_shapes_from_dataset(ds)
        builder = cls(
            seq_len=seq_len,
            n_features=n_features,
            n_outputs=n_outputs,
            units=units,
            n_layers=n_layers,
            add_dense=add_dense,
            dense_units=dense_units,
            dense_activation=dense_activation,
            dropout=dropout,
            use_norm=use_norm,
        )
        model = builder._build_uncompiled()

        if use_norm:
            builder._adapt_normalizer_from_dataset(ds, n_features=n_features, adapt_batches=adapt_batches)

        return model, builder

    # ----------------------- shape inference -----------------------
    @staticmethod
    def _infer_shapes_from_dataset(ds: tf.data.Dataset) -> Tuple[int, int, int]:
        # Peek one batch
        xb, yb = next(iter(ds))
        if xb.shape.rank != 3:
            raise ValueError(f"Expected X batch rank 3 (B, T, F); got shape {xb.shape}")
        seq_len = int(xb.shape[1])
        n_features = int(xb.shape[2])

        if yb.shape.rank == 2:
            n_outputs = int(yb.shape[1])
        elif yb.shape.rank == 1:
            n_outputs = 1
        else:
            raise ValueError(f"Unsupported y batch shape {yb.shape}")

        return seq_len, n_features, n_outputs

    # ----------------------- normalization adapt -----------------------
    def _adapt_normalizer_from_dataset(
        self,
        ds: tf.data.Dataset,
        *,
        n_features: int,
        adapt_batches: Optional[int] = None,
    ) -> None:
        """
        Adapt the Normalization layer **from (B, T, F) batches**.
        IMPORTANT: 'ds' must be finite during adapt (no .repeat()), or set adapt_batches.
        """
        if self.normalizer is None:
            return  # created in _build_uncompiled()

        # Keep only X from (X, y)
        ds_x = ds.map(lambda x, _y: x, num_parallel_calls=tf.data.AUTOTUNE)

        # Optionally cap how many batches to scan (useful if ds repeats infinitely)
        if adapt_batches is not None:
            ds_x = ds_x.take(adapt_batches)

        # Now adapt mean/std across all batches with shape (B, T, F)
        self.normalizer.adapt(ds_x)
    # ----------------------- build the LSTM -----------------------
    def _build_uncompiled(self) -> tf.keras.Model:
        inputs = layers.Input(shape=(self.seq_len, self.n_features), name="series")
        x = inputs

        if self.use_norm:
            # Create the normalization layer; adapted later via dataset
            self.normalizer = layers.Normalization(axis=-1, name="norm")
            x = self.normalizer(x)

        for i in range(self.n_layers):
            return_seq = (i < self.n_layers - 1)
            x = layers.LSTM(self.units, return_sequences=return_seq, name=f"lstm_{i+1}")(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout, name=f"dropout_{i+1}")(x)
    
        if self.add_dense:
            x = layers.Dense(self.dense_units,activation=self.dense_activation,name="final_mlp")(x) 
        outputs = layers.Dense(self.n_outputs, name="regression_head")(x)
        self.model = models.Model(inputs=inputs, outputs=outputs, name="lstm_multi_horizon")
        return self.model