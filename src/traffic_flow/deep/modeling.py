# traffic_flow/deep/modeling.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal

import tensorflow as tf
from tensorflow.keras import layers as L, Model


class AttentionPooling1D(L.Layer):
    """
    Mask-aware additive attention pooling over time.
    Input:  (B, T, F)
    Output: (B, F)
    """
    def __init__(self, proj_units: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.proj_units = proj_units
        self.proj = None
        self.score = None

    def build(self, input_shape):
        f = int(input_shape[-1])
        u = self.proj_units or f
        self.proj = L.Dense(u, activation="tanh")
        self.score = L.Dense(1, use_bias=False)
        super().build(input_shape)

    def call(self, x, mask=None):
        # x: (B, T, F), mask: (B, T) or None
        h = self.proj(x)                  # (B, T, U)
        e = self.score(h)                 # (B, T, 1)
        e = tf.squeeze(e, axis=-1)        # (B, T)
        if mask is not None:
            minus_inf = tf.constant(-1e9, dtype=e.dtype)
            e = tf.where(mask, e, minus_inf)
        a = tf.nn.softmax(e, axis=1)      # (B, T)
        a = tf.expand_dims(a, axis=-1)    # (B, T, 1)
        c = tf.reduce_sum(a * x, axis=1)  # (B, F)
        return c

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update(dict(proj_units=self.proj_units))
        return cfg


class LSTMBuilder:
    """
    Safe, constraint-aware builder used by TrafficDeepExperiment.
    Enforces:
      - attention_pooling ⇒ last recurrent layer returns sequences
      - stacked RNNs ⇒ all intermediate layers return sequences
      - only one temporal reducer (we use attention OR seq2one)
      - conv_frontend always 'same' padding (no kernel/length mismatch)
      - residual_head concatenates last-step raw features safely
    """

    @staticmethod
    def from_dataset(
        ds_train: tf.data.Dataset,
        *,
        units: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        use_norm: bool = True,
        add_dense: bool = False,
        dense_units: int = 128,
        dense_activation: Optional[str] = "relu",
        adapt_batches: Optional[int] = None,
        bidirectional: bool = True,
        recurrent_dropout: float = 0.0,
        conv_frontend: bool = False,
        conv_filters: int = 64,
        conv_kernel: int = 5,
        layer_norm_in_lstm: bool = False,
        attention_pooling: bool = False,
        residual_head: bool = False,
    ) -> Tuple[Model, Dict[str, Any]]:
        # Infer shapes from a single batch (robust to any batch size)
        try:
            xb, yb = next(iter(ds_train.take(1)))
        except Exception as e:
            raise ValueError(f"Cannot infer shapes from ds_train: {e}")

        if xb.shape.rank != 3:
            raise ValueError(f"Expected inputs rank 3 (B,T,F), got {xb.shape}")

        T = int(xb.shape[1])
        F = int(xb.shape[2])
        out_dim = int(yb.shape[-1]) if yb.shape.rank >= 2 else 1
        input_shape = (T, F)

        model = LSTMBuilder._build(
            input_shape=input_shape,
            output_dim=out_dim,
            units=units,
            n_layers=n_layers,
            dropout=dropout,
            use_norm=use_norm,
            add_dense=add_dense,
            dense_units=dense_units,
            dense_activation=dense_activation,
            adapt_batches=adapt_batches,
            ds_for_adapt=ds_train,
            bidirectional=bidirectional,
            recurrent_dropout=recurrent_dropout,
            conv_frontend=conv_frontend,
            conv_filters=conv_filters,
            conv_kernel=conv_kernel,
            layer_norm_in_lstm=layer_norm_in_lstm,
            attention_pooling=attention_pooling,
            residual_head=residual_head,
        )

        meta = dict(input_shape=input_shape, output_dim=out_dim, features=F, timesteps=T)
        return model, meta

    @staticmethod
    def _build(
        *,
        input_shape: Tuple[int, int],
        output_dim: int,
        units: int,
        n_layers: int,
        dropout: float,
        use_norm: bool,
        add_dense: bool,
        dense_units: int,
        dense_activation: Optional[str],
        adapt_batches: Optional[int],
        ds_for_adapt: Optional[tf.data.Dataset],
        bidirectional: bool,
        recurrent_dropout: float,
        conv_frontend: bool,
        conv_filters: int,
        conv_kernel: int,
        layer_norm_in_lstm: bool,
        attention_pooling: bool,
        residual_head: bool,
    ) -> Model:

        inputs = L.Input(shape=input_shape, name="inputs")
        x = inputs

        # Optional normalization on features (axis=-1)
        if use_norm:
            norm = L.Normalization(axis=-1, name="norm")
            if ds_for_adapt is None:
                raise ValueError("use_norm=True requires ds_for_adapt to adapt Normalization.")
            adapt_ds = ds_for_adapt.map(lambda a, b: a)
            if adapt_batches is not None:
                adapt_ds = adapt_ds.take(int(adapt_batches))
            norm.adapt(adapt_ds)
            x = norm(x)

        # Optional conv frontend to denoise/summarize local patterns
        if conv_frontend:
            x = L.Conv1D(
                filters=int(conv_filters),
                kernel_size=int(conv_kernel),
                padding="same",
                activation="relu",
                name="conv1",
            )(x)
            x = L.Dropout(dropout, name="conv_dropout")(x)

        # Decide sequence-ness of the last recurrent layer
        last_return_sequences = bool(attention_pooling) or (n_layers > 1)

        # Recurrent stack
        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            rs = True if (not is_last) else last_return_sequences

            rnn = L.LSTM(
                units=units,
                return_sequences=rs,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                name=f"lstm_{i+1}",
            )
            if bidirectional:
                rnn = L.Bidirectional(rnn, merge_mode="concat", name=f"bilstm_{i+1}")
            x = rnn(x)

            if layer_norm_in_lstm:
                # Apply LN to the outputs of each recurrent layer
                x = L.LayerNormalization(name=f"ln_{i+1}")(x)

        # Temporal reduction to (B, F*)
        if attention_pooling:
            # x is (B,T,F*); propagate masks if present
            x = AttentionPooling1D(name="attn_pool")(x)
        else:
            # If last RS=False, x is already (B, F*). If True, average pool.
            if last_return_sequences:
                x = L.GlobalAveragePooling1D(name="gap")(x)

        # Optional residual head with raw last-step features
        if residual_head:
            last_feats = L.Lambda(lambda t: t[:, -1, :], name="last_step_feats")(inputs)
            x = L.Concatenate(name="concat_residual")([x, last_feats])

        # Dense head
        if add_dense and dense_units > 0:
            x = L.Dense(int(dense_units), activation=dense_activation or "relu", name="dense")(x)

        outputs = L.Dense(int(output_dim), activation=None, name="output")(x)
        model = Model(inputs, outputs, name="safe_lstm")
        return model