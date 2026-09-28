"""
Keras model loading that survives Python/Keras upgrades.

The LSTM files trained before v1.8 contain a `Lambda(lambda t: tf.reduce_sum(t, axis=1))`
attention-pooling layer. Keras stores Lambda functions as *Python bytecode*,
which (a) is refused by Keras 3 safe_mode and (b) can't be unmarshalled on a
different Python version ("bad marshal data"). Result in v1.12: ASTRA.ML and
the crypto LSTM silently never loaded.

load_keras_model() first tries a normal load; if that fails it rewrites the
saved config, replacing that Lambda with an equivalent registered layer
(ReduceSumOverTime), rebuilds the graph and loads the original trained weights.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import zipfile

logger = logging.getLogger(__name__)

_ReduceSum = None


def _reduce_sum_layer():
    global _ReduceSum
    if _ReduceSum is None:
        import keras

        @keras.saving.register_keras_serializable(package="astra")
        class ReduceSumOverTime(keras.layers.Layer):
            """Equivalent of Lambda(lambda t: tf.reduce_sum(t, axis=1))."""

            def call(self, inputs, mask=None):
                return keras.ops.sum(inputs, axis=1)

            def compute_mask(self, inputs, mask=None):
                return None

            def compute_output_shape(self, input_shape):
                return (input_shape[0],) + tuple(input_shape[2:])

        _ReduceSum = ReduceSumOverTime
    return _ReduceSum


def _patch_config(cfg: dict) -> int:
    """Replace every Lambda layer in a functional config in-place. Returns count."""
    n = 0
    for layer in cfg.get("config", {}).get("layers", []):
        if layer.get("class_name") == "Lambda":
            name = layer["config"].get("name")
            layer["class_name"] = "ReduceSumOverTime"
            layer["module"] = None
            layer["registered_name"] = "astra>ReduceSumOverTime"
            layer["config"] = {"name": name, "trainable": True, "dtype": layer["config"].get("dtype", "float32")}
            n += 1
    return n


def load_keras_model(path: str, custom_objects: dict = None):
    import keras
    try:
        return keras.models.load_model(path, custom_objects=custom_objects)
    except Exception as first_err:
        if not path.endswith(".keras"):
            raise
        logger.info(f"Standard load failed for {os.path.basename(path)} ({str(first_err)[:80]}…); "
                    "rebuilding with compat layers")
    cls = _reduce_sum_layer()
    with zipfile.ZipFile(path) as zf:
        cfg = json.loads(zf.read("config.json"))
        replaced = _patch_config(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "model.weights.h5")
            with open(weights, "wb") as fh:
                fh.write(zf.read("model.weights.h5"))
            model = keras.saving.deserialize_keras_object(
                cfg, custom_objects={"ReduceSumOverTime": cls, "astra>ReduceSumOverTime": cls,
                                     **(custom_objects or {})})
            model.load_weights(weights)
    logger.info(f"Loaded {os.path.basename(path)} via compat loader ({replaced} Lambda layer(s) replaced)")
    return model
