"""Data prep, train and evaluate DNN model."""

import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, models
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Input,
)

logging.info(tf.version.VERSION)

# TODO: Parametrise the column list to use the same package for all positions.
CSV_COLUMNS = [
    "hash_id",
    "element_code",
    "season_name",
    "next_season_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "penalties_missed",
    "bps",
    "yellow_threshold",
    "red_cards",
    "own_goals",
    "influence",
    "creativity",
    "threat",
    "start_cost",
    "end_cost"
]

LABEL_COLUMN = "next_season_points"
DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
UNWANTED_COLS = ["hash_id", "element_code", "season_name"]

INPUT_COLS = [
    c for c in CSV_COLUMNS if c != LABEL_COLUMN and c not in UNWANTED_COLS
]

def features_and_labels(row_data):
    for unwanted_col in UNWANTED_COLS:
        row_data.pop(unwanted_col)
    label = row_data.pop(LABEL_COLUMN)
    return row_data, label


def load_dataset(pattern, batch_size, num_repeat):
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=pattern,
        batch_size=batch_size,
        column_names=CSV_COLUMNS,
        column_defaults=DEFAULTS,
        num_epochs=num_repeat,
        shuffle_buffer_size=1000000,
    )
    return dataset.map(features_and_labels)


def create_train_dataset(pattern, batch_size):
    dataset = load_dataset(pattern, batch_size, num_repeat=None)
    return dataset.prefetch(1)


def create_eval_dataset(pattern, batch_size):
    dataset = load_dataset(pattern, batch_size, num_repeat=1)
    return dataset.prefetch(1)


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


def build_dnn_model(nnsize, lr):
    inputs = {
        colname: Input(name=colname, shape=(1,), dtype="float32")
        for colname in INPUT_COLS
    }

    # Concatenate numeric inputs
    dnn_inputs = Concatenate()(list(inputs.values()))

    x = dnn_inputs
    for layer, nodes in enumerate(nnsize):
        x = Dense(nodes, activation="relu", name=f"h{layer}")(x)
    output = Dense(1, name="next_sason_points")(x)

    model = models.Model(inputs, output)

    lr_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=lr_optimizer, loss="mse", metrics=[rmse, "mse"])

    return model


def train_and_evaluate(hparams):
    # TODO 1b
    batch_size = hparams["batch_size"]
    lr = hparams["lr"]
    nnsize = [int(s) for s in hparams["nnsize"].split()]
    eval_data_path = hparams["eval_data_path"]
    num_evals = hparams["num_evals"]
    num_examples_to_train_on = hparams["num_examples_to_train_on"]
    output_dir = hparams["output_dir"]
    train_data_path = hparams["train_data_path"]

    model_export_path = os.path.join(output_dir, "savedmodel")
    checkpoint_path = os.path.join(output_dir, "checkpoints")
    tensorboard_path = os.path.join(output_dir, "tensorboard")

    if tf.io.gfile.exists(output_dir):
        tf.io.gfile.rmtree(output_dir)

    model = build_dnn_model(nnsize, lr)
    logging.info(model.summary())

    trainds = create_train_dataset(train_data_path, batch_size)
    evalds = create_eval_dataset(eval_data_path, batch_size)

    steps_per_epoch = num_examples_to_train_on // (batch_size * num_evals)

    checkpoint_cb = callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True, verbose=1
    )
    tensorboard_cb = callbacks.TensorBoard(tensorboard_path, histogram_freq=1)

    history = model.fit(
        trainds,
        validation_data=evalds,
        epochs=num_evals,
        steps_per_epoch=max(1, steps_per_epoch),
        verbose=2,  # 0=silent, 1=progress bar, 2=one line per epoch
        callbacks=[checkpoint_cb, tensorboard_cb],
    )

    # Exporting the model with default serving function.
    model.save(model_export_path)
    return history
