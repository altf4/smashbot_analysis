"""Advantage Bar Model class"""

import tensorflow as tf
import numpy as np
import os

def _parse_winner(record):
    context_feature_map = {
        "game_winner": tf.io.FixedLenFeature([], dtype=tf.float32),
    }
    ctx, _ = tf.io.parse_single_sequence_example(record,
                                              sequence_features=None,
                                              context_features=context_feature_map)

    return ctx["game_winner"]

def _parse_features(record):
    """Parse a batch of tfrecord data, output batch of tensors ready for training

    This is for use with tf.data, so everything here has to be executed as a tensorflow op.
    You can't do arbitrary python in this function.
    """
    feature_map = {
        "player1_character": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
        "player1_x": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "player1_y": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "player1_percent": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "player1_stock": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "player2_x": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "player2_y": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "player2_percent": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "player2_character": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
        "player2_stock": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "stage": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
        "stock_winner": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
    }

    _, parsed = tf.io.parse_single_sequence_example(record,
                                                      sequence_features=feature_map,
                                                      context_features=None)

    p1character = tf.one_hot(parsed["player1_character"], 26)
    p2character = tf.one_hot(parsed["player2_character"], 26)
    stage = tf.one_hot(parsed["stage"], 6)

    p1x = tf.expand_dims(parsed["player1_x"], 1)
    p1y = tf.expand_dims(parsed["player1_y"], 1)
    p1percent = tf.expand_dims(parsed["player1_percent"], 1)
    p1stock = tf.expand_dims(parsed["player1_stock"], 1)

    p2x = tf.expand_dims(parsed["player2_x"], 1)
    p2y = tf.expand_dims(parsed["player2_y"], 1)
    p2percent = tf.expand_dims(parsed["player2_percent"], 1)
    p2stock = tf.expand_dims(parsed["player2_stock"], 1)

    final = tf.concat([p1character,
                    p2character,
                    stage,
                    p1x,
                    p1y,
                    p1percent,
                    p1stock,
                    p2x,
                    p2y,
                    p2percent,
                    p2stock,
                    ], 1)
    return final

class AdvantageBarModel:
    """Tensorflow model for the advantage bar
    """
    def __init__(self):
        """AdvantageBarModel

        Input params:
            (one-hot): Character of player 1
            (one-hot): Character of player 2
            (one-hot): Stage
            (float): X coordinate of player 1
            (float): Y coordinate of player 1
            (float): Damage of player 1
            (float): Stock of player 1
            (float): X coordinate of player 2
            (float): Y coordinate of player 2
            (float): Damage of player 2
            (float): Stock of player 2
        """
        self._BATCH_SIZE = 10
        self._TIME_LENGTH = 20

        # Build the model
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(self._TIME_LENGTH, 66,)))
        self.model.add(tf.keras.layers.LSTM(128))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(32, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.0010),
                           loss="binary_crossentropy",
                           metrics=["accuracy"])
        print(self.model.summary())

    def load(self):
        """Load weights for the model from file"""
        self.model = tf.keras.models.load_model("savedmodel")

    def save(self):
        """Save the current model to file"""
        self.model.save("savedmodel")

    def train(self, epochs=10):
        """Train the model

        Assumes a directory structure (created by the "build" mode) like this:
        tfrecrds/
            train/
                *.tfrecord
            eval/
                *.tfrecord
        """
        dir = os.listdir("tfrecords/train/")
        training_files = ["tfrecords/train/" + s for s in dir]
        dir = os.listdir("tfrecords/eval/")
        eval_files = ["tfrecords/eval/" + s for s in dir]

        training_data = tf.data.TFRecordDataset(training_files)
        eval_data = tf.data.TFRecordDataset(eval_files)

        SHUFFLE_BUFFER_SIZE = 1
        # This is about a 20% split for ~1000 SLP files
        # TODO any way to make this dynamic? Maybe extrapolate from the number of tfrecords
        VALIDATION_SIZE = 10

        # The operatons below happen as part of the tf.data pipeline
        training_data = training_data.shuffle(SHUFFLE_BUFFER_SIZE)
        dataset_validation = training_data.take(VALIDATION_SIZE)
        dataset_train = training_data.skip(VALIDATION_SIZE)

        # Parse the tfrecord file into tensor datasets
        dataset_train_features = dataset_train.map(_parse_features)
        dataset_train_labels = dataset_train.map(_parse_winner)
        dataset_validation_features = dataset_validation.map(_parse_features)
        dataset_validation_labels = dataset_validation.map(_parse_winner)
        eval_data_features = eval_data.map(_parse_features)
        eval_data_labels = eval_data.map(_parse_winner)

        # This part is working
        print("Printing features")
        for thing in dataset_train_features:
            print(thing)

        print("Printing labels")
        for thing in dataset_train_labels:
            print(thing)

        # Window the data
        # XXX I think this should work, but it doesn't? It's always empty :(
        dataset_train_features = dataset_train_features.window(self._TIME_LENGTH, drop_remainder=True)
        dataset_train_features = dataset_train_features.flat_map(lambda window: window.batch(self._BATCH_SIZE))

        dataset_validation_features = dataset_validation_features.window(self._TIME_LENGTH, drop_remainder=True)
        dataset_validation_features = dataset_validation_features.flat_map(lambda window: window.batch(self._BATCH_SIZE))

        eval_data_features = eval_data_features.window(self._TIME_LENGTH, drop_remainder=True)
        eval_data_features = eval_data_features.flat_map(lambda window: window.batch(self._BATCH_SIZE))

        # dataset_train = dataset_train.batch(self._BATCH_SIZE)
        # dataset_validation = dataset_validation.batch(self._BATCH_SIZE)
        # eval_data = eval_data.batch(self._BATCH_SIZE)

        # Zip the datasets back together
        # TODO The sizes on these are probably wrong. I bet the zips won't match up
        training_set = tf.data.Dataset.zip((dataset_train_features, dataset_train_labels))
        validation_set = tf.data.Dataset.zip((dataset_validation_features, dataset_validation_labels))
        eval_set = tf.data.Dataset.zip((eval_data_features, eval_data_labels))

        self.model.fit(training_set,
                       validation_data=validation_set,
                       epochs=epochs)
        self.model.evaluate(eval_set)


    def predict(self, gamestate):
        """Given a single libmelee gamestate, make a prediction"""
        p1character = tf.one_hot(gamestate.player[1].character.value, 26).numpy()
        p2character = tf.one_hot(gamestate.player[2].character.value, 26).numpy()
        stage = tf.one_hot(AdvantageBarModel.stage_flatten(gamestate.stage.value), 6).numpy()

        input_array = np.concatenate([
            p1character,
            p2character,
            stage,
            [gamestate.player[1].x],
            [gamestate.player[1].y],
            [gamestate.player[1].percent],
            [gamestate.player[1].stock],
            [gamestate.player[2].x],
            [gamestate.player[2].y],
            [gamestate.player[2].percent],
            [gamestate.player[2].stock],
        ])

        input_array = np.array([input_array,])

        prediction = self.model.predict(input_array)
        return prediction

    @staticmethod
    def stage_flatten(stage):
        """Flattens the stage list to be 0-5

        It's easier for the ML this way, with fewer dead values
        """
        if stage == 0x19:
            return 0
        if stage == 0x18:
            return 1
        if stage == 0x12:
            return 2
        if stage == 0x1A:
            return 3
        if stage == 0x8:
            return 4
        if stage == 0x6:
            return 5
        return 0
