#!/usr/bin/env python3
import melee
import tensorflow as tf
import argparse
import os
import ubjson
import random
import numpy as np
import progressbar
import pathlib

parser = argparse.ArgumentParser(description='AI-powered Melee in-game advantage bar')
parser.add_argument('--train',
                    '-t',
                    action='store_true',
                    help='Training mode')
parser.add_argument('--evaluate',
                    '-e',
                    action='store_true',
                    help='Evaluation mode')
parser.add_argument('--predict',
                    '-p',
                    help='Prediction mode. Specify the directory where dolphin is')
parser.add_argument('--build',
                    '-b',
                    action='store_true',
                    help='Build dataset from SLP files')
args = parser.parse_args()


def _parse_record(record):
    """Parse a single record of a frame in the data and produce numpy output"""
    feature_map = {
        "player1_character": tf.io.FixedLenFeature([], dtype=tf.int64),
        "player1_x": tf.io.FixedLenFeature([], dtype=tf.float32),
        "player1_y": tf.io.FixedLenFeature([], dtype=tf.float32),
        "player1_percent": tf.io.FixedLenFeature([], dtype=tf.float32),
        "player2_x": tf.io.FixedLenFeature([], dtype=tf.float32),
        "player2_y": tf.io.FixedLenFeature([], dtype=tf.float32),
        "player2_percent": tf.io.FixedLenFeature([], dtype=tf.float32),
        "player2_character": tf.io.FixedLenFeature([], dtype=tf.int64),
        "stage": tf.io.FixedLenFeature([], dtype=tf.int64),
        "stock_winner": tf.io.FixedLenFeature([], dtype=tf.int64),
    }

    parsed = tf.io.parse_example(record, feature_map)

    p1character = tf.one_hot(parsed["player1_character"], 26)
    p2character = tf.one_hot(parsed["player2_character"], 26)
    stage = tf.one_hot(parsed["stage"], 6)

    p1x = tf.expand_dims(parsed["player1_x"], 1)
    p1y = tf.expand_dims(parsed["player1_y"], 1)
    p1percent = tf.expand_dims(parsed["player1_percent"], 1)

    p2x = tf.expand_dims(parsed["player2_x"], 1)
    p2y = tf.expand_dims(parsed["player2_y"], 1)
    p2percent = tf.expand_dims(parsed["player2_percent"], 1)

    final = tf.concat([p1character,
                    p2character,
                    stage,
                    p1x,
                    p1y,
                    p1percent,
                    p2x,
                    p2y,
                    p2percent
                    ], 1)

    return final, parsed["stock_winner"]

class AdvantageBarModel:
    """Tensorflow model for the advantage bar
    """
    def __init__(self):
        """AdvantageBarModel

        Input params:
            (one-hot): Stage
            (one-hot): Character of player 1
            (float): X coordinate of player 1
            (float): Y coordinate of player 1
            (float): Damage of player 1
            (one-hot): Character of player 2
            (float): X coordinate of player 2
            (float): Y coordinate of player 2
            (float): Damage of player 2
        """
        # Build the model
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(64,)))
        self.model.add(tf.keras.layers.Dense(128, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(32, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                           loss="binary_crossentropy",
                           metrics=["accuracy"])
        print(self.model.summary())

    def load(self):
        self.model = tf.keras.models.load_model("savedmodel")

    def save(self):
        self.model.save("savedmodel")

    def train(self, epochs=5):
        dir = os.listdir("tfrecords/train/")
        training_files = ["tfrecords/train/" + s for s in dir]
        dir = os.listdir("tfrecords/eval/")
        eval_files = ["tfrecords/eval/" + s for s in dir]

        training_data = tf.data.TFRecordDataset(training_files)
        eval_data = tf.data.TFRecordDataset(eval_files)

        BATCH_SIZE = 10000
        SHUFFLE_BUFFER_SIZE = 100
        # This is about a 20% split for ~1000 SLP files
        VALIDATION_SIZE = 1572000 #TODO any way to make this dynamic? I think not...

        training_data = training_data.shuffle(SHUFFLE_BUFFER_SIZE)
        dataset_validation = training_data.take(VALIDATION_SIZE)
        dataset_train = training_data.skip(VALIDATION_SIZE)

        dataset_train = dataset_train.batch(BATCH_SIZE)
        dataset_validation = dataset_validation.batch(BATCH_SIZE)
        eval_data = eval_data.batch(BATCH_SIZE)

        dataset_train = dataset_train.map(_parse_record)
        dataset_validation = dataset_validation.map(_parse_record)
        eval_data = eval_data.map(_parse_record)

        self.model.fit(dataset_train,
                       validation_data=dataset_validation,
                       epochs=epochs)
        self.model.evaluate(eval_data)


    def predict(self, gamestate):
        p1character = tf.one_hot(gamestate.player[1].character.value, 26).numpy()
        p2character = tf.one_hot(gamestate.player[2].character.value, 26).numpy()
        stage = tf.one_hot(_stage_flatten(gamestate.stage.value), 6).numpy()

        input_array = np.concatenate([
            p1character,
            p2character,
            stage,
            [gamestate.player[1].x],
            [gamestate.player[1].y],
            [gamestate.player[1].percent],
            [gamestate.player[2].x],
            [gamestate.player[2].y],
            [gamestate.player[2].percent],
        ])

        # input_array = np.expand_dims(input_array, 1)
        input_array = np.array([input_array,])

        # print(input_array, input_array.shape)
        prediction = self.model.predict(input_array)
        return prediction

    def build(self):
        pass

def who_died(past_p1, past_p2, current_p1, current_p2):
    """Returns who died."""
    if past_p1 > current_p1:
        return 0
    if past_p2 > current_p2:
        return 1
    return -1

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _stage_flatten(stage):
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

# Builds a dataset out of SLP files in training_data/
if args.build:
    print("Building dataset...")
    directory = 'training_data/'
    num_files = len([f for f in os.listdir(directory)if os.path.isfile(os.path.join(directory, f))])
    bar = progressbar.ProgressBar(maxval=num_files)
    for entry in bar(os.scandir(directory)):
        frames = []
        if entry.path.endswith(".slp") and entry.is_file():
            console = melee.Console(is_dolphin=False, path=entry.path)
            try:
                console.connect()
            except ubjson.decoder.DecoderException as ex:
                print("Got error, skipping file", ex)
                continue
            stocks = (4,4)
            # Pick a game to be part of the evaluation set 20% of the time
            is_evaluation = random.random() > 0.8
            # Temp holder of frames, until a stock is lost
            frames_temp = []

            try:
                # Iterate through each frame of the game
                while True:
                    gamestate = console.step()
                    if gamestate is None:
                        break
                    else:
                        # Save the data in this frame for later. We don't know the label yet
                        frame = {
                            "player1_character": gamestate.player[1].character.value,
                            "player1_x": gamestate.player[1].x,
                            "player1_y": gamestate.player[1].y,
                            "player1_percent": gamestate.player[1].percent,
                            "player2_x": gamestate.player[2].x,
                            "player2_y": gamestate.player[2].y,
                            "player2_percent": gamestate.player[2].percent,
                            "player2_character": gamestate.player[1].character.value,
                            "stage": _stage_flatten(gamestate.stage.value)
                        }
                        frames_temp.append(frame)

                        # Did someone lose a stock? Add the labels on
                        died = who_died(stocks[0], stocks[1], gamestate.player[1].stock, gamestate.player[2].stock)
                        if died > -1:
                            for frame in frames_temp:
                                newframe = tf.train.Example(features=tf.train.Features(feature={
                                    "player1_character": _int64_feature(frame["player1_character"]),
                                    "player1_x": _float_feature(frame["player1_x"]),
                                    "player1_y": _float_feature(frame["player1_y"]),
                                    "player1_percent": _float_feature(frame["player1_percent"]),
                                    "player2_x": _float_feature(frame["player2_x"]),
                                    "player2_y": _float_feature(frame["player2_y"]),
                                    "player2_percent": _float_feature(frame["player2_percent"]),
                                    "player2_character": _int64_feature(frame["player2_character"]),
                                    "stage": _int64_feature(frame["stage"]),
                                    "stock_winner": _int64_feature(died)
                                }))
                                frames.append(newframe)
                            frames_temp = []

                        stocks = (gamestate.player[1].stock, gamestate.player[2].stock)
            except melee.console.SlippiVersionTooLow as ex:
                print("Slippi version too low", ex)

            filename = None
            if is_evaluation:
                filename = "tfrecords/eval" / pathlib.Path(pathlib.Path(entry.path + ".tfrecord").name)
            else:
                filename = "tfrecords/train" /  pathlib.Path(pathlib.Path(entry.path + ".tfrecord").name)
            if len(frames) > 0:
                with tf.io.TFRecordWriter(str(filename)) as file_writer:
                    for frame in frames:
                        file_writer.write(frame.SerializeToString())

if args.train:
    print("Training...")
    model = AdvantageBarModel()
    model.train()
    model.save()


if args.predict:
    print("Predicting...")
    model = AdvantageBarModel()
    model.load()

    # Start a real game
    console = melee.Console(path=args.predict,
                            slippi_address="127.0.0.1",
                            slippi_port=51441,
                            blocking_input=False,
                            logger=None)
    # Run the console
    console.run()

    # Connect to the console
    print("Connecting to console...")
    if not console.connect():
        print("ERROR: Failed to connect to the console.")
        print("\tIf you're trying to autodiscover, local firewall settings can " +
              "get in the way. Try specifying the address manually.")
        sys.exit(-1)
    print("...Connected")

    # Main loop
    while True:
        # "step" to the next frame
        gamestate = console.step()
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            print(model.predict(gamestate))
