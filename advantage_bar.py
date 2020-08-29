#!/usr/bin/env python3
import melee
import tensorflow as tf
import argparse
import os
import ubjson
import random
import numpy as np
import progressbar

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
                    action='store_true',
                    help='Prediction mode')
parser.add_argument('--build',
                    '-b',
                    action='store_true',
                    help='Build dataset from SLP files')
args = parser.parse_args()


class AdvantageBarModel:
    """Tensorflow model for the advantage bar
    """
    def __init__(self):
        """AdvantageBarModel

        Input params:
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
        self.model.add(tf.keras.layers.InputLayer(input_shape=(8,)))
        self.model.add(tf.keras.layers.Dense(128, activation="relu"))
        # self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        # self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(32, activation="relu"))
        # self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                           loss="binary_crossentropy",
                           metrics=["accuracy"])
        print(self.model.summary())

    def load(self):
        pass

    def save(self):
        pass

    def train(self, epochs=10):
        with np.load("dataset.npz") as data:
            frames_train = data['frames_train']
            labels_train = data['labels_train']
            frames_eval = data['frames_eval']
            labels_eval = data['labels_eval']

            print(len(frames_train), len(labels_train))
            print(len(frames_eval), len(labels_eval))

            train_dataset = tf.data.Dataset.from_tensor_slices((frames_train, labels_train))
            test_dataset = tf.data.Dataset.from_tensor_slices((frames_eval, labels_eval))

            BATCH_SIZE = 64
            SHUFFLE_BUFFER_SIZE = 100
            train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
            test_dataset = test_dataset.batch(BATCH_SIZE)

            self.model.fit(train_dataset, epochs=epochs)

            self.model.evaluate(test_dataset)


    def build(self):
        pass


def who_died(past_p1, past_p2, current_p1, current_p2):
    """Returns who died."""
    if past_p1 > current_p1:
        return 0
    if past_p2 > current_p2:
        return 1
    return -1

# Builds a dataset out of SLP files in training_data/
if args.build:
    print("Building dataset...")
    directory = r'training_data/'
    frames_eval = []
    frames_train = []
    labels_eval = []
    labels_train = []

    num_files = len([f for f in os.listdir(directory)if os.path.isfile(os.path.join(directory, f))])
    bar = progressbar.ProgressBar(maxval=num_files)
    for entry in bar(os.scandir(directory)):
        if entry.path.endswith(".slp") and entry.is_file():
            console = melee.Console(is_dolphin=False, path=entry.path)
            try:
                console.connect()
            except ubjson.decoder.DecoderException as ex:
                print("Got error, skipping file", ex)
                continue
            stocks = (4,4)
            # Pick a game to be part of the evaluation set 20% of the time
            is_evaluation = int(random.random() > 0.8)
            # Temp holder of frames, until a stock is lost
            frames_temp = []

            try:
                # Iterate through each frame of the game
                while True:
                    gamestate = console.step()
                    if gamestate is None:
                        break
                    else:
                        # Prepare the new frame
                        newframe = np.array((
                            gamestate.player[1].character.value,
                            gamestate.player[1].x,
                            gamestate.player[1].y,
                            gamestate.player[1].percent,
                            gamestate.player[2].character.value,
                            gamestate.player[2].x,
                            gamestate.player[2].y,
                            gamestate.player[2].percent,
                            ))

                        frames_temp.append(newframe)

                        # Did someone lose a stock?
                        died = who_died(stocks[0], stocks[1], gamestate.player[1].stock, gamestate.player[2].stock)
                        if died > -1:
                            if is_evaluation:
                                labels_eval.extend([died] * len(frames_temp))
                                frames_eval.extend(frames_temp)
                            else:
                                labels_train.extend([died] * len(frames_temp))
                                frames_train.extend(frames_temp)
                            frames_temp = []

                        stocks = (gamestate.player[1].stock, gamestate.player[2].stock)
            except melee.console.SlippiVersionTooLow as ex:
                print("Slippi version too low", ex)

    np.savez("dataset.npz", frames_train=frames_train, labels_train=labels_train,
                            frames_eval=frames_eval, labels_eval=labels_eval)

if args.train:
    print("Training...")
    model = AdvantageBarModel()
    model.train()
