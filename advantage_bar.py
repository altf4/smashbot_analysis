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
from model import AdvantageBarModel

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
            game_winner = -1
            try:
                # Iterate through each frame of the game
                while True:
                    gamestate = console.step()
                    if gamestate is None:
                        if stocks[0] > stocks[1]:
                            game_winner = 0
                        if stocks[0] < stocks[1]:
                            game_winner = 1
                        break
                    else:
                        # Save the data in this frame for later. We don't know the label yet
                        frame = {
                            "player1_character": gamestate.player[1].character.value,
                            "player1_x": gamestate.player[1].x,
                            "player1_y": gamestate.player[1].y,
                            "player1_percent": gamestate.player[1].percent,
                            "player1_stock": gamestate.player[1].stock,
                            "player2_x": gamestate.player[2].x,
                            "player2_y": gamestate.player[2].y,
                            "player2_percent": gamestate.player[2].percent,
                            "player2_stock": gamestate.player[2].stock,
                            "player2_character": gamestate.player[1].character.value,
                            "stage": AdvantageBarModel.stage_flatten(gamestate.stage.value),
                            "stock_winner": -1
                        }
                        frames_temp.append(frame)

                        # Did someone lose a stock? Add the labels on
                        died = who_died(stocks[0], stocks[1], gamestate.player[1].stock, gamestate.player[2].stock)
                        if died > -1:
                            for frame in frames_temp:
                                frame["stock_winner"] = died
                                frames.append(frame)
                            frames_temp = []

                        stocks = (gamestate.player[1].stock, gamestate.player[2].stock)
            except melee.console.SlippiVersionTooLow as ex:
                print("Slippi version too low", ex)

            filename = None
            if is_evaluation:
                filename = "tfrecords/eval" / pathlib.Path(pathlib.Path(entry.path + ".tfrecord").name)
            else:
                filename = "tfrecords/train" /  pathlib.Path(pathlib.Path(entry.path + ".tfrecord").name)
            if len(frames) > 0 and game_winner > -1:
                with tf.io.TFRecordWriter(str(filename)) as file_writer:
                    for frame in frames:
                        newframe = tf.train.Example(features=tf.train.Features(feature={
                            "player1_character": _int64_feature(frame["player1_character"]),
                            "player1_x": _float_feature(frame["player1_x"]),
                            "player1_y": _float_feature(frame["player1_y"]),
                            "player1_percent": _float_feature(frame["player1_percent"]),
                            "player1_stock": _float_feature(frame["player1_stock"]),
                            "player2_x": _float_feature(frame["player2_x"]),
                            "player2_y": _float_feature(frame["player2_y"]),
                            "player2_percent": _float_feature(frame["player2_percent"]),
                            "player2_stock": _float_feature(frame["player2_stock"]),
                            "player2_character": _int64_feature(frame["player2_character"]),
                            "stage": _int64_feature(frame["stage"]),
                            "stock_winner": _int64_feature(died),
                            "game_winner": _int64_feature(game_winner),
                        }))
                        file_writer.write(newframe.SerializeToString())

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
