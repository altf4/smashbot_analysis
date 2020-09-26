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
parser.add_argument('--max_split',
                    '-m',
                    help='Split building dataset into this many pieces')
parser.add_argument('--split',
                    '-s',
                    help='Handle split number S')
args = parser.parse_args()

def who_died(past_p1, past_p2, current_p1, current_p2):
    """Returns who died."""
    if past_p1 > current_p1:
        return 0
    if past_p2 > current_p2:
        return 1
    return -1

def _float_feature(value):
  return tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=[x])) for x in value])

def _int64_feature(value):
  return tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=[x])) for x in value])

if args.build:
    """Builds the tfrecord dataset

    We can't use raw SLP files with tensorflow, we need to parse the files,
    extract the important information from them, and then store then in
    .tfrecord files that integrate well with tf.data. That's what this step does.

    Input:
        SLP files located in training_data/
            (Flat structure. No nested folders)
    Output:
        .tfrecord files located in tfrecords/ folder
    """
    print("Building dataset...")

    max_split, split = 1, 0
    if args.max_split and args.split:
        max_split, split = int(args.max_split), int(args.split)

    directory = 'training_data/'
    num_files = len([f for f in os.listdir(directory)if os.path.isfile(os.path.join(directory, f))])
    bar = progressbar.ProgressBar(maxval=num_files)
    file_index = 0
    for entry in bar(os.scandir(directory)):
        file_index += 1
        if file_index % max_split != split:
            continue

        frames = {
            "player1_character": [],
            "player1_x": [],
            "player1_y": [],
            "player1_percent": [],
            "player1_stock": [],
            "player1_action": [],
            "player2_x": [],
            "player2_y": [],
            "player2_percent": [],
            "player2_stock": [],
            "player2_action": [],
            "player2_character": [],
            "stage": [],
            "frame": [],
            "stock_winner": [],
            "game_winner": []
        }
        if entry.path.endswith(".slp") and entry.is_file():
            console = None
            try:
                console = melee.Console(is_dolphin=False, path=entry.path, allow_old_version=True)
            except Exception as ex:
                print("Got error, skipping file", ex)
                continue
            try:
                console.connect()
            except ubjson.decoder.DecoderException as ex:
                print("Got error, skipping file", ex)
                continue
            stocks = (4,4)
            ports = None
            # Pick a game to be part of the evaluation set 20% of the time
            is_evaluation = random.random() > 0.8

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
                        # Only do this once per game
                        if ports is None:
                            ports = []
                            for port, _ in gamestate.player.items():
                                ports.append(port)
                            if len(ports) != 2:
                                break
                            ports = tuple(ports)

                        # Player one and two, but not necessarily those ports
                        player_one = gamestate.player[ports[0]]
                        player_two = gamestate.player[ports[1]]

                        # Put the frame's data in. We don't know the label yet though
                        frames["player1_character"].append(player_one.character.value)
                        frames["player1_x"].append(player_one.x)
                        frames["player1_y"].append(player_one.y)
                        frames["player1_percent"].append(player_one.percent)
                        frames["player1_stock"].append(player_one.stock)
                        frames["player1_action"].append(player_one.action.value)
                        frames["player2_x"].append(player_two.x)
                        frames["player2_y"].append(player_two.y)
                        frames["player2_percent"].append(player_two.percent)
                        frames["player2_stock"].append(player_two.stock)
                        frames["player2_action"].append(player_two.action.value)
                        frames["player2_character"].append(player_one.character.value)
                        frames["stage"].append(AdvantageBarModel.stage_flatten(gamestate.stage.value))
                        frames["frame"].append(gamestate.frame)

                        # Did someone lose a stock? Add the labels on
                        died = who_died(stocks[0], stocks[1], player_one.stock, player_two.stock)
                        if died > -1:
                            frames_needed = len(frames["frame"]) - len(frames["stock_winner"])
                            frames["stock_winner"].extend([died] * frames_needed)
                        stocks = (player_one.stock, player_two.stock)

            except melee.console.SlippiVersionTooLow as ex:
                print("Slippi version too low", ex)
            except Exception as ex:
                print("Error processing file", ex)
                continue

            filename = None
            if is_evaluation:
                filename = "tfrecords/eval" / pathlib.Path(pathlib.Path(entry.path + ".tfrecord").name)
            else:
                filename = "tfrecords/train" /  pathlib.Path(pathlib.Path(entry.path + ".tfrecord").name)
            # Must be >30 seconds in the match and have a winner
            if len(frames) > 1800 and game_winner > -1:
                with tf.io.TFRecordWriter(str(filename)) as file_writer:
                    # This is all that we actually have full data for
                    data_cap = len(frames["stock_winner"])

                    print("\n", data_cap)

                    # "Context" features are static for the whole data record. Not in the time series
                    context_features = tf.train.Features(feature={
                        "game_winner": tf.train.Feature(float_list=tf.train.FloatList(value=[game_winner])),
                        "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[data_cap]))
                    })

                    features = {
                        "player1_character": _int64_feature(frames["player1_character"][:data_cap]),
                        "player1_x": _float_feature(frames["player1_x"][:data_cap]),
                        "player1_y": _float_feature(frames["player1_y"][:data_cap]),
                        "player1_percent": _float_feature(frames["player1_percent"][:data_cap]),
                        "player1_stock": _float_feature(frames["player1_stock"][:data_cap]),
                        "player1_action": _int64_feature(frames["player1_action"][:data_cap]),
                        "player2_x": _float_feature(frames["player2_x"][:data_cap]),
                        "player2_y": _float_feature(frames["player2_y"][:data_cap]),
                        "player2_percent": _float_feature(frames["player2_percent"][:data_cap]),
                        "player2_stock": _float_feature(frames["player2_stock"][:data_cap]),
                        "player2_action": _int64_feature(frames["player2_action"][:data_cap]),
                        "player2_character": _int64_feature(frames["player2_character"][:data_cap]),
                        "stage": _int64_feature(frames["stage"][:data_cap]),
                        "frame": _float_feature(frames["frame"][:data_cap]),
                        "stock_winner": _float_feature(frames["stock_winner"][:data_cap]),
                    }
                    newexample = tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list=features), context=context_features)
                    file_writer.write(newexample.SerializeToString())

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
                            is_dolphin=False)
    # Run the console
    console.run()

    # Connect to the console
    print("Connecting to console...")
    if not console.connect():
        print("ERROR: Failed to connect to the console.")
        sys.exit(-1)
    print("...Connected")

    # Main loop
    while True:
        # "step" to the next frame
        gamestate = console.step()
        if gamestate is None:
            break
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            print(model.predict(gamestate))
