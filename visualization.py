#!/usr/bin/env python3
"""Visualization PoC for the SmashBot analysis"""
import argparse
import melee
import bar_chart_race as bcr
import pandas as pd

parser = argparse.ArgumentParser(description='AI-powered Melee in-game advantage bar')
parser.add_argument('--replayfile',
                    '-r',
                    help='SLP file to generate dataset on')
parser.add_argument('--datafile',
                    '-d',
                    help='Visualize data file')
args = parser.parse_args()

columns = ['Player 1','Player 2']

if args.replayfile:
    from model import AdvantageBarModel

    print("Reading SLP file...")
    console = melee.Console(is_dolphin=False, path=args.replayfile)
    console.connect()

    model = AdvantageBarModel()
    model.load()

    frames = []

    # Iterate through each frame of the game
    while True:
        gamestate = console.step()
        if gamestate is None:
            break
        else:
            # Save the data in this frame for later. We don't know the label yet
            prediction  = model.predict(gamestate)[0][0]
            frames.append([prediction, 1 - prediction])

    dataset = pd.DataFrame(frames, columns=('Player 1','Player 2'))
    print(dataset)
    dataset.to_csv('dataframe.csv', index=False)

if args.datafile:
    print("Making visualization...")
    dataset = pd.read_csv('dataframe.csv')
    # bcr.bar_chart_race(dataset)
    bcr.bar_chart_race(dataset,
                        'video.mp4',
                        figsize=(5, 3),
                        fixed_order=columns,
                        fixed_max=True,
                        steps_per_period=1,
                        period_length=1/60,
                        interpolate_period=False)
