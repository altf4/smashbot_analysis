#!/usr/bin/env python3
import melee
import argparse
import os
import progressbar
import pathlib

parser = argparse.ArgumentParser(description='Prune your SLP dataset by finding invalid games')
parser.add_argument('--files',
                    '-f',
                    help='Directory of SLP files')
parser.add_argument('--delete', '-d', action='store_true',
                    help='Delete files found by the script',
                    default=False)
args = parser.parse_args()

def record_bad_file(bad_file):
    with open("bad_files.txt", "a", encoding="utf-8") as f:
        f.write("%s\n" % bad_file.path)

if args.delete:
    with open("bad_files.txt", "r") as ifile:
        for line in ifile:
            os.remove(line.strip())
            print(line)

if args.files:
    num_files = len([f for f in os.listdir(args.files)if os.path.isfile(os.path.join(args.files, f))])
    bar = progressbar.ProgressBar(maxval=num_files)

    for entry in bar(os.scandir(args.files)):
        console = None
        try:
            console = melee.Console(is_dolphin=False, path=entry.path, allow_old_version=True)
        except Exception as ex:
            record_bad_file(entry)
            continue
        try:
            console.connect()
        except Exception as ex:
            record_bad_file(entry)
            continue
        ports = None
        percents = (0,0)
        damage = 0

        try:
            # Iterate through each frame of the game
            frame_count = 0
            while True:
                gamestate = console.step()
                # if game is over
                if gamestate is None:
                    if frame_count < 1800:
                        record_bad_file(entry)
                        break
                    if damage < 100:
                        # Probably a handwarmer if <100 damage dealt
                        record_bad_file(entry)
                        break
                    break
                else:
                    # Only do this once per game
                    if ports is None:
                        ports = []
                        for port, _ in gamestate.player.items():
                            ports.append(port)
                        if len(ports) != 2:
                            record_bad_file(entry)
                            break
                        ports = tuple(ports)

                    if frame_count > 1800 and damage > 100:
                        # Quit early. This game is good
                        break

                    # Player one and two, but not necessarily those ports
                    player_one = gamestate.player[ports[0]]
                    player_two = gamestate.player[ports[1]]

                    if gamestate.stage not in [melee.Stage.BATTLEFIELD, melee.Stage.FINAL_DESTINATION,
                                                melee.Stage.DREAMLAND, melee.Stage.FOUNTAIN_OF_DREAMS,
                                                melee.Stage.YOSHIS_STORY, melee.Stage.POKEMON_STADIUM]:
                        record_bad_file(entry)
                        break

                    if player_one.is_cpu or player_two.is_cpu:
                        record_bad_file(entry)
                        break

                    if percents[0] < player_one.percent:
                        damage += player_one.percent - percents[0]
                    if percents[1] < player_two.percent:
                        damage += player_two.percent - percents[1]

                    percents = (player_one.percent, player_two.percent)

                    frame_count += 1

        except Exception as ex:
            record_bad_file(entry)
            continue
