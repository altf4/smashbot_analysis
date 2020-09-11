#!/usr/bin/env python3
import tornado.ioloop
import tornado.web
import os
import sys
import time
import melee
import argparse
from model import AdvantageBarModel

parser = argparse.ArgumentParser(description='SmashBot Analysis UI')
parser.add_argument('--dolphin_path',
                    '-e',
                    help='Visualization Mode')
args = parser.parse_args()


model = None
console = None
gamestate = None

class FrameHandler(tornado.web.RequestHandler):
    def get(self):
        global model
        global gamestate
        predictions = [[0.5, 0.5]]
        if gamestate:
            predictions = model.predict(gamestate)
        self.write(str(predictions[0][0]))

def make_app():
    settings = {
        "static_path": os.path.join(os.path.dirname(__file__), "static"),
        "xsrf_cookies": False,
    }
    return tornado.web.Application([
        (r"/frame", FrameHandler),
     ], **settings)

def refresh_prediction():
    # Schedule this function to be called again in half a second
    #   Effectively runs it twice per second
    tornado.ioloop.IOLoop.current().call_later(delay=1/2, callback=refresh_prediction)
    global gamestate
    gamestate = console.step()
    # Keep grabbing gamestates until it returns None. So we know it's the latest
    #   We're using polling mode, so we need to
    while gamestate:
        new_gamestate = console.step()
        if new_gamestate is None:
            break
        gamestate = new_gamestate


if __name__ == "__main__":
    console = melee.Console(path=args.dolphin_path,
                            slippi_address="127.0.0.1",
                            slippi_port=51441,
                            blocking_input=False,
                            polling_mode=True,
                            logger=None)

    print("Connecting to console...")
    console.run()

    model = AdvantageBarModel()
    model.load()

    time.sleep(2)

    if not console.connect():
        print("ERROR: Failed to connect to the console.")
        sys.exit(-1)
    print("Connected!")

    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().call_later(delay=1/60, callback=refresh_prediction)
    tornado.ioloop.IOLoop.current().start()
