#!/usr/bin/env python3
import tornado.ioloop
import tornado.web
import os
import sys
import time
import melee
from model import AdvantageBarModel

model = None
console = None
gamestate = None

class MainHandler(tornado.web.RequestHandler):
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
        (r"/frame", MainHandler),
     ], **settings)

def your_function():
    tornado.ioloop.IOLoop.current().call_later(delay=1, callback=your_function)
    global gamestate
    gamestate = console.step()
    # Keep grabbing gamestates until it returns None. So we know it's the latest
    while gamestate:
        new_gamestate = console.step()
        if new_gamestate is None:
            break
        gamestate = new_gamestate


if __name__ == "__main__":
    console = melee.Console(path="/home/altf4/Code/Ishiiruka/build/Binaries/",
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
    tornado.ioloop.IOLoop.current().call_later(delay=1/60, callback=your_function)
    tornado.ioloop.IOLoop.current().start()
