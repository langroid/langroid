from langroid.io.base import InputProvider, OutputProvider
import socketio
import threading
import re


def input_processor(message):
    outputs = []
    color_split = message.split("][")
    for i, cur in enumerate(color_split):
        if i > 0:
            cur = "[" + cur
        if i < len(color_split) - 1:
            cur = cur + "]"
        match = re.search(r"\[[^\]]+\]", cur)
        if match:
            color = match.group()[1:-1]
        else:
            color = "black"
        for x in cur.split("\n"):
            outputs.append([color, re.sub(r"\[[^\]]+\]", "", x)])

    messages = []
    for o in range(len(outputs)):
        messages.append(f"[{outputs[o][0].split(' ')[-1]}]{outputs[o][1]}")

    return messages


class WebSocketInputProvider(InputProvider):
    def __init__(self, name):
        super().__init__(name)
        self.returned_value = None
        self.sio = socketio.Client()

        @self.sio.on("message")
        def on_message(data):
            self.returned_value = data["text"]

        self.sio.connect("http://localhost:3000")

        threading.Thread(target=self.setup, daemon=True).start()

    def setup(self):
        try:
            self.sio.wait()
        except KeyboardInterrupt:
            self.sio.disconnect()

    def __call__(self, message, default=""):
        messages = input_processor(message)
        for m in messages:
            self.sio.emit("receiveMessage", m)
        while self.returned_value is None:
            pass
        returned_value = self.returned_value
        self.returned_value = None
        return returned_value


class WebSocketOutputProvider(OutputProvider):
    def __init__(self, name):
        super().__init__(name)
        self.returned_value = None
        self.streaming = False
        self.sio = socketio.Client()

        @self.sio.on("message")
        def on_message(data):
            self.returned_value = data["text"]

        self.sio.connect("http://localhost:3000")

        threading.Thread(target=self.setup, daemon=True).start()

    def setup(self):
        try:
            self.sio.wait()
        except KeyboardInterrupt:
            self.sio.disconnect()

    def handle_message(self, message, prefix):
        messages = input_processor(message)
        for m in messages:
            self.sio.emit("receiveMessage", f"{prefix}{m}")

    def __call__(self, message: str, streaming: bool = False):
        if streaming:
            if self.streaming:
                self.handle_message(message, "<s>")
            else:
                self.streaming = True
                self.handle_message(message, "")
        else:
            self.streaming = False
            self.handle_message(message, "")
