from langroid.io.base import InputProvider, OutputProvider
import socketio
import threading

class WebSocketInputProvider(InputProvider):
    def __init__(self, name):
        super().__init__(name)
        self.returned_value = None
        self.sio = socketio.Client()

        @self.sio.on('message')
        def on_message(data):
            self.returned_value = data["text"]

        self.sio.connect('http://localhost:3000')

        threading.Thread(target=self.setup, daemon=True).start()

    def setup(self):
        try:
            self.sio.wait()
        except KeyboardInterrupt:
            self.sio.disconnect()

    def __call__(self, message, default=""):
        self.sio.emit('receiveMessage', message)
        while self.returned_value is None:
            pass
        returned_value = self.returned_value
        self.returned_value = None
        return returned_value

class WebSocketOutputProvider(OutputProvider):
    def __init__(self, name):
        super().__init__(name)
        self.returned_value = None
        self.sio = socketio.Client()

        @self.sio.on('message')
        def on_message(data):
            self.returned_value = data["text"]

        self.sio.connect('http://localhost:3000')

        threading.Thread(target=self.setup, daemon=True).start()

    def setup(self):
        try:
            self.sio.wait()
        except KeyboardInterrupt:
            self.sio.disconnect()

    def __call__(self, message: str):
        self.sio.emit('receiveMessage', message)