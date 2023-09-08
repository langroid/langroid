const path = require("path");
const http = require("http");
const express = require("express");
const socketio = require("socket.io");
const Filter = require("bad-words");
const cors = require("cors");

const { generateMessage } = require("./utils/messages");

const app = express();
const server = http.createServer(app);
const io = socketio(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

require("dotenv").config();

const port = process.env.PORT || 3000;
const publicDirectoryPath = path.join(__dirname, "../public");

app.use(cors());
app.use(express.static(publicDirectoryPath));

io.on("connection", socket => {

  socket.on("sendMessage", (message, callback) => {
    io.emit("message", generateMessage(message));
    callback();
  });

  socket.on("receiveMessage", (message) => {
    io.emit("receive", generateMessage(message));
    //callback();
  });

  socket.on("disconnect", () => {

  });
});

server.listen(port, () => {
  console.log(`Server is up on port ${port}!`);
});
