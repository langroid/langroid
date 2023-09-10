
function extractValue(str) {
    const pattern = /^\[([^\]]+)\]/; // Matches strings starting with [some_content]
    str = str.replace(/^\s+|\s+$/g, ''); // Remove all whitespace (including new line characters
    const match = str.match(pattern);
    if (match) {
        return {"str":str.replace(match[0],""),"color":match[1]}; // The captured content inside the square brackets
    } else {
        return {"str":str,"color":"black"};
    }
}

const generateMessage = (text) => {
    let stream = false;
    if (text.startsWith("<s>")){
      text = text.slice(3);
      stream = true;
  }
  out = extractValue(text);
  return {
    "text": out.str,
    "color": out.color,
    "stream": stream
  };
};

module.exports = {
  generateMessage
};
