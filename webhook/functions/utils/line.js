const axios = require("axios");

const LINE_HEADER = {
  "Content-Type": "application/json",
  Authorization: `Bearer ${process.env.LINE_CHANNEL_ACCESS_TOKEN}`
};

// Function to get image binary from LINE
const getImageBinary = async (messageId) => {
  const originalImage = await axios({
    method: "get",
    headers: LINE_HEADER,
    url: `https://api-data.line.me/v2/bot/message/${messageId}/content`,
    responseType: "arraybuffer"
  });
  return originalImage.data;
};

// Function to send reply message to LINE
const reply = (token, payload) => {
  return axios({
    method: "post",
    url: "https://api.line.me/v2/bot/message/reply",
    headers: LINE_HEADER,
    data: {
      replyToken: token,
      messages: [payload]
    }
  });
};

module.exports = { getImageBinary, reply };
