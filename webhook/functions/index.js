const axios = require('axios');
const FormData = require('form-data');
const line = require('./utils/line');
const gemini = require('./utils/gemini');

const keywords = ["ไข่", "เชื้อ", "ฟัก", "ไก่", "การตรวจสอบ", "การฟักไข่"];

async function processImage(imageId, replyToken, userId) {
  try {
    console.log(`Downloading image with ID: ${imageId}`);
    const response = await line.getImageBinary(imageId);

    console.log(`Image downloaded, sending to API`);

    const form = new FormData();
    form.append('image', response, 'image.jpg');
    form.append('userId', userId);

    const apiResponse = await axios.post('http://127.0.0.1:5000/predict', form, {
      headers: {
        ...form.getHeaders(),
        'Content-Type': 'multipart/form-data'
      }
    });

    if (apiResponse.status === 200) {
      const prediction = apiResponse.data.prediction;

      console.log(`Sending reply with prediction: ${prediction}`);
      
      await line.reply(replyToken, {
        type: 'text',
        text: `ผลลัพธ์: ${prediction}`
      });

      console.log(`Reply sent with prediction: ${prediction}`);
    } else {
      throw new Error(`API responded with status: ${apiResponse.status}`);
    }
  } catch (error) {
    console.error('Error processing image:', error.message);
    await line.reply(replyToken, {
      type: 'text',
      text: 'มีข้อผิดพลาดในการประมวลผลรูปภาพ'
    });
  }
}

async function handleTextMessage(text, replyToken, userId) {
  const hasKeyword = keywords.some(keyword => text.includes(keyword));

  if (hasKeyword) {
    const response = await gemini.textOnly(text);
    await line.reply(replyToken, {
      type: 'text',
      text: response
    });
  } else {
    await line.reply(replyToken, {
      type: 'text',
      text: 'ขอโทษค่ะ ฉันสามารถตอบคำถามเกี่ยวกับไข่ไก่เท่านั้น'
    });
  }
}

const { onRequest } = require('firebase-functions/v2/https');

exports.webhook = onRequest(async (req, res) => {
  if (req.method === 'POST') {
    try {
      const events = req.body.events;
      for (const event of events) {
        const userId = event.source.userId; // ดึง userId จาก event
        switch (event.type) {
          case 'message':
            if (event.message.type === 'image') {
              await processImage(event.message.id, event.replyToken, userId);
            } else if (event.message.type === 'text') {
              await handleTextMessage(event.message.text, event.replyToken, userId);
            }
            break;
          default:
            console.warn('Unhandled event type:', event.type);
        }
      }
      res.status(200).send('Events processed');
    } catch (error) {
      console.error('Error processing events:', error.message);
      res.status(500).send('Internal Server Error');
    }
  } else {
    res.status(405).send('Method Not Allowed');
  }
});
