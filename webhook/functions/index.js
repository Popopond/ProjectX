const axios = require('axios');
const FormData = require('form-data');
const line = require('./utils/line');
const gemini = require('./utils/gemini');

// คำสำคัญสำหรับการค้นหา
const keywords = ["ไข่", "เชื้อ", "ฟัก", "ไก่", "การตรวจสอบ", "การฟักไข่"];

// ฟังก์ชันสำหรับดึงข้อมูล LINE ID จาก LINE Bot API
async function getBotLineId(userId) {
  try {
    const response = await axios.get(`https://api.line.me/v2/bot/profile/${userId}`, {
      headers: {
        'Authorization': `Bearer {CHANNEL_ACCESS_TOKEN}`, // Replace {CHANNEL_ACCESS_TOKEN} with your actual channel access token
        'Content-Type': 'application/json'
      }
    });

    const data = response.data;
    return data.userId; // Assuming data.userId contains the correct LINE ID from the bot
  } catch (error) {
    console.error('Error fetching LINE Bot ID:', error);
    return null;
  }
}

// ฟังก์ชันเพื่อตรวจสอบสถานะสมาชิกและจำนวนการตรวจสอบ
const checkMembershipAndLimit = async (userId) => {
  try {
    const response = await axios.get(`http://127.0.0.1:5000/check_membership/${userId}`);
    return response.data; // คืนค่าเป็นสถานะสมาชิกและจำนวนการตรวจสอบ
  } catch (error) {
    console.error('Error checking membership:', error.message);
    return { membership_status: 'non-member', check_count: 0 }; // คืนค่าหากเกิดข้อผิดพลาด
  }
};

// ฟังก์ชันการตรวจสอบว่ามีสมาชิกในระบบหรือไม่
const checkIfUserExists = async (userId) => {
  try {
    const response = await axios.get(`http://127.0.0.1:5000/check_membership/${userId}`);
    return response.data; // คืนค่าเป็นสถานะสมาชิกและจำนวนการตรวจสอบ
  } catch (error) {
    console.error('Error checking if user exists:', error.message);
    return null; // หากเกิดข้อผิดพลาดคืนค่า null
  }
};

// ฟังก์ชันสมัครหรืออัปเดตข้อมูลสมาชิก
const registerOrUpdateUser = async (userId, profileData) => {
  try {
    // ตรวจสอบว่าผู้ใช้มีอยู่ในระบบหรือไม่
    const existingUser = await checkIfUserExists(userId);

    if (existingUser) {
      // หากผู้ใช้มีอยู่แล้ว ให้ทำการอัปเดตข้อมูล
      const updateResponse = await axios.put(`http://127.0.0.1:5000/update_user/${userId}`, profileData);
      console.log('User updated:', updateResponse.data);
      return updateResponse.data;
    } else {
      // หากผู้ใช้ยังไม่อยู่ในระบบ ให้ทำการสมัครสมาชิกใหม่
      const createResponse = await axios.post(`http://127.0.0.1:5000/create_user`, profileData);
      console.log('User created:', createResponse.data);
      return createResponse.data;
    }
  } catch (error) {
    console.error('Error registering or updating user:', error.message);
  }
};

// ฟังก์ชันการประมวลผลภาพ
const processImage = async (imageId, replyToken, userId) => {
  try {
    const membershipData = await checkMembershipAndLimit(userId);

    // ตรวจสอบสถานะสมาชิกและจำนวนการตรวจสอบ
    if (membershipData.membership_status === 'non-member' && membershipData.check_count >= 5) {
      await line.reply(replyToken, {
        type: 'text',
        text: `คุณได้ใช้สิทธิ์ตรวจสอบไข่ครบจำนวนแล้ว กรุณาสมัครสมาชิกเพื่อใช้บริการเพิ่มเติม\nลงทะเบียนที่นี่: https://rust-paint-muscle.glitch.me/`
      });
      return;
    }

    console.log(`Downloading image with ID: ${imageId}`);
    const response = await line.getImageBinary(imageId);

    const form = new FormData();
    form.append('image', response, 'image.jpg');
    form.append('userId', userId.toString());

    const apiResponse = await axios.post('http://127.0.0.1:5000/predict', form, {
      headers: {
        ...form.getHeaders(),
        'Content-Type': 'multipart/form-data'
      }
    });

    if (apiResponse.status === 200) {
      const predictions = apiResponse.data.predictions;
      const predictionText = predictions.map(pred => `${pred.status} (${pred.accuracy})`).join(', ');

      await line.reply(replyToken, {
        type: 'text',
        text: `ผลลัพธ์: ${predictionText}`
      });
    }
  } catch (error) {
    if (error.response) {
      switch (error.response.status) {
        case 403:
          await line.reply(replyToken, {
            type: 'text',
            text: `คุณได้ใช้บริการตรวจสอบไข่ครบแล้ว กรุณาลงทะเบียนสมาชิกเพื่อใช้บริการต่อ\nลงทะเบียนที่นี่: https://rust-paint-muscle.glitch.me/`
          });
          break;
        case 404:
          await line.reply(replyToken, {
            type: 'text',
            text: `ไม่พบข้อมูลสมาชิกในระบบ กรุณาลงทะเบียนใหม่`
          });
          break;
        default:
          console.error('Error processing image:', error.message);
      }
    } else {
      console.error('Error processing image:', error.message);
    }
  }
};


// ฟังก์ชันจัดการข้อความ
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


// ฟังก์ชัน webhook
const { onRequest } = require('firebase-functions/v2/https');
const { error } = require('firebase-functions/logger');

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
