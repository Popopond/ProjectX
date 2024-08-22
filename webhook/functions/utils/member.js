const axios = require('axios');
const FormData = require('form-data');

// Function to register user
async function registerUser(image, userId) {
  try {
    const form = new FormData();
    form.append('image', image, 'slip.jpg');
    form.append('userId', userId);

    const apiResponse = await axios.post('http://127.0.0.1:5000/register', form, {
      headers: {
        ...form.getHeaders(),
        'Content-Type': 'multipart/form-data'
      }
    });

    return apiResponse.data;
  } catch (error) {
    console.error('Error registering user:', error.message);
    throw error;
  }
}

// Function to check membership
async function checkMembership(userId) {
  try {
    const response = await axios.get('http://127.0.0.1:5000/check_membership', {
      params: { userId }
    });

    return response.data;
  } catch (error) {
    console.error('Error checking membership:', error.message);
    throw error;
  }
}

module.exports = { registerUser, checkMembership };
