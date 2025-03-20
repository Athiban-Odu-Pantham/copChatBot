// script.js

document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', function(e) {
  if (e.key === 'Enter') {
    sendMessage();
  }
});

async function sendMessage() {
  const inputBox = document.getElementById('user-input');
  const userText = inputBox.value.trim();
  if (userText === "") return;
  
  // Display the user's message
  appendMessage('user', userText);
  inputBox.value = "";
  
  // Get the bot response from the Flask backend
  const responseText = await getBotResponse(userText);
  appendMessage('bot', responseText);
}

function appendMessage(sender, text) {
  const chatWindow = document.getElementById('chat-window');
  const messageDiv = document.createElement('div');
  messageDiv.classList.add('chat-message');
  messageDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
  messageDiv.innerText = text;
  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function getBotResponse(input) {
  try {
    const response = await fetch(`/query?query=${encodeURIComponent(input)}`);
    if (!response.ok) {
      return "I'm sorry, I couldn't find any legal details for that query.";
    }
    const data = await response.json();
    return `Description: ${capitalize(data.description)}\nOffense: ${data.offense}\nPunishment: ${data.punishment}\nSection: ${data.section}`;
  } catch (error) {
    console.error('Error:', error);
    return "There was an error processing your request.";
  }
}

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}
