// --- Filename: script.js ---

document.addEventListener("DOMContentLoaded", () => {
    const sendBtn = document.getElementById("send-btn");
    const userInput = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");

    // --- CRITICAL ---
    // Create a unique session ID for this user's conversation
    // This tells LangGraph to use its memory for this specific user.
    const sessionId = "session_" + Date.now();

    // Add initial bot message
    addMessageToChatbox("Hello! I'm the SLCI assistant. How can I help you today?", "bot");

    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    });

    async function sendMessage() {
        const userText = userInput.value.trim();
        if (userText === "") return;

        // 1. Display user's message
        addMessageToChatbox(userText, "user");
        userInput.value = ""; // Clear input

        try {
            // 2. Send message to the Flask API
            const response = await fetch("http://127.0.0.1:5000/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    "message": userText,
                    "session_id": sessionId // Send the unique session ID
                }),
            });

            if (!response.ok) {
                throw new Error("API request failed");
            }

            const data = await response.json();
            const botReply = data.reply;

            // 3. Display bot's reply
            addMessageToChatbox(botReply, "bot");

        } catch (error) {
            console.error("Error:", error);
            addMessageToChatbox("Sorry, I'm having trouble connecting. Please try again.", "bot");
        }
    }

    function addMessageToChatbox(text, sender) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", `${sender}-message`);
        messageElement.innerText = text;
        chatBox.appendChild(messageElement);
        // Auto-scroll to the bottom
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});