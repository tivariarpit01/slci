// --- Filename: script.js ---

document.addEventListener("DOMContentLoaded", () => {
    const sendBtn = document.getElementById("send-btn");
    const userInput = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const categoryBtnContainer = document.getElementById("category-buttons");
    const chatBtns = document.querySelectorAll(".chat-btn");
    
    // Get the refresh button
    const refreshBtn = document.getElementById("refresh-btn");

    // 'let' allows us to change it
    let sessionId;

    // --- Event Listeners ---
    sendBtn.addEventListener("click", sendChatMessage);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendChatMessage();
        }
    });

    chatBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            userInput.value = btn.innerText;
            sendChatMessage();
        });
    });

    // Refresh button listener
    refreshBtn.addEventListener("click", startNewSession);

    
    // --- Function to start/reset the chat ---
    function startNewSession() {
        // 1. Clear the chat box
        chatBox.innerHTML = "";
        
        // 2. Create a new session ID
        sessionId = "session_" + Date.now();
        
        // 3. Show category buttons again
        if (categoryBtnContainer) {
            categoryBtnContainer.style.display = "flex";
        }
        
        // 4. Add the initial bot message
        addMessageToChatbox("Hello! I'm the SLCI assistant. How can I help you today?", "bot");
    }

    // --- Chat Functions ---
    async function sendChatMessage() {
        const userText = userInput.value.trim();
        if (userText === "") return;
        
        // Hide categories after first user message
        if (categoryBtnContainer) {
            categoryBtnContainer.style.display = "none";
        }

        addMessageToChatbox(userText, "user");
        userInput.value = "";
        addMessageToChatbox("Typing...", "bot", "typing-indicator");

        try {
            const response = await fetch("http://127.0.0.1:5000/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    "message": userText,
                    "session_id": sessionId // Uses the current sessionId
                }),
            });

            if (!response.ok) throw new Error("API request failed");
            const data = await response.json();
            
            removeTypingIndicator();
            addMessageToChatbox(data.reply, "bot");

        } catch (error) {
            console.error("Chat Error:", error);
            removeTypingIndicator();
            addMessageToChatbox("Sorry, I'm having trouble connecting. Please try again.", "bot");
        }
    }

    function addMessageToChatbox(text, sender, id = null) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", `${sender}-message`);
        messageElement.innerText = text;
        if (id) messageElement.id = id;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function removeTypingIndicator() {
        const indicator = document.getElementById("typing-indicator");
        if (indicator) chatBox.removeChild(indicator);
    }
    
    // --- Start the chat when the page loads ---
    startNewSession();
});