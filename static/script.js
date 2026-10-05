const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send-btn");
const chatBox = document.getElementById("chat-box");
const typing = document.getElementById("typing");
const clearButton = document.getElementById("clear-btn");


function addMessage(message, sender) {

    const messageDiv = document.createElement("div");

    messageDiv.classList.add(
        "message",
        sender === "user"
            ? "user-message"
            : "assistant-message"
    );

    const avatar = document.createElement("div");
    avatar.classList.add("avatar");

    avatar.textContent =
        sender === "user" ? "You" : "AI";


    const content = document.createElement("div");
    content.classList.add("message-content");


    const name = document.createElement("div");
    name.classList.add("message-name");

    name.textContent =
        sender === "user"
            ? "You"
            : "College Assistant";


    const text = document.createElement("div");
    text.classList.add("message-text");

    text.textContent = message;


    content.appendChild(name);
    content.appendChild(text);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);

    chatBox.appendChild(messageDiv);

    chatBox.scrollTop = chatBox.scrollHeight;
}


async function askQuestion() {

    const question = questionInput.value.trim();

    if (!question) {
        return;
    }


    // Add user message
    addMessage(question, "user");


    // Clear input
    questionInput.value = "";


    // Show loading
    typing.classList.remove("hidden");
    sendButton.disabled = true;


    try {

        const response = await fetch("/ask", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            })
        });


        const data = await response.json();


        if (!response.ok) {

            addMessage(
                data.answer || "Something went wrong.",
                "assistant"
            );

            return;
        }


        addMessage(
            data.answer,
            "assistant"
        );


    } catch (error) {

        console.error(error);

        addMessage(
            "Unable to connect to the server. Please make sure the Flask application is running.",
            "assistant"
        );

    } finally {

        typing.classList.add("hidden");
        sendButton.disabled = false;
        questionInput.focus();

    }
}


// Send button
sendButton.addEventListener(
    "click",
    askQuestion
);


// Enter key
questionInput.addEventListener(
    "keydown",
    function(event) {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            askQuestion();
        }

    }
);


// Clear conversation
clearButton.addEventListener(
    "click",
    function() {

        chatBox.innerHTML = "";

        addMessage(
            "Conversation cleared. Ask me a new question! 👋",
            "assistant"
        );

    }
);
