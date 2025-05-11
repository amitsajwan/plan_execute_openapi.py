// script.js
const messagesDiv = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');
const graphJsonViewPre = document.getElementById('graphJsonView').querySelector('pre');
const sendButton = document.getElementById('sendButton');
const thinkingIndicator = document.getElementById('thinkingIndicator');
let ws;

function showThinking(show) {
    thinkingIndicator.style.display = show ? 'inline-block' : 'none';
    sendButton.disabled = show;
    messageInput.disabled = show;
}

function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') {
        try {
            return JSON.stringify(unsafe, null, 2); // Fallback for non-strings
        } catch (e) {
            return String(unsafe); // Ultimate fallback
        }
    }
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatMessageContent(contentStr, type) {
    if (typeof contentStr !== 'string') {
        return '<pre>' + escapeHtml(JSON.stringify(contentStr, null, 2)) + '</pre>';
    }

    // Apply rich formatting only for 'final' or 'agent' (default) message types
    if (type !== 'agent' && type !== 'final') {
        return escapeHtml(contentStr); // Basic escaping for other types
    }
    
    let html = '';
    const codeBlockPlaceholders = [];
    
    // Pre-process to handle fenced code blocks
    // Regex for Python multi-line strings: escape backslashes for Python, then JS will see them.
    // So, in JS, we use single backslashes for \n, \w, \s etc.
    let processedContentStr = contentStr.replace(/```(\w*)\n([\s\S]*?)\n```/g, (match, lang, code) => {
        const placeholder = `__CODEBLOCK_${codeBlockPlaceholders.length}__`;
        const langClass = lang ? `language-${escapeHtml(lang)}` : 'language-plaintext';
        codeBlockPlaceholders.push(`<pre><code class="${langClass}">${escapeHtml(code)}</code></pre>`);
        return placeholder;
    });
    processedContentStr = processedContentStr.replace(/```\n([\s\S]*?)\n```/g, (match, code) => {
        const placeholder = `__CODEBLOCK_${codeBlockPlaceholders.length}__`;
        codeBlockPlaceholders.push(`<pre><code>${escapeHtml(code)}</code></pre>`);
        return placeholder;
    });

    const lines = processedContentStr.split('\n'); // Split by actual newline character in JS string
    let inList = false;
    let paragraphBuffer = [];

    function flushParagraph() {
        if (paragraphBuffer.length > 0) {
            html += `<p>${paragraphBuffer.join('<br>')}</p>`; // Use <br> for multi-line paragraphs
            paragraphBuffer = [];
        }
    }

    for (let i = 0; i < lines.length; i++) {
        let line = lines[i]; 

        if (line.startsWith("### ")) {
            flushParagraph(); if (inList) { html += '</ul>'; inList = false; }
            html += `<h4>${escapeHtml(line.substring(4))}</h4>`;
        } else if (line.startsWith("## ")) {
            flushParagraph(); if (inList) { html += '</ul>'; inList = false; }
            html += `<h5>${escapeHtml(line.substring(3))}</h5>`;
        } else if (line.startsWith("# ")) {
            flushParagraph(); if (inList) { html += '</ul>'; inList = false; }
            html += `<h6>${escapeHtml(line.substring(2))}</h6>`;
        } else if (line.startsWith("- ") || line.startsWith("* ")) {
            flushParagraph();
            if (!inList) {
                html += '<ul>';
                inList = true;
            }
            html += `<li>${escapeHtml(line.substring(line.startsWith("- ") ? 2 : 1).trim())}</li>`;
        } else if (line.startsWith("__CODEBLOCK_")) { 
            flushParagraph(); if (inList) { html += '</ul>'; inList = false; }
            const placeholderIndex = parseInt(line.substring("__CODEBLOCK_".length, line.lastIndexOf("__")));
            if (placeholderIndex >= 0 && placeholderIndex < codeBlockPlaceholders.length) {
                html += codeBlockPlaceholders[placeholderIndex];
            } else {
                html += escapeHtml(line); 
            }
        } else { 
            if (inList && line.trim() !== "" && !line.trim().startsWith("- ") && !line.trim().startsWith("* ")) {
                html += '</ul>'; inList = false;
            }
            
            if (line.trim() === "") {
                flushParagraph(); 
            } else {
                paragraphBuffer.push(escapeHtml(line)); 
            }
        }
    }
    flushParagraph(); 
    if (inList) html += '</ul>'; 
    
    return html || "<p>" + escapeHtml(contentStr) + "</p>";
}

function addChatMessage(message, type) {
    const messageElement = document.createElement('div');
    messageElement.innerHTML = message; 
    messageElement.className = 'message ' + type;
    messagesDiv.appendChild(messageElement);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Define sendMessage in the global scope or ensure it's defined before use
window.sendMessage = function() { 
    if (ws && ws.readyState === WebSocket.OPEN) {
        const messageText = messageInput.value;
        if (messageText.trim() === "") return;
        addChatMessage("You: " + escapeHtml(messageText), "user");
        ws.send(messageText);
        messageInput.value = '';
        messageInput.style.height = 'auto'; 
        showThinking(true);
    } else {
        addChatMessage("WebSocket is not connected.", "error");
    }
}

function connect() {
    const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = wsProtocol + "//" + location.host + "/ws/openapi_agent";
    addChatMessage("Connecting to: " + wsUrl, "info");
    ws = new WebSocket(wsUrl);

    ws.onopen = () => { addChatMessage("WebSocket connected.", "info"); showThinking(false); };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        let content = data.content;
        const type = data.type || "agent";

        if (type === "graph_update") {
            graphJsonViewPre.textContent = JSON.stringify(content, null, 2);
            addChatMessage("Execution graph has been updated.", "info");
            console.log("Graph Update Received:", content);
            return; 
        }

        if (type === "status" && content && content.toLowerCase().includes("processing")) {
            showThinking(true);
        } else if (type === "final" || type === "error" || type === "info" || type === "warning") {
            showThinking(false);
        }
        
        content = formatMessageContent(content, type);
        addChatMessage(`Agent (${type}): ${content}`, type);
    };

    ws.onerror = (error) => { 
        addChatMessage("WebSocket error. Check console. If page HTTPS, WS must be WSS.", "error"); 
        console.error("WebSocket error object:", error);
        showThinking(false);
    };
    ws.onclose = (event) => { 
        let reason = "";
        if (event.code) reason += `Code: ${event.code} `;
        if (event.reason) reason += `Reason: ${event.reason} `;
        if (event.wasClean) reason += `(Clean close) `; else reason += `(Unclean close) `;
        addChatMessage("WebSocket disconnected. " + reason + "Attempting to reconnect in 5s...", "info");
        console.log("WebSocket close event:", event);
        showThinking(false);
        setTimeout(connect, 5000);
    };
}

messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

messageInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) { 
        e.preventDefault(); 
        sendMessage(); 
    }
});

// Call connect to establish WebSocket connection
connect();
</script>
</body>
</html>
