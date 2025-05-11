// script.js
document.addEventListener('DOMContentLoaded', () => {
    const messagesDiv = document.getElementById('messages');
    const messageInput = document.getElementById('messageInput');
    const graphJsonViewPre = document.getElementById('graphJsonViewPre');
    const mermaidDagContainer = document.getElementById('mermaidDagDiagram');
    const sendButton = document.getElementById('sendButton');
    const thinkingIndicator = document.getElementById('thinkingIndicator');
    
    const jsonTabButton = document.getElementById('jsonTabButton');
    const dagTabButton = document.getElementById('dagTabButton');
    const graphJsonViewContent = document.getElementById('graphJsonViewContent');
    const graphDagViewContent = document.getElementById('graphDagViewContent');

    let ws;
    let currentGraphData = null; // Store the latest graph data

    function showThinking(show) {
        thinkingIndicator.style.display = show ? 'inline-block' : 'none';
        if (sendButton) sendButton.disabled = show;
        if (messageInput) messageInput.disabled = show;
    }

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {
            try {
                return JSON.stringify(unsafe, null, 2); 
            } catch (e) {
                return String(unsafe); 
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

        if (type !== 'agent' && type !== 'final') {
            return escapeHtml(contentStr); 
        }
        
        let html = '';
        const codeBlockPlaceholders = [];
        
        // Pre-process to handle fenced code blocks
        // Regex for Python multi-line strings: Python's \n becomes \\n in JSON string, then \n in JS string.
        // So, in JS, we use single backslashes for \n, \w, \s etc.
        let processedContentStr = contentStr.replace(/```(\w*)\n([\s\S]*?)\n```/g, (match, lang, code) => {
            const placeholder = `__CODEBLOCK_${codeBlockPlaceholders.length}__`;
            const langClass = lang ? `language-${escapeHtml(lang)}` : 'language-plaintext';
            // Ensure code itself is escaped before putting into placeholder
            codeBlockPlaceholders.push(`<pre><code class="${langClass}">${escapeHtml(code)}</code></pre>`);
            return placeholder;
        });
         processedContentStr = processedContentStr.replace(/```\n([\s\S]*?)\n```/g, (match, code) => { // For code blocks without lang
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
                    html += `<p>${escapeHtml(line)}</p>`; // Fallback if placeholder is malformed
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
        
        return html || "<p>" + escapeHtml(contentStr) + "</p>"; // Fallback if all lines were e.g. placeholders
    }


    function addChatMessage(message, type) {
        const messageElement = document.createElement('div');
        messageElement.innerHTML = message; 
        messageElement.className = 'message ' + type;
        messagesDiv.appendChild(messageElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }
    
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

    function generateMermaidDefinition(graphData) {
        if (!graphData || !graphData.nodes || !graphData.edges) {
            return "graph TD\\n    error[\"Invalid graph data for Mermaid\"];";
        }
        let mermaidDef = "graph TD;\n"; // Top Down direction
        mermaidDef += "    classDef startEnd fill:#555,stroke:#333,stroke-width:2px,color:#fff,font-weight:bold,rx:5,ry:5;\n";
        mermaidDef += "    classDef apiNode fill:#4A90E2,stroke:#2c5282,stroke-width:2px,color:#fff,rx:5,ry:5;\n";

        // Sanitize node IDs for Mermaid
        const sanitizeNodeId = (id) => id.replace(/[^a-zA-Z0-9_]/g, '_');

        graphData.nodes.forEach(node => {
            const nodeId = sanitizeNodeId(node.display_name || node.operationId);
            let label = escapeHtml(node.summary || node.operationId);
            
            // Add operationId to label if summary is present and different
            if (node.summary && node.summary !== node.operationId && node.operationId !== "START_NODE" && node.operationId !== "END_NODE") {
                label = `${escapeHtml(node.summary)} <br/> <small>(${escapeHtml(node.operationId)})</small>`;
            } else if (node.operationId === "START_NODE" || node.operationId === "END_NODE") {
                label = `<b>${escapeHtml(node.operationId)}</b>`;
            } else {
                 label = `<b>${escapeHtml(node.operationId)}</b>`; // Default to operationId if no summary
            }


            if (node.operationId === "START_NODE" || node.operationId === "END_NODE") {
                 mermaidDef += `    ${nodeId}("${label}");\n`; // Use "" for complex labels with markdown/html
                 mermaidDef += `    class ${nodeId} startEnd;\n`;
            } else {
                mermaidDef += `    ${nodeId}("${label}");\n`;
                mermaidDef += `    class ${nodeId} apiNode;\n`;
            }
        });

        graphData.edges.forEach(edge => {
            const fromNodeId = sanitizeNodeId(edge.from_node);
            const toNodeId = sanitizeNodeId(edge.to_node);
            const edgeLabel = edge.description ? `|"${escapeHtml(edge.description.substring(0, 50))}"|` : ""; // Limit label length
            mermaidDef += `    ${fromNodeId} -->${edgeLabel} ${toNodeId};\n`;
        });
        return mermaidDef;
    }

    async function renderMermaidGraph(definition) {
        if (!definition || typeof mermaid === 'undefined') {
            mermaidDagContainer.innerHTML = "Mermaid library not loaded or definition is empty.";
            return;
        }
        try {
            mermaidDagContainer.innerHTML = ""; // Clear previous
            // Ensure the container is visible when mermaid tries to render
            const isDagViewActive = graphDagViewContent.classList.contains('active');
            if (!isDagViewActive) { // Temporarily make it visible if not, then hide
                graphDagViewContent.style.display = 'block';
            }

            const { svg } = await mermaid.render('mermaid-graph-svg-' + Date.now(), definition); // Unique ID for rendering
            mermaidDagContainer.innerHTML = svg;
            
            if (!isDagViewActive) { // Restore display state
                graphDagViewContent.style.display = 'none';
            }

        } catch (error) {
            console.error("Mermaid rendering error:", error, "\nDefinition:", definition);
            mermaidDagContainer.textContent = "Error rendering DAG. Check console for details and definition.";
        }
    }
    
    window.showGraphTab = function(tabName) {
        graphJsonViewContent.classList.remove('active');
        graphDagViewContent.classList.remove('active');
        if (jsonTabButton) jsonTabButton.classList.remove('active');
        if (dagTabButton) dagTabButton.classList.remove('active');

        if (tabName === 'json') {
            graphJsonViewContent.classList.add('active');
            if (jsonTabButton) jsonTabButton.classList.add('active');
        } else if (tabName === 'dag') {
            graphDagViewContent.classList.add('active'); // Make it active (display: block)
            if (dagTabButton) dagTabButton.classList.add('active');
            if (currentGraphData) { 
                const mermaidDef = generateMermaidDefinition(currentGraphData);
                renderMermaidGraph(mermaidDef);
            } else {
                 mermaidDagContainer.innerHTML = "Graph not yet available or cannot be rendered.";
            }
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
                currentGraphData = content; 
                if (graphJsonViewPre) graphJsonViewPre.textContent = JSON.stringify(content, null, 2);
                addChatMessage("Execution graph has been updated.", "info");
                console.log("Graph Update Received:", content);
                if (graphDagViewContent.classList.contains('active')) {
                    const mermaidDef = generateMermaidDefinition(currentGraphData);
                    renderMermaidGraph(mermaidDef);
                }
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
    
    // Ensure DOM is fully loaded before attaching event listeners to tab buttons
    if (jsonTabButton) jsonTabButton.onclick = () => showGraphTab('json');
    if (dagTabButton) dagTabButton.onclick = () => showGraphTab('dag');
    
    showGraphTab('json'); // Initialize with JSON tab active
    connect(); // Establish WebSocket connection
});
</script>
</body>
</html>
