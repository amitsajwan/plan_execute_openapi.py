// script.js
document.addEventListener('DOMContentLoaded', () => {
    // Existing DOM Elements
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

    // New DOM Elements for Workflow Interaction
    const runWorkflowButton = document.getElementById('runWorkflowButton');
    const workflowConfirmationModal = document.getElementById('workflowConfirmationModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalOperationId = document.getElementById('modalOperationId');
    const modalNodeId = document.getElementById('modalNodeId');
    const modalMethod = document.getElementById('modalMethod');
    const modalPath = document.getElementById('modalPath');
    const modalPayload = document.getElementById('modalPayload');
    const modalConfirmButton = document.getElementById('modalConfirmButton');
    const modalCancelButton = document.getElementById('modalCancelButton');
    const workflowLogMessagesDiv = document.getElementById('workflowLogMessages');

    let ws;
    let currentGraphData = null;
    let currentWorkflowInterruptionData = null;

    function showThinking(show) {
        thinkingIndicator.style.display = show ? 'inline-block' : 'none';
        if (sendButton) sendButton.disabled = show;
        if (messageInput) messageInput.disabled = show;
        if (runWorkflowButton) runWorkflowButton.disabled = show;
    }

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {
            // This case should ideally be handled before calling escapeHtml if special formatting is needed.
            // For safety, stringify if it's an object, otherwise convert to string.
            if (typeof unsafe === 'object' && unsafe !== null) {
                try {
                    return JSON.stringify(unsafe); // Simple stringify, no pretty print here
                } catch (e) {
                    return String(unsafe);
                }
            }
            return String(unsafe);
        }
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function formatMessageContent(contentStr, type) {
        // This function now primarily formats the core content.
        // Prefixes like "Info:", "Error:" will be handled by addChatMessage.
        if (typeof contentStr !== 'string') {
            // If content is an object, pretty-print JSON within <pre> tags.
            return '<pre>' + escapeHtml(JSON.stringify(contentStr, null, 2)) + '</pre>';
        }

        // Markdown-like formatting for strings (headings, lists, code blocks)
        let html = '';
        const codeBlockPlaceholders = [];
        // Regex for fenced code blocks
        let processedContentStr = contentStr.replace(/```(\w*)\n([\s\S]*?)\n```/g, (match, lang, code) => {
            const placeholder = `__CODEBLOCK_${codeBlockPlaceholders.length}__`;
            const langClass = lang ? `language-${escapeHtml(lang)}` : 'language-plaintext';
            codeBlockPlaceholders.push(`<pre><code class="${langClass}">${escapeHtml(code)}</code></pre>`);
            return placeholder;
        });
         processedContentStr = processedContentStr.replace(/```\n([\s\S]*?)\n```/g, (match, code) => { // For code blocks without lang
            const placeholder = `__CODEBLOCK_${codeBlockPlaceholders.length}__`;
            codeBlockPlaceholders.push(`<pre><code>${escapeHtml(code)}</code></pre>`);
            return placeholder;
        });

        const lines = processedContentStr.split('\n');
        let inList = false;
        let paragraphBuffer = [];

        function flushParagraph() {
            if (paragraphBuffer.length > 0) {
                html += `<p>${paragraphBuffer.join('<br>')}</p>`;
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
                if (!inList) { html += '<ul>'; inList = true; }
                html += `<li>${escapeHtml(line.substring(line.startsWith("- ") ? 2 : 1).trim())}</li>`;
            } else if (line.startsWith("__CODEBLOCK_")) {
                flushParagraph(); if (inList) { html += '</ul>'; inList = false; }
                const placeholderIndex = parseInt(line.substring("__CODEBLOCK_".length, line.lastIndexOf("__")));
                if (placeholderIndex >= 0 && placeholderIndex < codeBlockPlaceholders.length) {
                    html += codeBlockPlaceholders[placeholderIndex];
                } else { // Fallback if placeholder is malformed
                    html += `<p>${escapeHtml(line)}</p>`;
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

    // MODIFIED addChatMessage to accept a prefix
    function addChatMessage(prefix, mainContent, type) {
        const messageElement = document.createElement('div');
        
        const formattedMainContent = formatMessageContent(mainContent, type);
        let fullMessageHtml;

        if (prefix) {
            fullMessageHtml = `${escapeHtml(prefix)}: ${formattedMainContent}`;
        } else {
            fullMessageHtml = formattedMainContent;
        }
        
        messageElement.innerHTML = fullMessageHtml;
        messageElement.className = 'message ' + type;
        messagesDiv.appendChild(messageElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }
    
    function addWorkflowLogMessage(logContent, type) {
        if (workflowLogMessagesDiv) {
            const logElement = document.createElement('div');
            if (typeof logContent === 'object') {
                logElement.innerHTML = `<span class="log-timestamp">${new Date().toLocaleTimeString()}</span> <pre>${escapeHtml(JSON.stringify(logContent, null, 2))}</pre>`;
            } else {
                logElement.innerHTML = `<span class="log-timestamp">${new Date().toLocaleTimeString()}</span> ${escapeHtml(logContent)}`;
            }
            logElement.className = 'workflow-log-entry workflow-' + type;
            workflowLogMessagesDiv.appendChild(logElement);
            workflowLogMessagesDiv.scrollTop = workflowLogMessagesDiv.scrollHeight;
        } else {
            addChatMessage("WF Log", `(${type}) ${typeof logContent === 'object' ? JSON.stringify(logContent) : logContent}`, `workflow-${type}-fallback`);
        }
    }


    window.sendMessage = function() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            const messageText = messageInput.value;
            if (messageText.trim() === "") return;
            addChatMessage("You", messageText, "user"); // Prefix "You", content is messageText
            ws.send(messageText);
            messageInput.value = '';
            messageInput.style.height = 'auto';
            showThinking(true);
        } else {
            addChatMessage("System", "WebSocket is not connected. Cannot send message.", "error");
        }
    }

    window.runWorkflow = function() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            const commandText = "run current workflow";
            addChatMessage("You", commandText, "user");
            ws.send(commandText);
            showThinking(true);
            addWorkflowLogMessage("Attempting to start workflow...", "info");
        } else {
            addChatMessage("System", "WebSocket is not connected. Cannot run workflow.", "error");
        }
    }

    function showWorkflowConfirmationModal(data) {
        if (!workflowConfirmationModal || !modalTitle) {
            console.error("Workflow confirmation modal elements not found in HTML.");
            addChatMessage("System", "Error: UI for workflow confirmation is missing. Check console.", "error");
            return;
        }
        currentWorkflowInterruptionData = data;
        modalTitle.textContent = `Confirm API Call: ${data.operationId || 'N/A'}`;
        modalOperationId.textContent = data.operationId || 'N/A';
        modalNodeId.textContent = data.node_id || 'N/A';
        modalMethod.textContent = data.method || 'N/A';
        modalPath.textContent = data.calculated_path || data.path_template || 'N/A';
        let payloadText = "";
        if (data.calculated_payload !== undefined && data.calculated_payload !== null) { // Check for undefined or null
            try {
                payloadText = JSON.stringify(data.calculated_payload, null, 2);
            } catch (e) {
                payloadText = "Error formatting payload. See raw data in console.";
                console.error("Error stringifying payload for modal:", data.calculated_payload, e);
            }
        } else {
            payloadText = "No request payload for this operation, or payload is not applicable/calculated.";
        }
        modalPayload.value = payloadText;
        workflowConfirmationModal.style.display = 'flex';
    }

    window.hideWorkflowConfirmationModal = function() {
        if (workflowConfirmationModal) workflowConfirmationModal.style.display = 'none';
        currentWorkflowInterruptionData = null;
    }

    if (modalConfirmButton) {
        modalConfirmButton.onclick = () => {
            if (!currentWorkflowInterruptionData) {
                addChatMessage("System", "Error: No interruption data to confirm.", "error");
                hideWorkflowConfirmationModal();
                return;
            }
            let confirmedPayloadStr = modalPayload.value;
            let confirmedPayloadJson;
            try {
                // Allow empty string for no payload, or parse JSON
                if (confirmedPayloadStr.trim() === "") {
                    confirmedPayloadJson = null; // Or an empty object {} if your backend expects that for no payload
                } else {
                    confirmedPayloadJson = JSON.parse(confirmedPayloadStr);
                }
            } catch (e) {
                addChatMessage("System", "Error: Confirmed payload is not valid JSON. Please correct it or leave empty if no payload.", "error-modal");
                alert("Payload is not valid JSON. Please correct it or leave empty if no payload.");
                return;
            }

            const resumeMessage = `User confirms payload for node ${currentWorkflowInterruptionData.node_id}: ${JSON.stringify(confirmedPayloadJson)}`;
            addChatMessage("You", resumeMessage, "user"); // Send as "You"
            ws.send(resumeMessage); // Send the full string for backend parsing
            
            addWorkflowLogMessage(`Payload confirmed for node ${currentWorkflowInterruptionData.node_id}. Resuming...`, "info");
            hideWorkflowConfirmationModal();
            showThinking(true);
        };
    }
     if (modalCancelButton) {
        modalCancelButton.onclick = () => {
            if (currentWorkflowInterruptionData) {
                addWorkflowLogMessage(`Confirmation cancelled by user for node ${currentWorkflowInterruptionData.node_id}.`, "warning");
                 // Optionally, send a "cancel" message to the backend here if workflows can be explicitly cancelled.
                // ws.send(`User cancelled confirmation for node ${currentWorkflowInterruptionData.node_id}`);
            }
            hideWorkflowConfirmationModal();
        };
    }

    function generateMermaidDefinition(graphData) {
        // ... (no changes to this function) ...
        if (!graphData || !graphData.nodes || !graphData.edges) {
            return "graph TD\\n    error[\"Invalid graph data for Mermaid\"];";
        }
        let mermaidDef = "graph TD;\n";
        mermaidDef += "    classDef startEnd fill:#555,stroke:#333,stroke-width:2px,color:#fff,font-weight:bold,rx:5,ry:5;\n";
        mermaidDef += "    classDef apiNode fill:#4A90E2,stroke:#2c5282,stroke-width:2px,color:#fff,rx:5,ry:5;\n";
        mermaidDef += "    classDef runningNode fill:#f6ad55,stroke:#dd6b20,stroke-width:3px,color:#000,font-weight:bold,rx:5,ry:5; \n";
        mermaidDef += "    classDef successNode fill:#48bb78,stroke:#2f855a,stroke-width:2px,color:#fff,rx:5,ry:5; \n";
        mermaidDef += "    classDef errorNode fill:#f56565,stroke:#c53030,stroke-width:2px,color:#fff,rx:5,ry:5; \n";

        const sanitizeNodeId = (id) => id.replace(/[^a-zA-Z0-9_]/g, '_');

        graphData.nodes.forEach(node => {
            const nodeId = sanitizeNodeId(node.display_name || node.operationId);
            let label = escapeHtml(node.summary || node.operationId);
            if (node.summary && node.summary !== node.operationId && node.operationId !== "START_NODE" && node.operationId !== "END_NODE") {
                label = `${escapeHtml(node.summary)} <br/> <small>(${escapeHtml(node.operationId)})</small>`;
            } else if (node.operationId === "START_NODE" || node.operationId === "END_NODE") {
                label = `<b>${escapeHtml(node.operationId)}</b>`;
            } else {
                 label = `<b>${escapeHtml(node.operationId)}</b>`;
            }

            mermaidDef += `    ${nodeId}("${label}");\n`;
            if (node.operationId === "START_NODE" || node.operationId === "END_NODE") {
                 mermaidDef += `    class ${nodeId} startEnd;\n`;
            } else {
                mermaidDef += `    class ${nodeId} apiNode;\n`;
            }
        });

        graphData.edges.forEach(edge => {
            const fromNodeId = sanitizeNodeId(edge.from_node);
            const toNodeId = sanitizeNodeId(edge.to_node);
            const edgeLabel = edge.description ? `|"${escapeHtml(edge.description.substring(0, 50))}"|` : "";
            mermaidDef += `    ${fromNodeId} -->${edgeLabel} ${toNodeId};\n`;
        });
        return mermaidDef;
    }

    async function renderMermaidGraph(definition) {
        // ... (no changes to this function) ...
        if (!definition || typeof mermaid === 'undefined') {
            mermaidDagContainer.innerHTML = "Mermaid library not loaded or definition is empty.";
            return;
        }
        try {
            mermaidDagContainer.innerHTML = ""; 
            const isDagViewActive = graphDagViewContent.classList.contains('active');
            if (!isDagViewActive) {
                graphDagViewContent.style.display = 'block';
            }

            const { svg } = await mermaid.render('mermaid-graph-svg-' + Date.now(), definition);
            mermaidDagContainer.innerHTML = svg;
            
            if (!isDagViewActive) {
                graphDagViewContent.style.display = 'none';
            }

        } catch (error) {
            console.error("Mermaid rendering error:", error, "\nDefinition:", definition);
            mermaidDagContainer.textContent = "Error rendering DAG. Check console for details and definition.";
        }
    }
    
    function updateMermaidNodeStatus(nodeId, statusClass) {
        // ... (no changes to this function, assuming it works by re-rendering) ...
        if (!currentGraphData || !mermaidDagContainer.querySelector('svg')) {
            console.warn("Cannot update Mermaid node status: No graph data or SVG not rendered.");
            return;
        }
        const sanitizedNodeId = nodeId.replace(/[^a-zA-Z0-9_]/g, '_');
        let nodeFound = false;
        currentGraphData.nodes.forEach(node => {
            if ((node.display_name || node.operationId) === nodeId) {
                node._tempStatusClass = statusClass;
                nodeFound = true;
            } else {
                delete node._tempStatusClass;
            }
        });

        if (nodeFound && graphDagViewContent.classList.contains('active')) {
            let mermaidDef = "graph TD;\n";
            mermaidDef += "    classDef startEnd fill:#555,stroke:#333,stroke-width:2px,color:#fff,font-weight:bold,rx:5,ry:5;\n";
            mermaidDef += "    classDef apiNode fill:#4A90E2,stroke:#2c5282,stroke-width:2px,color:#fff,rx:5,ry:5;\n";
            mermaidDef += "    classDef runningNode fill:#f6ad55,stroke:#dd6b20,stroke-width:3px,color:#000,font-weight:bold,rx:5,ry:5; \n";
            mermaidDef += "    classDef successNode fill:#48bb78,stroke:#2f855a,stroke-width:2px,color:#fff,rx:5,ry:5; \n";
            mermaidDef += "    classDef errorNode fill:#f56565,stroke:#c53030,stroke-width:2px,color:#fff,rx:5,ry:5; \n";

            currentGraphData.nodes.forEach(node => {
                const gNodeId = (node.display_name || node.operationId).replace(/[^a-zA-Z0-9_]/g, '_');
                let label = escapeHtml(node.summary || node.operationId);
                 if (node.summary && node.summary !== node.operationId && node.operationId !== "START_NODE" && node.operationId !== "END_NODE") {
                    label = `${escapeHtml(node.summary)} <br/> <small>(${escapeHtml(node.operationId)})</small>`;
                } else if (node.operationId === "START_NODE" || node.operationId === "END_NODE") {
                    label = `<b>${escapeHtml(node.operationId)}</b>`;
                } else {
                    label = `<b>${escapeHtml(node.operationId)}</b>`;
                }
                mermaidDef += `    ${gNodeId}("${label}");\n`;
                if (node._tempStatusClass) {
                    mermaidDef += `    class ${gNodeId} ${node._tempStatusClass};\n`;
                } else if (node.operationId === "START_NODE" || node.operationId === "END_NODE") {
                    mermaidDef += `    class ${gNodeId} startEnd;\n`;
                } else {
                    mermaidDef += `    class ${gNodeId} apiNode;\n`;
                }
            });
            currentGraphData.edges.forEach(edge => {
                 const fromNodeId = (edge.from_node).replace(/[^a-zA-Z0-9_]/g, '_');
                 const toNodeId = (edge.to_node).replace(/[^a-zA-Z0-9_]/g, '_');
                 const edgeLabel = edge.description ? `|"${escapeHtml(edge.description.substring(0, 50))}"|` : "";
                 mermaidDef += `    ${fromNodeId} -->${edgeLabel} ${toNodeId};\n`;
            });
            renderMermaidGraph(mermaidDef);
        }
    }

    window.showGraphTab = function(tabName) {
        // ... (no changes to this function) ...
        graphJsonViewContent.classList.remove('active');
        graphDagViewContent.classList.remove('active');
        if (jsonTabButton) jsonTabButton.classList.remove('active');
        if (dagTabButton) dagTabButton.classList.remove('active');

        if (tabName === 'json') {
            graphJsonViewContent.classList.add('active');
            if (jsonTabButton) jsonTabButton.classList.add('active');
        } else if (tabName === 'dag') {
            graphDagViewContent.classList.add('active');
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
        addChatMessage("System", "Connecting to " + wsUrl, "info"); // Use new addChatMessage
        ws = new WebSocket(wsUrl);

        ws.onopen = () => { addChatMessage("System", "WebSocket connected.", "info"); showThinking(false); };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            let content = data.content;
            const type = data.type || "agent";

            if (type === "final" || type === "error" || type.startsWith("workflow_execution_")) {
                showThinking(false);
            }
            if (type === "status" && content && typeof content === 'string' && content.toLowerCase().includes("processing")) {
                 showThinking(true);
            }

            if (type === "graph_update") {
                currentGraphData = content;
                if (graphJsonViewPre) graphJsonViewPre.textContent = JSON.stringify(content, null, 2);
                addChatMessage("System", "Execution graph has been updated.", "info");
                if (graphDagViewContent.classList.contains('active')) {
                    const mermaidDef = generateMermaidDefinition(currentGraphData);
                    renderMermaidGraph(mermaidDef);
                }
                return;
            }

            if (type.startsWith("workflow_")) {
                const workflowEventType = type.substring("workflow_".length);
                addWorkflowLogMessage(content, workflowEventType);

                if (workflowEventType === "interrupt_confirmation_required") {
                    showWorkflowConfirmationModal(content);
                    showThinking(false);
                } else if (workflowEventType === "node_execution_started") {
                    addChatMessage("Workflow", `Executing node '${content.node_id || content.operationId}'...`, "workflow-info");
                    if (content.node_id) updateMermaidNodeStatus(content.node_id, 'runningNode');
                } else if (workflowEventType === "node_execution_succeeded") {
                     addChatMessage("Workflow", `Node '${content.node_id}' succeeded. ${content.result_preview ? 'Result: ' + escapeHtml(content.result_preview) : ''}`, "workflow-success");
                    if (content.node_id) updateMermaidNodeStatus(content.node_id, 'successNode');
                } else if (workflowEventType === "node_execution_failed") {
                     addChatMessage("Workflow", `Node '${content.node_id}' failed. Error: ${escapeHtml(JSON.stringify(content.error))}`, "workflow-error");
                    if (content.node_id) updateMermaidNodeStatus(content.node_id, 'errorNode');
                } else if (workflowEventType === "execution_completed") {
                    addChatMessage("Workflow", "Execution completed.", "workflow-info");
                } else if (workflowEventType === "execution_failed") {
                    addChatMessage("Workflow", `Execution failed. Reason: ${escapeHtml(JSON.stringify(content.error || content))}`, "workflow-error");
                }
            } else if (type === "user") { // Should be handled by sendMessage, but as a safeguard
                 addChatMessage("You", content, "user");
            } else { // Handle regular agent/system messages (info, error, final, etc.)
                let messagePrefix = type.charAt(0).toUpperCase() + type.slice(1);
                let displayContent = content;

                if (type === "info" && typeof content === 'object' && content !== null) {
                    displayContent = content.message || JSON.stringify(content); // Prioritize .message
                    if (content.session_id && content.message) { // Add session_id only if .message was present
                         displayContent += ` (Session: ${content.session_id})`;
                    }
                }
                // For all non-workflow messages, 'displayContent' is now the core content (either original string or extracted/stringified object)
                // 'formatMessageContent' will then format this core content.
                // The prefix is added by addChatMessage.
                addChatMessage(messagePrefix, displayContent, type);
            }
        };

        ws.onerror = (error) => {
            addChatMessage("System", "WebSocket error. Check console. (If page is HTTPS, WS must be WSS).", "error");
            console.error("WebSocket error object:", error);
            showThinking(false);
        };
        ws.onclose = (event) => {
            let reason = "";
            if (event.code) reason += `Code: ${event.code} `;
            if (event.reason) reason += `Reason: ${event.reason} `;
            if (event.wasClean) reason += `(Clean close) `; else reason += `(Unclean close) `;
            addChatMessage("System", "WebSocket disconnected. " + reason + "Attempting to reconnect in 5s...", "info");
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

    if (runWorkflowButton) {
        runWorkflowButton.onclick = runWorkflow;
    } else {
        console.warn("runWorkflowButton not found. Add it to index.html to enable workflow execution via UI button.");
    }
    
    if (jsonTabButton) jsonTabButton.onclick = () => showGraphTab('json');
    if (dagTabButton) dagTabButton.onclick = () => showGraphTab('dag');
    
    showGraphTab('json');
    connect();
});
