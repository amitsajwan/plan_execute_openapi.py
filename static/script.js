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
    const runWorkflowButton = document.getElementById('runWorkflowButton'); // Will be null if not added to HTML
    const workflowConfirmationModal = document.getElementById('workflowConfirmationModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalOperationId = document.getElementById('modalOperationId');
    const modalNodeId = document.getElementById('modalNodeId');
    const modalMethod = document.getElementById('modalMethod');
    const modalPath = document.getElementById('modalPath');
    const modalPayload = document.getElementById('modalPayload');
    const modalConfirmButton = document.getElementById('modalConfirmButton');
    const modalCancelButton = document.getElementById('modalCancelButton'); // Assuming you add this button
    const workflowLogMessagesDiv = document.getElementById('workflowLogMessages'); // Optional dedicated log

    let ws;
    let currentGraphData = null;
    let currentWorkflowInterruptionData = null; // To store data for the current interruption

    function showThinking(show) {
        thinkingIndicator.style.display = show ? 'inline-block' : 'none';
        if (sendButton) sendButton.disabled = show;
        if (messageInput) messageInput.disabled = show;
        if (runWorkflowButton) runWorkflowButton.disabled = show; // Disable run workflow button too
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
        // Allow direct HTML for workflow messages if they are carefully constructed server-side
        // or use specific formatting for them. For now, keep existing logic.
        if (type.startsWith("workflow_")) { // Basic handling for workflow messages
            if (typeof contentStr === 'object') {
                return '<pre>' + escapeHtml(JSON.stringify(contentStr, null, 2)) + '</pre>';
            }
            return escapeHtml(contentStr); // Default to escaping for safety
        }

        // Existing formatting logic for agent messages
        if (typeof contentStr !== 'string') {
            return '<pre>' + escapeHtml(JSON.stringify(contentStr, null, 2)) + '</pre>';
        }
        if (type !== 'agent' && type !== 'final' && type !== 'intermediate' && type !== 'error' && type !== 'info' && type !== 'warning' && type !== 'status') {
             // If it's not a known agent message type, escape it simply.
             // This is a fallback, ideally all messages have a clear type.
            return escapeHtml(contentStr);
        }

        let html = '';
        const codeBlockPlaceholders = [];
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
                } else {
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
        return html || "<p>" + escapeHtml(contentStr) + "</p>";
    }

    function addChatMessage(messageContent, type, isRawHtml = false) {
        const messageElement = document.createElement('div');
        if (isRawHtml) {
            messageElement.innerHTML = messageContent; // Use if content is already safe HTML
        } else {
            messageElement.innerHTML = formatMessageContent(messageContent, type);
        }
        messageElement.className = 'message ' + type; // e.g., 'message agent', 'message workflow_info'
        messagesDiv.appendChild(messageElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    // Function to add messages to the dedicated workflow log (if it exists)
    function addWorkflowLogMessage(logContent, type) {
        if (workflowLogMessagesDiv) {
            const logElement = document.createElement('div');
            // Workflow logs might be more structured, consider specific formatting
            if (typeof logContent === 'object') {
                logElement.innerHTML = `<span class="log-timestamp">${new Date().toLocaleTimeString()}</span> <pre>${escapeHtml(JSON.stringify(logContent, null, 2))}</pre>`;
            } else {
                logElement.innerHTML = `<span class="log-timestamp">${new Date().toLocaleTimeString()}</span> ${escapeHtml(logContent)}`;
            }
            logElement.className = 'workflow-log-entry workflow-' + type; // e.g. workflow-node_started
            workflowLogMessagesDiv.appendChild(logElement);
            workflowLogMessagesDiv.scrollTop = workflowLogMessagesDiv.scrollHeight;
        } else {
            // Fallback to main chat if dedicated log area doesn't exist
            addChatMessage(`WF Log (${type}): ${typeof logContent === 'object' ? JSON.stringify(logContent) : logContent}`, `workflow-${type}-fallback`);
        }
    }


    window.sendMessage = function() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            const messageText = messageInput.value;
            if (messageText.trim() === "") return;
            addChatMessage(`You: ${escapeHtml(messageText)}`, "user"); // Display user message
            ws.send(messageText); // Send raw text to backend
            messageInput.value = '';
            messageInput.style.height = 'auto';
            showThinking(true);
        } else {
            addChatMessage("WebSocket is not connected. Cannot send message.", "error");
        }
    }

    window.runWorkflow = function() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            const commandText = "run current workflow"; // Specific command for the backend router
            addChatMessage(`You: ${escapeHtml(commandText)}`, "user");
            ws.send(commandText);
            showThinking(true);
            addWorkflowLogMessage("Attempting to start workflow...", "info");
        } else {
            addChatMessage("WebSocket is not connected. Cannot run workflow.", "error");
        }
    }

    // --- Workflow Confirmation Modal Functions ---
    function showWorkflowConfirmationModal(data) {
        if (!workflowConfirmationModal || !modalTitle) { // Check if modal elements exist
            console.error("Workflow confirmation modal elements not found in HTML.");
            addChatMessage("Error: UI for workflow confirmation is missing. Check console.", "error");
            // Fallback: send an auto-deny or log error, as user cannot confirm.
            // For now, just log and the workflow will likely stall or timeout on backend.
            return;
        }
        currentWorkflowInterruptionData = data; // Store for sending confirmation

        modalTitle.textContent = `Confirm API Call: ${data.operationId || 'N/A'}`;
        modalOperationId.textContent = data.operationId || 'N/A';
        modalNodeId.textContent = data.node_id || 'N/A';
        modalMethod.textContent = data.method || 'N/A';
        modalPath.textContent = data.calculated_path || data.path_template || 'N/A';
        
        let payloadText = "";
        if (data.calculated_payload) {
            try {
                payloadText = JSON.stringify(data.calculated_payload, null, 2);
            } catch (e) {
                payloadText = "Error formatting payload. See raw data in console.";
                console.error("Error stringifying payload for modal:", data.calculated_payload, e);
            }
        } else {
            payloadText = "No request payload for this operation, or payload is not yet calculated.";
        }
        modalPayload.value = payloadText;
        
        workflowConfirmationModal.style.display = 'flex';
    }

    window.hideWorkflowConfirmationModal = function() { // Make it globally accessible if needed
        if (workflowConfirmationModal) workflowConfirmationModal.style.display = 'none';
        currentWorkflowInterruptionData = null;
    }

    if (modalConfirmButton) {
        modalConfirmButton.onclick = () => {
            if (!currentWorkflowInterruptionData) {
                addChatMessage("Error: No interruption data to confirm.", "error");
                hideWorkflowConfirmationModal();
                return;
            }
            let confirmedPayloadStr = modalPayload.value;
            let confirmedPayloadJson;
            try {
                confirmedPayloadJson = JSON.parse(confirmedPayloadStr);
            } catch (e) {
                addChatMessage("Error: Confirmed payload is not valid JSON. Please correct it.", "error-modal"); // You might need a specific class for modal errors
                alert("Payload is not valid JSON. Please correct it."); // Simple alert for now
                return;
            }

            // Send message to backend to resume workflow
            // The backend's interactive_query_planner needs to parse this text.
            const resumeMessage = `User confirms payload for node ${currentWorkflowInterruptionData.node_id}: ${JSON.stringify(confirmedPayloadJson)}`;
            addChatMessage(`You: ${escapeHtml(resumeMessage)}`, "user");
            ws.send(resumeMessage);
            
            addWorkflowLogMessage(`Payload confirmed for node ${currentWorkflowInterruptionData.node_id}. Resuming...`, "info");
            hideWorkflowConfirmationModal();
            showThinking(true); // Show thinking while backend processes resume
        };
    }
     if (modalCancelButton) { // Assuming you have a cancel button with id="modalCancelButton"
        modalCancelButton.onclick = () => {
            if (currentWorkflowInterruptionData) {
                // Inform backend about cancellation if necessary, or just close UI
                // For now, just closes UI. Backend will timeout if it expects a response.
                addWorkflowLogMessage(`Confirmation cancelled by user for node ${currentWorkflowInterruptionData.node_id}.`, "warning");
            }
            hideWorkflowConfirmationModal();
        };
    }


    // --- Mermaid Graph Functions ---
    function generateMermaidDefinition(graphData) {
        if (!graphData || !graphData.nodes || !graphData.edges) {
            return "graph TD\\n    error[\"Invalid graph data for Mermaid\"];";
        }
        let mermaidDef = "graph TD;\n";
        mermaidDef += "    classDef startEnd fill:#555,stroke:#333,stroke-width:2px,color:#fff,font-weight:bold,rx:5,ry:5;\n";
        mermaidDef += "    classDef apiNode fill:#4A90E2,stroke:#2c5282,stroke-width:2px,color:#fff,rx:5,ry:5;\n";
        mermaidDef += "    classDef runningNode fill:#f6ad55,stroke:#dd6b20,stroke-width:3px,color:#000,font-weight:bold,rx:5,ry:5; \n"; // For running nodes
        mermaidDef += "    classDef successNode fill:#48bb78,stroke:#2f855a,stroke-width:2px,color:#fff,rx:5,ry:5; \n"; // For success nodes
        mermaidDef += "    classDef errorNode fill:#f56565,stroke:#c53030,stroke-width:2px,color:#fff,rx:5,ry:5; \n";   // For error nodes


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
                mermaidDef += `    class ${nodeId} apiNode;\n`; // Default class
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
        if (!definition || typeof mermaid === 'undefined') {
            mermaidDagContainer.innerHTML = "Mermaid library not loaded or definition is empty.";
            return;
        }
        try {
            mermaidDagContainer.innerHTML = ""; 
            const isDagViewActive = graphDagViewContent.classList.contains('active');
            if (!isDagViewActive) {
                graphDagViewContent.style.display = 'block'; // Temporarily show for rendering
            }

            const { svg } = await mermaid.render('mermaid-graph-svg-' + Date.now(), definition);
            mermaidDagContainer.innerHTML = svg;
            
            if (!isDagViewActive) {
                graphDagViewContent.style.display = 'none'; // Hide again if it wasn't active
            }
        } catch (error) {
            console.error("Mermaid rendering error:", error, "\nDefinition:", definition);
            mermaidDagContainer.textContent = "Error rendering DAG. Check console.";
        }
    }
    
    // Function to update node class in Mermaid graph (e.g., for highlighting)
    // Note: Directly manipulating Mermaid's generated SVG can be fragile.
    // A better way might be to re-render with updated class definitions if Mermaid supports it easily,
    // or use Mermaid's API if it allows dynamic class changes.
    // For simplicity, this example might re-render or just log.
    function updateMermaidNodeStatus(nodeId, statusClass) { // statusClass e.g., 'runningNode', 'successNode', 'errorNode'
        if (!currentGraphData || !mermaidDagContainer.querySelector('svg')) {
            console.warn("Cannot update Mermaid node status: No graph data or SVG not rendered.");
            return;
        }
        const sanitizedNodeId = nodeId.replace(/[^a-zA-Z0-9_]/g, '_');
        // This is a simplified approach: Re-render the graph with the new class for the node.
        // Find the node in currentGraphData and add a temporary 'statusClass' property
        let nodeFound = false;
        currentGraphData.nodes.forEach(node => {
            if ((node.display_name || node.operationId) === nodeId) {
                node._tempStatusClass = statusClass; // Add a temporary property
                nodeFound = true;
            } else {
                delete node._tempStatusClass; // Remove from others
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
        addChatMessage("System: Connecting to " + wsUrl, "info");
        ws = new WebSocket(wsUrl);

        ws.onopen = () => { addChatMessage("System: WebSocket connected.", "info"); showThinking(false); };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            let content = data.content;
            const type = data.type || "agent"; // Default type if not specified

            // Stop thinking indicator for most final messages from agent or workflow
            if (type === "final" || type === "error" || type.startsWith("workflow_execution_")) {
                showThinking(false);
            }
            if (type === "status" && content && content.toLowerCase().includes("processing")) {
                 showThinking(true);
            }


            if (type === "graph_update") {
                currentGraphData = content;
                if (graphJsonViewPre) graphJsonViewPre.textContent = JSON.stringify(content, null, 2);
                addChatMessage("System: Execution graph has been updated.", "info");
                if (graphDagViewContent.classList.contains('active')) { // If DAG view is active, re-render
                    const mermaidDef = generateMermaidDefinition(currentGraphData);
                    renderMermaidGraph(mermaidDef);
                }
                return; // Handled graph update
            }

            // Handle workflow-specific messages
            if (type.startsWith("workflow_")) {
                const workflowEventType = type.substring("workflow_".length);
                addWorkflowLogMessage(content, workflowEventType); // Log to dedicated area or main chat

                if (workflowEventType === "interrupt_confirmation_required") {
                    showWorkflowConfirmationModal(content); // `content` should be the interruption data
                    showThinking(false); // Stop thinking as we are waiting for user
                } else if (workflowEventType === "node_execution_started") {
                    addChatMessage(`Workflow: Executing node '${content.node_id || content.operationId}'...`, "workflow-info");
                    if (content.node_id) updateMermaidNodeStatus(content.node_id, 'runningNode');
                } else if (workflowEventType === "node_execution_succeeded") {
                     addChatMessage(`Workflow: Node '${content.node_id}' succeeded. ${content.result_preview ? 'Result: ' + escapeHtml(content.result_preview) : ''}`, "workflow-success");
                    if (content.node_id) updateMermaidNodeStatus(content.node_id, 'successNode');
                } else if (workflowEventType === "node_execution_failed") {
                     addChatMessage(`Workflow: Node '${content.node_id}' failed. Error: ${escapeHtml(JSON.stringify(content.error))}`, "workflow-error");
                    if (content.node_id) updateMermaidNodeStatus(content.node_id, 'errorNode');
                } else if (workflowEventType === "execution_completed") {
                    addChatMessage("Workflow: Execution completed.", "workflow-info");
                } else if (workflowEventType === "execution_failed") {
                    addChatMessage(`Workflow: Execution failed. Reason: ${escapeHtml(JSON.stringify(content.error || content))}`, "workflow-error");
                }
                // Other workflow event types can be handled here as needed
            } else {
                // Handle regular agent messages
                const messagePrefix = type.charAt(0).toUpperCase() + type.slice(1);
                addChatMessage(`${messagePrefix}: ${formatMessageContent(content, type)}`, type);
            }
        };

        ws.onerror = (error) => {
            addChatMessage("System: WebSocket error. Check console. (If page is HTTPS, WS must be WSS).", "error");
            console.error("WebSocket error object:", error);
            showThinking(false);
        };
        ws.onclose = (event) => {
            let reason = "";
            if (event.code) reason += `Code: ${event.code} `;
            if (event.reason) reason += `Reason: ${event.reason} `;
            if (event.wasClean) reason += `(Clean close) `; else reason += `(Unclean close) `;
            addChatMessage("System: WebSocket disconnected. " + reason + "Attempting to reconnect in 5s...", "info");
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

    // Attach event listener for the "Run Workflow" button if it exists
    if (runWorkflowButton) {
        runWorkflowButton.onclick = runWorkflow;
    } else {
        console.warn("runWorkflowButton not found. Add it to index.html to enable workflow execution via UI button.");
    }
    
    if (jsonTabButton) jsonTabButton.onclick = () => showGraphTab('json');
    if (dagTabButton) dagTabButton.onclick = () => showGraphTab('dag');
    
    showGraphTab('json'); // Initialize with JSON tab active
    connect(); // Establish WebSocket connection
});
