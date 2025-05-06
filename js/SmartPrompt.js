import { app } from "/scripts/app.js";
import { api } from "../../../scripts/api.js"; // Assuming textHighlight's relative path works

// Flag to prevent multiple enhancements (mimicking textHighlight)
const enhancedTextareas = new WeakSet();

console.log("SmartPrompt Extension: Loading script");

// Helper function to escape HTML characters
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    // Keep only essential HTML entity escaping
    return text.replace(/[&<>\"\']/g, m => map[m]);
}

// NEW: Helper function to wrap text in a styled span
function colorizeText(style, text) {
    let resultHtml = '';
    let currentWord = '';
    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        if (char === ' ') {
            if (currentWord) {
                resultHtml += `<span style="${style}">${escapeHtml(currentWord)}</span>`;
                currentWord = '';
            }
            resultHtml += ' '; // Append the actual space
        } else {
            currentWord += char;
        }
    }
    // Append any remaining word at the end
    if (currentWord) {
        resultHtml += `<span style="${style}">${escapeHtml(currentWord)}</span>`;
    }
    return resultHtml;
}

// Helper function to style default text, highlighting punctuation
function styleDefaultText(text) {
    const defaultTextStyle = "color: rgb(200, 200, 200);"; // Your default grey
    const punctuationStyle = "color: black;";          // Style for punctuation
    const pipeStyle = "color: red;";                   // Style for vertical line / braces
    let resultHtml = '';
    // Iterate character by character
    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        if (['.', ',', '(', ')', ';', ':'].includes(char)) { // Removed { and }
             resultHtml += colorizeText(punctuationStyle, char);
        } else if (['|', '{', '}'].includes(char)) { // Added { and }
             resultHtml += colorizeText(pipeStyle, char);
        } else {
            // Optimization: Group consecutive non-punctuation characters
            let nonPunctSubstring = char;
            // Keep the while loop condition as is, it correctly handles braces and pipe
            while (i + 1 < text.length && !['.', ',', '(', ')', ';', ':', '{', '}', '|'].includes(text[i + 1])) {
                i++;
                nonPunctSubstring += text[i];
            }
             resultHtml += colorizeText(defaultTextStyle, nonPunctSubstring);
        }
    }
    return resultHtml;
}

// Function to sync textarea content to the overlay (styles comments, floats, preceding colon, punctuation, and default text)
function syncSmartPromptText(textarea, overlay) {
    const lines = textarea.value.split('\n');
    let htmlContent = '';
    // Main regex for comments and colon-floats
    const highlightRegex = /(\/\/.*)|:(\s*\d+\.\d*|:?\s*\.?\d*)/g;
    // Define styles
    const defaultTextStyle = "color: rgb(200, 200, 200);";
    const commentStyle = "color: rgb(0, 0, 255);";
    const floatStyle = "color: rgba(0, 255, 0, 1);";

    lines.forEach(line => {
        let lastIndex = 0;
        let lineHtml = '';
        let match;
		console.log("line: " + line);

        while ((match = highlightRegex.exec(line)) !== null) {
            // Append text before the match, processed by the helper
            if (match.index > lastIndex) {
                lineHtml += styleDefaultText(line.substring(lastIndex, match.index));
            }
			console.log(lineHtml + " , " + match);
            // Append the matched part(s) using colorizeText
            if (match[1]) { // Comment
                lineHtml += colorizeText(commentStyle, match[1]);
            } else if (match[2]) { // Float
                lineHtml += ':' + colorizeText(floatStyle, match[2]); // Float text
            }
            lastIndex = highlightRegex.lastIndex;
        }

        // Append any remaining text after the last match, processed by the helper
        if (lastIndex < line.length) {
             lineHtml += styleDefaultText(line.substring(lastIndex));
        }

        // Handle empty lines: Render a styled non-breaking space using default color
        // Explicitly use nbsp for empty lines to ensure they have content/height
        htmlContent += (lineHtml || `<span style="${defaultTextStyle}">&nbsp;</span>`) + '<br>';
    });

    if (htmlContent.endsWith('<br>')) {
        htmlContent = htmlContent.substring(0, htmlContent.length - 4);
    }

    overlay.innerHTML = htmlContent;
    overlay.scrollTop = textarea.scrollTop;
    overlay.scrollLeft = textarea.scrollLeft;
}

// Function to set overlay style based on textarea
function setOverlayStyle(inputEl, overlayEl) {
    const textareaStyle = window.getComputedStyle(inputEl);
    // Base styles for the overlay
    overlayEl.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: ${textareaStyle.width};
        height: ${textareaStyle.height};
        font-family: ${textareaStyle.fontFamily};
        font-size: ${textareaStyle.fontSize};
        font-weight: ${textareaStyle.fontWeight};
        line-height: ${textareaStyle.lineHeight};
        letter-spacing: ${textareaStyle.letterSpacing};
        padding: ${textareaStyle.padding};
        margin: ${textareaStyle.margin};
        border: ${textareaStyle.border};
        box-sizing: ${textareaStyle.boxSizing};
        overflow: hidden; /* Let scroll sync handle visible area */
        pointer-events: none;
        white-space: pre-wrap;
        word-wrap: break-word;
        color: transparent; /* Keep overlay text transparent */
        background-color: var(--comfy-input-bg); /* Explicitly use the input background variable */
        z-index: 1; /* Ensure overlay is above textarea background */
    `;
    // --- Make underlying textarea text semi-transparent ---
    inputEl.style.opacity = '0.4'; // Adjust value (0.0 to 1.0) as needed

     // Ensure textarea background remains transparent
    inputEl.style.background = "transparent !important"; // Use !important cautiously
    inputEl.style.position = "relative";
    inputEl.style.zIndex = "0";
    // Set caret color explicitly to ensure visibility against potentially dimmed text
    inputEl.style.caretColor = textareaStyle.color || 'var(--input-text)';
}

// Adopted enhanceTextarea structure
function enhanceSmartPromptTextarea(textarea) {
    if (enhancedTextareas.has(textarea)) return;
    enhancedTextareas.add(textarea);

    // Create overlay div (similar to textHighlight.js)
    const overlayEl = document.createElement("div");
    overlayEl.className = "smartprompt-overlay"; // Use our own class
    // Insert before the textarea
    textarea.parentNode.insertBefore(overlayEl, textarea);

    // Make textarea transparent and relative
    textarea.style.position = "relative";
    textarea.style.background = "transparent";
    textarea.style.zIndex = "0";
    textarea.style.caretColor = window.getComputedStyle(textarea).color; // Ensure caret is visible

    // Initial setup
    setOverlayStyle(textarea, overlayEl);
    syncSmartPromptText(textarea, overlayEl); // Initial sync with our logic

    // Scroll sync
    textarea.addEventListener("scroll", () => {
        overlayEl.scrollTop = textarea.scrollTop;
        overlayEl.scrollLeft = textarea.scrollLeft;
    });

    // Input sync
    textarea.addEventListener("input", () => {
        syncSmartPromptText(textarea, overlayEl);
        // Re-apply style in case of resize/reflow needed due to content change? Maybe not needed.
        // setOverlayStyle(textarea, overlayEl);
    });

    // Observe textarea style/size changes (like textHighlight.js)
    const observer = new MutationObserver(() => {
        // Use requestAnimationFrame to avoid layout thrashing
        requestAnimationFrame(() => {
             if (document.contains(textarea)) { // Check if textarea still exists
                setOverlayStyle(textarea, overlayEl);
                // Resync needed if width/height affects wrapping
                syncSmartPromptText(textarea, overlayEl);
             } else {
                observer.disconnect(); // Stop observing if textarea is gone
             }
        });
    });
    observer.observe(textarea, {
        attributes: true, attributeFilter: ["style", "class"], // Watch style/class changes
        childList: false, subtree: false // Don't need these for textarea
    });

    // Also observe size changes using ResizeObserver (more reliable for size)
     const resizeObserver = new ResizeObserver(() => {
        requestAnimationFrame(() => {
            if (document.contains(textarea)) {
                setOverlayStyle(textarea, overlayEl);
                syncSmartPromptText(textarea, overlayEl);
            } else {
                resizeObserver.disconnect();
            }
        });
    });
    resizeObserver.observe(textarea);


    // Clean up overlay and observers if textarea's node is removed
     const onRemoved = textarea.closest('.comfy-node')?.onRemoved; // Find the node's onRemoved
     if(textarea.closest('.comfy-node')) {
         textarea.closest('.comfy-node').onRemoved = function() {
            console.log("SmartPrompt: Cleaning up overlay and observers for removed node.");
            observer.disconnect();
            resizeObserver.disconnect();
            if (overlayEl && overlayEl.parentNode) {
                overlayEl.remove();
            }
             onRemoved?.apply(this, arguments); // Call original if exists
         }
     }
}

app.registerExtension({
    name: "Comfy.SmartHelperNodes.SmartPrompt",
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) { // Renamed app to avoid conflict
        if (nodeData.name === "SmartPrompt") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                const widget = this.widgets.find(w => w.name === "text");
                if (widget && widget.inputEl) {
                    // Use timeout to ensure element is fully in DOM before enhancing
                    setTimeout(() => {
                         if (document.contains(widget.inputEl)) { // Final check
                            enhanceSmartPromptTextarea(widget.inputEl);
                         } else {
                             console.error("SmartPrompt: Textarea not in DOM for enhancement.");
                         }
                    }, 0);
                }
            };
        } else {
            // Optional: Log other nodes being processed if needed for debugging
            // console.log(`SmartPrompt Extension: Skipping node type ${nodeData.name}`);
        }
    }
});

console.log("SmartPrompt Extension: Script loaded and registered"); 

// Function to set overlay position based on textarea (Keep this separate)
function setOverlayPosition(textarea, overlay) {
    const textareaStyle = window.getComputedStyle(textarea);
    requestAnimationFrame(() => {
         const styleProps = ['left', 'top', 'width', 'height', 'display', 'transform', 'transformOrigin'];
         styleProps.forEach(prop => {
             if (overlay.style[prop] !== textareaStyle[prop]) {
                  overlay.style[prop] = textareaStyle[prop];
             }
         });
    });
} 