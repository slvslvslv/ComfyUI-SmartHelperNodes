// This JS class is used to handle the specific visibility issues for the ComfyUI-SmartHelperNodes extension.

import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'   

const SMART_HV_LORA_STACK_NAME = "Smart HV LoRA Stack";
const SMART_HV_LORA_STACK_COUNT_WIDGET = "lora_count";

const nodeWidgetHandlers = {
    "SmartHVLoraStack": {
        [SMART_HV_LORA_STACK_COUNT_WIDGET]: handleSmartHVLoraStackCount
    }
};

function handleSmartHVLoraStackCount(node, widget) {
    handleVisibility(node, widget.value, SMART_HV_LORA_STACK_NAME);
}

function handleVisibility(node, count, type) {
    if (type === SMART_HV_LORA_STACK_NAME) {
        //console.log("Smart: handleVisibility called for", type, "with count", count);
        for (let i = 1; i <= 50; i++) {
            const show = i <= count;
            const enabledWidget = findWidgetByName(node, `lora_${i}_enabled`);
            const nameWidget = findWidgetByName(node, `lora_${i}_name`);
            const strengthWidget = findWidgetByName(node, `lora_${i}_strength`);

            toggleWidget(node, enabledWidget, show);
            toggleWidget(node, nameWidget, show);
            toggleWidget(node, strengthWidget, show);

            // Set disabled state based on enabled checkbox
            if (nameWidget && strengthWidget && enabledWidget) {
                nameWidget.disabled = !enabledWidget.value;
                strengthWidget.disabled = !enabledWidget.value;
            }
        }
    }
}

function findWidgetByName(node, name) {
    return node.widgets?.find(w => w.name === name);
}

function toggleWidget(node, widget, show = false) {
    if (widget) {
        widget.hidden = !show;
        
        // Trigger height recalculation
        if (node.computeSize) {
            const size = node.computeSize();
            node.size[1] = size[1];
        }
        
        // Mark canvas as dirty to trigger redraw
        node.setDirtyCanvas(true, true);
    }
}

function setupVisibilityHandler(node, countWidgetName, blockType) {
    const countWidget = node.widgets.find(w => w.name === countWidgetName);
    if (countWidget) {
        handleVisibility(node, countWidget.value, blockType);
    }
    return countWidget;
}

// Update the registerExtension implementation
app.registerExtension({
    name: "SmartHelperNodes",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SmartHVLoraStack") {
            // Store original computeSize
            const origComputeSize = nodeType.prototype.computeSize;
            
            // Override computeSize
            nodeType.prototype.computeSize = function() {
                const size = origComputeSize ? origComputeSize.call(this) : [200, 100];
                
                // Calculate height based on visible widgets
                let height = 50; // Base height
                
                for (const w of this.widgets || []) {
                    if (w.hidden) continue;
                    height += w.computeSize ? w.computeSize()[1] + 4 : 24;
                }
                
                return [size[0], height];
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Set up widget callbacks
                const loraCountWidget = setupVisibilityHandler(this, SMART_HV_LORA_STACK_COUNT_WIDGET, SMART_HV_LORA_STACK_NAME);
                
                if (loraCountWidget) {
                    loraCountWidget.callback = () => {
                        handleVisibility(this, loraCountWidget.value, SMART_HV_LORA_STACK_NAME);
                    };
                }
                
                // Set up callbacks for enabled checkboxes
                for (let i = 1; i <= 50; i++) {
                    const enabledWidget = this.widgets.find(w => w.name === `lora_${i}_enabled`);
                    if (enabledWidget) {
                        enabledWidget.callback = () => {
                            const nameWidget = findWidgetByName(this, `lora_${i}_name`);
                            const strengthWidget = findWidgetByName(this, `lora_${i}_strength`);
                            if (nameWidget && strengthWidget) {
                                nameWidget.disabled = !enabledWidget.value;
                                strengthWidget.disabled = !enabledWidget.value;
                            }
                        };
                    }
                }
                
                // Initial setup - hide widgets based on current count
                if (loraCountWidget) {
                    handleVisibility(this, loraCountWidget.value, SMART_HV_LORA_STACK_NAME);
                }

                // Handle workflow switching and initial load
                const onConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function() {
                    const result = onConnectionsChange?.apply(this, arguments);
                    setupVisibilityHandler(this, SMART_HV_LORA_STACK_COUNT_WIDGET, SMART_HV_LORA_STACK_NAME);
                    return result;
                };

                const onNodeGraphConfigure = this.onConfigure;
                this.onConfigure = function() {
                    const result = onNodeGraphConfigure?.apply(this, arguments);
                    setupVisibilityHandler(this, SMART_HV_LORA_STACK_COUNT_WIDGET, SMART_HV_LORA_STACK_NAME);
                    return result;
                };                

                return result;
            };
        }
    }
});
