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

const SMART_MODEL_LOADER_VISIBILITY = {
    SmartModelLoader: {
        safetensors: ["unet_name", "weight_dtype"],
        gguf: ["gguf_name"],
    },
    SmartDualModelLoader: {
        safetensors: ["unet_name_1", "unet_name_2", "weight_dtype"],
        gguf: ["gguf_name_1", "gguf_name_2"],
    },
};

const SMART_BUS_DEFAULT_LABELS = [
    "any 1",
    "any 2",
    "any 3",
    "any 4",
    "any 5",
    "any 6",
    "any 7",
    "any 8",
    "any 9",
    "any 10",
];

const SMART_BUS_NODE_CONFIG = {
    SmartBusIn: {
        inputOffset: 1,
        outputOffset: null,
        editable: true,
        slotCount: 10,
    },
    SmartBusOut: {
        inputOffset: null,
        outputOffset: 1,
        editable: false,
        slotCount: 10,
    },
    SmartBusIn5: {
        inputOffset: 1,
        outputOffset: null,
        editable: true,
        slotCount: 5,
    },
    SmartBusOut5: {
        inputOffset: null,
        outputOffset: 1,
        editable: false,
        slotCount: 5,
    },
};

function isSmartBusInNodeType(nodeType) {
    return nodeType === "SmartBusIn" || nodeType === "SmartBusIn5";
}

function isSmartBusOutNodeType(nodeType) {
    return nodeType === "SmartBusOut" || nodeType === "SmartBusOut5";
}

function getSmartBusPropertyNames(config) {
    return SMART_BUS_DEFAULT_LABELS.slice(0, config.slotCount);
}

function ensureSmartBusProperties(node, config) {
    if (!node.properties) {
        node.properties = {};
    }
    if (config.editable && !node.properties_info) {
        node.properties_info = [];
    }

    const propertyNames = getSmartBusPropertyNames(config);

    if (config.editable && node.properties_info) {
        node.properties_info = node.properties_info.filter(
            (info) => !SMART_BUS_DEFAULT_LABELS.includes(info.name) || propertyNames.includes(info.name)
        );
    }

    propertyNames.forEach((defaultLabel) => {
        const propertyName = defaultLabel;
        const existingValue = node.properties[propertyName];
        if (config.editable && !node.properties_info.some((info) => info.name === propertyName)) {
            node.addProperty(propertyName, existingValue ?? defaultLabel, "string");
        }
        if (existingValue === undefined || existingValue === null || existingValue === "") {
            node.properties[propertyName] = defaultLabel;
        } else {
            node.properties[propertyName] = existingValue;
        }
    });
}

function getSmartBusLabel(node, index) {
    const fallback = SMART_BUS_DEFAULT_LABELS[index];
    const value = node.properties?.[fallback];
    if (typeof value !== "string") {
        return fallback;
    }

    const trimmed = value.trim();
    return trimmed || fallback;
}

function copySmartBusLabels(sourceNode, targetNode) {
    const sourceConfig = SMART_BUS_NODE_CONFIG[sourceNode.type];
    const targetConfig = SMART_BUS_NODE_CONFIG[targetNode.type];
    if (!sourceConfig || !targetConfig) {
        return;
    }

    ensureSmartBusProperties(sourceNode, sourceConfig);
    ensureSmartBusProperties(targetNode, targetConfig);

    const sharedSlotCount = Math.min(sourceConfig.slotCount, targetConfig.slotCount);
    SMART_BUS_DEFAULT_LABELS.slice(0, sharedSlotCount).forEach((defaultLabel, index) => {
        targetNode.properties[defaultLabel] = getSmartBusLabel(sourceNode, index);
    });
}

function getSmartBusSourceNode(node) {
    const linkId = node.inputs?.[0]?.link;
    if (!linkId || !node.graph?.links) {
        return null;
    }

    const link = node.graph.links[linkId];
    if (!link) {
        return null;
    }

    return node.graph.getNodeById?.(link.origin_id) ?? null;
}

/** Walk bus input chain (BusOut passthrough …) until a SmartBusIn that owns labels. */
function getSmartBusLabelSourceNode(node) {
    let current = getSmartBusSourceNode(node);
    const visited = new Set();
    while (current) {
        if (visited.has(current.id)) {
            return null;
        }
        visited.add(current.id);
        if (isSmartBusInNodeType(current.type)) {
            return current;
        }
        if (isSmartBusOutNodeType(current.type)) {
            current = getSmartBusSourceNode(current);
            continue;
        }
        break;
    }
    return null;
}

function syncSmartBusOutFromSource(node) {
    const labelSource = getSmartBusLabelSourceNode(node);
    if (labelSource) {
        copySmartBusLabels(labelSource, node);
    } else {
        const immediate = getSmartBusSourceNode(node);
        if (immediate && isSmartBusOutNodeType(immediate.type)) {
            copySmartBusLabels(immediate, node);
        }
    }

    const outLinkIds = node.outputs?.[0]?.links ?? [];
    if (!node.graph || !outLinkIds.length) {
        return;
    }
    for (const linkId of outLinkIds) {
        const link = node.graph.links?.[linkId];
        if (!link) {
            continue;
        }
        if (link.target_slot != null && link.target_slot !== 0) {
            continue;
        }
        const targetNode = node.graph.getNodeById?.(link.target_id);
        if (!targetNode || !isSmartBusOutNodeType(targetNode.type)) {
            continue;
        }
        const targetCfg = SMART_BUS_NODE_CONFIG[targetNode.type];
        syncSmartBusOutFromSource(targetNode);
        applySmartBusLabels(targetNode, targetCfg);
    }
}

function syncConnectedSmartBusOutputs(node) {
    const linkIds = node.outputs?.[0]?.links ?? [];
    if (!node.graph || !linkIds.length) {
        return;
    }

    for (const linkId of linkIds) {
        const link = node.graph.links?.[linkId];
        if (!link) {
            continue;
        }

        const targetNode = node.graph.getNodeById?.(link.target_id);
        if (!targetNode || !isSmartBusOutNodeType(targetNode.type)) {
            continue;
        }

        const targetCfg = SMART_BUS_NODE_CONFIG[targetNode.type];
        syncSmartBusOutFromSource(targetNode);
        applySmartBusLabels(targetNode, targetCfg);
    }
}

function applySmartBusLabels(node, config) {
    ensureSmartBusProperties(node, config);

    if (node.inputs?.[0]) {
        node.inputs[0].label = "bus";
    }
    if (node.outputs?.[0]) {
        node.outputs[0].label = "bus";
    }

    SMART_BUS_DEFAULT_LABELS.slice(0, config.slotCount).forEach((_, index) => {
        const label = getSmartBusLabel(node, index);

        if (config.inputOffset !== null) {
            const input = node.inputs?.[config.inputOffset + index];
            if (input) {
                input.label = label;
            }
        }

        if (config.outputOffset !== null) {
            const output = node.outputs?.[config.outputOffset + index];
            if (output) {
                output.label = label;
            }
        }
    });

    if (node.computeSize) {
        node.size = node.computeSize();
    }
    node.setDirtyCanvas?.(true, true);
}

function applySmartModelLoaderGgufVisibility(node, config) {
    const ggufW = findWidgetByName(node, "gguf");
    if (!ggufW || !config) return;
    const isGguf = ggufW.value === true;
    for (const name of config.safetensors) {
        toggleWidget(node, findWidgetByName(node, name), !isGguf);
    }
    for (const name of config.gguf) {
        toggleWidget(node, findWidgetByName(node, name), isGguf);
    }
}

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

        const smartLoaderCfg = SMART_MODEL_LOADER_VISIBILITY[nodeData.name];
        if (smartLoaderCfg) {
            const origComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function () {
                const size = origComputeSize ? origComputeSize.call(this) : [200, 100];
                let height = 50;
                for (const w of this.widgets || []) {
                    if (w.hidden) continue;
                    height += w.computeSize ? w.computeSize()[1] + 4 : 24;
                }
                return [size[0], height];
            };

            const onNodeCreatedLoader = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreatedLoader?.apply(this, arguments);
                const ggufW = findWidgetByName(this, "gguf");
                const sync = () => applySmartModelLoaderGgufVisibility(this, smartLoaderCfg);
                if (ggufW) {
                    const prev = ggufW.callback;
                    ggufW.callback = () => {
                        prev?.apply(ggufW, arguments);
                        sync();
                    };
                }
                sync();
                const onConfigure = this.onConfigure;
                this.onConfigure = function () {
                    const r = onConfigure?.apply(this, arguments);
                    sync();
                    return r;
                };
                return result;
            };
        }

        const smartBusCfg = SMART_BUS_NODE_CONFIG[nodeData.name];
        if (smartBusCfg) {
            const onNodeCreatedBus = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreatedBus?.apply(this, arguments);
                if (isSmartBusOutNodeType(nodeData.name)) {
                    syncSmartBusOutFromSource(this);
                }
                applySmartBusLabels(this, smartBusCfg);
                return result;
            };

            const onConfigureBus = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const result = onConfigureBus?.apply(this, arguments);
                if (isSmartBusOutNodeType(nodeData.name)) {
                    syncSmartBusOutFromSource(this);
                }
                applySmartBusLabels(this, smartBusCfg);
                return result;
            };

            const onPropertyChangedBus = nodeType.prototype.onPropertyChanged;
            nodeType.prototype.onPropertyChanged = function (property, value, prevValue) {
                const result = onPropertyChangedBus?.apply(this, arguments);
                const propertyNames = getSmartBusPropertyNames(smartBusCfg);
                if (isSmartBusInNodeType(nodeData.name) && propertyNames.includes(property)) {
                    this.properties[property] = (typeof value === "string" && value.trim()) ? value.trim() : property;
                    applySmartBusLabels(this, smartBusCfg);
                    syncConnectedSmartBusOutputs(this);
                }
                return result;
            };

            const onConnectionsChangeBus = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function () {
                const result = onConnectionsChangeBus?.apply(this, arguments);
                if (isSmartBusOutNodeType(nodeData.name)) {
                    syncSmartBusOutFromSource(this);
                }
                applySmartBusLabels(this, smartBusCfg);
                if (isSmartBusInNodeType(nodeData.name)) {
                    syncConnectedSmartBusOutputs(this);
                }
                return result;
            };
        }
    }
});
