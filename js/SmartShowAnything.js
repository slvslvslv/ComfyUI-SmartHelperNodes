import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Register extension for SmartShowAnything node
app.registerExtension({
    name: "SmartHelperNodes.SmartShowAnything",
    
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "SmartShowAnything") {
            // Function to populate the node with text widgets
            function populate(text) {
                if (this.widgets) {
                    // Find and remove any existing text widgets
                    const pos = this.widgets.findIndex((w) => w.name === "text");
                    if (pos !== -1) {
                        for (let i = pos; i < this.widgets.length; i++) {
                            this.widgets[i].onRemove?.();
                        }
                        this.widgets.length = pos;
                    }
                }

                // Add new text widgets for each item in the text array
                for (const list of text) {
                    const w = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: true }], app).widget;
                    w.inputEl.readOnly = true;
                    w.inputEl.style.opacity = 0.6;
                    w.value = list;
                }

                // Adjust the node size to fit the content
                requestAnimationFrame(() => {
                    const sz = this.computeSize();
                    if (sz[0] < this.size[0]) {
                        sz[0] = this.size[0];
                    }
                    if (sz[1] < this.size[1]) {
                        sz[1] = this.size[1];
                    }
                    this.onResize?.(sz);
                    app.graph.setDirtyCanvas(true, false);
                });
            }

            // When the node is executed, display the text from the message
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                populate.call(this, message.text);
            };

            // When a workflow is loaded, restore the widget state
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                if (this.widgets_values?.length) {
                    populate.call(this, this.widgets_values);
                }
            };
        }
    }
}); 