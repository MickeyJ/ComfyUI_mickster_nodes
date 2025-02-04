import { app } from "../../scripts/app.js";
import { ImageCellWidget } from "./utils_widgets.js";

app.registerExtension({
    name: "mickster_nodes.image_switch_select",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log('\nNode being registered:', nodeData.name, '\n');
        
        if (nodeData.name === "ImageSwitchSelect") {
            
            // Modify the node's constructor
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create arrays for multiple elements
                this.imageElements = new Array(6).fill(null);
                this.uploadButtons = [];
                this.cellWidgets = [];

                // First create all upload buttons
                for (let i = 0; i < 6; i++) {
                    const button = this.addWidget("button", `upload_${i}`, `Upload ${i + 1}`, () => {
                        this.fileInputs[i].click();
                    });
                    
                    this.uploadButtons.push(button);
                }

                // Then create all image widgets
                for (let i = 0; i < 6; i++) {
                    const widget = new ImageCellWidget(this, i);
                    this.addCustomWidget(widget);
                    this.cellWidgets.push(widget);
                }

                // Create hidden file inputs
                this.fileInputs = [];
                for (let i = 0; i < 6; i++) {
                    const fileInput = document.createElement("input");
                    fileInput.type = "file";
                    fileInput.accept = "image/*";
                    fileInput.style.display = "none";
                    fileInput.dataset.index = i;
                    
                    // Handle file selection
                    fileInput.addEventListener("change", (event) => {
                        const file = event.target.files[0];
                        const index = parseInt(event.target.dataset.index);
                        if (file) {
                            // Create a proper path that ComfyUI can understand
                            const relativePath = `${file.name}`;
                            
                            // Update preview for specific grid cell
                            loadImage(this, `/view?filename=${relativePath}&type=input`, index);
                            
                            // Update button text to show filename
                            this.uploadButtons[index].name = `${index + 1}: ${file.name}`;

                            // If this is the currently selected cell, update the widget value
                            if (index === this.selectedIndex) {
                                const imageWidget = this.widgets.find(w => w.name === "image");
                                if (imageWidget) {
                                    imageWidget.value = relativePath;
                                    imageWidget.callback?.(relativePath);
                                }
                            }
                        }
                    });

                    document.body.appendChild(fileInput);
                    this.fileInputs.push(fileInput);
                }

                // In the node constructor, add selected image tracking
                this.selectedIndex = 0;  // Track which image is selected

                return result;
            };

            // Update image loading to handle multiple images
            const loadImage = function(node, src, index) {
                const img = new Image();
                img.onload = () => {
                    node.imageElements[index] = img;
                    node.setDirtyCanvas(true, true);
                };
                img.src = src;
            };

            nodeType.prototype.onExecuted = function(message) {
                console.log("onExecuted called with:", {
                    selectedIndex: this.selectedIndex,
                    message: message
                });
                
                if (message?.detail?.output?.image) {
                    const imagePath = message.detail.output.image;
                    if (!this.imageElements) {
                        this.imageElements = new Array(6).fill(null);
                    }
                    // Only update the selected image
                    if (!this.imageElements[this.selectedIndex]) {
                        console.log("Loading image for index:", this.selectedIndex);
                        loadImage(this, `/view?filename=${imagePath}&type=input`, this.selectedIndex);
                    }
                }
            };

            // Cleanup on node removal
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                // Remove file inputs
                this.fileInputs?.forEach(input => {
                    input.parentElement?.removeChild(input);
                });
                
                if (onRemoved) {
                    onRemoved.apply(this, arguments);
                }
            };

            // Update computeSize to handle single column layout
            nodeType.prototype.computeSize = function(size = [400, 200]) {
                const minWidth = 300;
                const width = Math.max(minWidth, size ? size[0] : minWidth);
                
                // Calculate widget dimensions
                const padding = 10;
                const titleHeight = 30;
                const buttonHeight = 30;
                const buttonSpacing = 5;
                const totalButtonsHeight = 6 * buttonHeight + 5 * buttonSpacing;
                
                // Calculate widget height based on width
                const widgetHeight = (width - padding * 2) * 0.75;  // Match widget aspect ratio
                const widgetSpacing = padding;
                const totalWidgetHeight = 6 * widgetHeight + 5 * widgetSpacing;
                
                // Total height needed
                const totalHeight = titleHeight + totalButtonsHeight + padding * 5 + totalWidgetHeight;
                
                // Return the larger of: calculated height or requested height
                const height = Math.max(totalHeight, size ? size[1] : totalHeight);
                
                return [width, height];
            };

            // Add onResize handler
            nodeType.prototype.onResize = function() {
                console.log("Node resized to:", this.size);  // Add debug log
                this.cellWidgets?.forEach(widget => widget.onResize());
            };

            // Add serialization methods to the node
            nodeType.prototype.serialize = function() {
                // Get base serialization
                const data = LGraphNode.prototype.serialize.call(this);
                
                // Add our custom data
                data.imagePaths = this.uploadButtons.map(b => b.name.split(": ")[1] || "");
                data.selectedIndex = this.selectedIndex;
                
                return data;
            };

            nodeType.prototype.configure = function(serializedData) {
                // Configure base node first
                LGraphNode.prototype.configure.call(this, serializedData);
                
                if (!serializedData) return;
                
                // Restore our custom data
                if (serializedData.imagePaths) {
                    serializedData.imagePaths.forEach((path, i) => {
                        if (path) {
                            this.uploadButtons[i].name = `${i + 1}: ${path}`;
                            loadImage(this, `/view?filename=${path}&type=input`, i);
                        }
                    });
                }
                
                if (typeof serializedData.selectedIndex === "number") {
                    this.selectedIndex = serializedData.selectedIndex;
                }
            };
        }
    }
});