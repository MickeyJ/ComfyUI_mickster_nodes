import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ImageSelector",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ImageSelector") {
            // Add custom widget
            nodeType.prototype.onNodeCreated = function () {
                // Create containers for each image preview
                this.imageContainers = {};
                for (let i = 1; i <= 10; i++) {
                    const container = document.createElement("div");
                    container.style.width = "100px";
                    container.style.height = "100px";
                    container.style.margin = "5px";
                    container.style.backgroundColor = "#333";
                    container.style.display = "none";  // Hide initially
                    this.imageContainers[`image${i}`] = container;
                }

                // Initialize widgets
                this.widgets = this.widgets || [];
                
                // Add containers after their respective inputs
                const inputs = this.inputs;
                if (inputs) {
                    Object.values(inputs).forEach((input, index) => {
                        if (index < 10) {  // Only for image inputs
                            const imageNum = index + 1;
                            input.container.appendChild(this.imageContainers[`image${imageNum}`]);
                        }
                    });
                }
            };

            // Handle updates from backend
            nodeType.prototype.onExecuted = function (message) {
                if (message.ui) {
                    const { images, selected_index } = message.ui;
                    
                    // Update each image container
                    Object.keys(this.imageContainers).forEach((key, index) => {
                        const container = this.imageContainers[key];
                        if (images[index]) {
                            container.style.display = "block";
                            container.innerHTML = `<img src="/view?filename=${images[index]}&type=input" style="width: 100%; height: 100%; object-fit: cover; border: ${index === selected_index ? '2px solid #00ff00' : 'none'};">`;
                        } else {
                            container.style.display = "none";
                        }
                    });
                }
            };
        }
    }
});