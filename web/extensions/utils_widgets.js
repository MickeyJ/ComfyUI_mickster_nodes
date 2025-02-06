import { app } from "../../scripts/app.js";

export class UploadButtonWidget {
    constructor(node, index, callback) {
        this.node = node;
        this.index = index;
        this.type = 'button';
        this.name = `upload_image_${index}`;
        this.value = `Upload Image ${index + 1}`;  // Button text
        this.options = {};
        this.callback = callback;
        this.serializable = false;
        this.configurable = false;
    }
}

export class ImageCellWidget {
    constructor(node, index) {
        this.node = node;
        this.index = index;
        this.name = `cell_${index}`;
        this.type = "IMAGE_CELL";
        this.options = {};
        
        // Calculate dimensions
        const padding = 10;
        const titleHeight = 30;
        const buttonHeight = 30;
        const buttonSpacing = 5;
        const totalButtonsHeight = 6 * buttonHeight + 5 * buttonSpacing;
        const gridStartY = titleHeight + totalButtonsHeight + padding * 5;
        
        const width = node.size[0] - padding * 2;
        this.x = padding;
        this.width = width;
        this.height = width * 0.75;
        this.y = gridStartY + (index * (this.height + padding));  // Position each widget below buttons

        // Add widget properties
        this.serializable = false;
        this.configurable = false;

    }
    
    computeSize() {
        return [this.width || 140, this.height || 30];
    }
    
    draw(ctx) {
        // Draw cell border (dark grey)
        ctx.strokeStyle = "#666";
        ctx.lineWidth = 1;
        ctx.strokeRect(this.x, this.y, this.width, this.height);
        
        // Draw select button as circle on left side
        const buttonRadius = 20;  // Bigger circle
        const buttonX = this.x + buttonRadius + 10;  // 10px padding from left
        const buttonY = this.y + this.height/2;  // Center vertically
        
        // Draw circle
        ctx.beginPath();
        ctx.arc(buttonX, buttonY, buttonRadius, 0, Math.PI * 2);
        
        // Fill based on selection state
        if (this.index === this.node.selectedIndex) {
            ctx.fillStyle = "#00ff00";  // Bright green when selected
        } else {
            ctx.fillStyle = "#2a2a2a";  // Dark grey when not selected
        }
        ctx.fill();
        
        // Draw circle border
        ctx.strokeStyle = "#666";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Update image area calculations for the wider button
        const buttonWidth = (buttonRadius * 2) + 20;  // Total width including padding
        
        // Draw image if exists
        if (this.node.imageElements[this.index]) {
            ctx.save();
            
            // Leave space for button on left side
            const buttonPadding = 10;  // Space between button and image
            const imageAreaX = this.x + buttonWidth + buttonPadding;
            const imageAreaWidth = this.width - buttonWidth - buttonPadding;
            
            // Calculate image dimensions maintaining aspect ratio
            const imgAspect = this.node.imageElements[this.index].width / this.node.imageElements[this.index].height;
            let drawWidth, drawHeight;
            
            if (imgAspect > 4/3) {
                drawWidth = imageAreaWidth;
                drawHeight = imageAreaWidth / imgAspect;
            } else {
                drawHeight = this.height;
                drawWidth = this.height * imgAspect;
                if (drawWidth > imageAreaWidth) {
                    drawWidth = imageAreaWidth;
                    drawHeight = imageAreaWidth / imgAspect;
                }
            }
            
            // Center image in remaining space
            const imageX = imageAreaX + (imageAreaWidth - drawWidth) / 2;
            const imageY = this.y + (this.height - drawHeight) / 2;
            
            ctx.drawImage(this.node.imageElements[this.index], imageX, imageY, drawWidth, drawHeight);
            ctx.restore();
        }
    }

    isInsideWidget(pos) {
        if (!pos) return false;
        const x = pos[0] - this.node.pos[0];
        const y = pos[1] - this.node.pos[1];
        return x >= this.x && x <= this.x + this.width && y >= this.y && y <= this.y + this.height;
    }

    mouse(event, pos, node) {
        if (event.type === "pointerdown") {
            this.node.selectedIndex = this.index;
            // Get the image path from the button text
            const buttonText = this.node.uploadButtons[this.index].name;
            const imagePath = buttonText.split(": ")[1];
            // Update both the image path and selected index
            if (imagePath) {
                // Find and update both widgets
                const widgets = this.node.widgets;
                const imageWidget = widgets.find(w => w.name === "image");
                const indexWidget = widgets.find(w => w.name === "selected_index");
                
                if (imageWidget) {
                    imageWidget.value = imagePath;
                    imageWidget.callback?.(imagePath);
                }
                if (indexWidget) {
                    indexWidget.value = this.index;
                    indexWidget.callback?.(this.index);
                }
            }
            this.node.setDirtyCanvas(true);
            return true;
        }
        return false;
    }

    // Add method to recalculate dimensions
    updateDimensions() {
        const padding = 10;
        const titleHeight = 30;
        const buttonHeight = 30;
        const buttonSpacing = 5;
        const totalButtonsHeight = 6 * buttonHeight + 5 * buttonSpacing;
        const gridStartY = titleHeight + totalButtonsHeight + padding * 5;
        
        // Calculate available height for all widgets
        const availableHeight = this.node.size[1] - gridStartY - padding;
        // Divide available height among 6 widgets with spacing
        const widgetHeight = (availableHeight - (5 * padding)) / 6;
        
        const width = this.node.size[0] - padding * 2;
        this.x = padding;
        this.width = width;
        this.height = widgetHeight;
        this.y = gridStartY + (this.index * (widgetHeight + padding));
    }

    // Add method to handle resize
    onResize() {
        console.log(`Widget ${this.index} resizing`);  // Add debug log
        this.updateDimensions();
    }
} 