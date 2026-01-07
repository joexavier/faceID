/**
 * Bounding Box Editor
 * Allows users to select and resize face bounding boxes on images
 */

class BoundingBoxEditor {
    constructor(containerId, imageUrl, faces = []) {
        this.container = document.getElementById(containerId);
        this.faces = faces;
        this.selectedFaceId = null;
        this.imageScale = 1;
        this.onFaceSelect = null;
        this.onBboxChange = null;

        this.init(imageUrl);
    }

    init(imageUrl) {
        // Create image element
        this.image = document.createElement('img');
        this.image.src = imageUrl;
        this.image.style.maxWidth = '100%';
        this.image.style.display = 'block';

        this.image.onload = () => {
            this.imageScale = this.image.naturalWidth / this.image.clientWidth;
            this.renderFaces();
        };

        this.container.appendChild(this.image);
        this.container.style.position = 'relative';
        this.container.style.display = 'inline-block';
    }

    setFaces(faces) {
        this.faces = faces;
        this.renderFaces();
    }

    renderFaces() {
        // Remove existing overlays
        this.container.querySelectorAll('.face-overlay').forEach(el => el.remove());

        // Create overlays for each face
        this.faces.forEach(face => {
            this.createOverlay(face);
        });
    }

    createOverlay(face) {
        const bbox = face.bbox;
        const overlay = document.createElement('div');
        overlay.className = 'face-overlay';
        overlay.dataset.faceId = face.id;

        // Scale coordinates from image to display
        const left = bbox.left / this.imageScale;
        const top = bbox.top / this.imageScale;
        const width = bbox.width / this.imageScale;
        const height = bbox.height / this.imageScale;

        Object.assign(overlay.style, {
            position: 'absolute',
            left: left + 'px',
            top: top + 'px',
            width: width + 'px',
            height: height + 'px',
            border: '3px solid #ff0000',
            cursor: 'pointer',
            boxSizing: 'border-box'
        });

        // Add resize handles
        ['nw', 'ne', 'sw', 'se'].forEach(pos => {
            const handle = document.createElement('div');
            handle.className = `resize-handle ${pos}`;
            Object.assign(handle.style, {
                position: 'absolute',
                width: '12px',
                height: '12px',
                background: '#00ff00',
                border: '2px solid #fff',
                display: 'none'
            });

            if (pos.includes('n')) handle.style.top = '-6px';
            if (pos.includes('s')) handle.style.bottom = '-6px';
            if (pos.includes('w')) handle.style.left = '-6px';
            if (pos.includes('e')) handle.style.right = '-6px';

            handle.style.cursor = pos + '-resize';
            overlay.appendChild(handle);
        });

        // Click to select
        overlay.addEventListener('click', (e) => {
            e.stopPropagation();
            this.selectFace(face.id);
        });

        // Make resizable
        this.makeResizable(overlay, face.id);

        this.container.appendChild(overlay);
    }

    selectFace(faceId) {
        this.selectedFaceId = faceId;

        // Update visual state
        this.container.querySelectorAll('.face-overlay').forEach(el => {
            const isSelected = parseInt(el.dataset.faceId) === faceId;
            el.style.borderColor = isSelected ? '#00ff00' : '#ff0000';
            el.style.borderWidth = isSelected ? '4px' : '3px';
            el.querySelectorAll('.resize-handle').forEach(h => {
                h.style.display = isSelected ? 'block' : 'none';
            });
        });

        // Callback
        if (this.onFaceSelect) {
            this.onFaceSelect(faceId);
        }
    }

    makeResizable(overlay, faceId) {
        const handles = overlay.querySelectorAll('.resize-handle');
        let isResizing = false;
        let startX, startY, startWidth, startHeight, startLeft, startTop;
        let activeHandle;

        handles.forEach(handle => {
            handle.addEventListener('mousedown', (e) => {
                e.stopPropagation();
                isResizing = true;
                activeHandle = handle.className.split(' ')[1];
                startX = e.clientX;
                startY = e.clientY;
                startWidth = overlay.offsetWidth;
                startHeight = overlay.offsetHeight;
                startLeft = overlay.offsetLeft;
                startTop = overlay.offsetTop;

                document.addEventListener('mousemove', resize);
                document.addEventListener('mouseup', stopResize);
            });
        });

        const resize = (e) => {
            if (!isResizing) return;

            const dx = e.clientX - startX;
            const dy = e.clientY - startY;

            if (activeHandle.includes('e')) {
                overlay.style.width = Math.max(20, startWidth + dx) + 'px';
            }
            if (activeHandle.includes('w')) {
                const newWidth = startWidth - dx;
                if (newWidth > 20) {
                    overlay.style.width = newWidth + 'px';
                    overlay.style.left = startLeft + dx + 'px';
                }
            }
            if (activeHandle.includes('s')) {
                overlay.style.height = Math.max(20, startHeight + dy) + 'px';
            }
            if (activeHandle.includes('n')) {
                const newHeight = startHeight - dy;
                if (newHeight > 20) {
                    overlay.style.height = newHeight + 'px';
                    overlay.style.top = startTop + dy + 'px';
                }
            }
        };

        const stopResize = () => {
            if (!isResizing) return;
            isResizing = false;

            document.removeEventListener('mousemove', resize);
            document.removeEventListener('mouseup', stopResize);

            // Calculate new bbox in image coordinates
            const newBbox = {
                left: Math.round(overlay.offsetLeft * this.imageScale),
                top: Math.round(overlay.offsetTop * this.imageScale),
                right: Math.round((overlay.offsetLeft + overlay.offsetWidth) * this.imageScale),
                bottom: Math.round((overlay.offsetTop + overlay.offsetHeight) * this.imageScale)
            };

            // Callback
            if (this.onBboxChange) {
                this.onBboxChange(faceId, newBbox);
            }
        };
    }

    getSelectedFaceId() {
        return this.selectedFaceId;
    }

    getScaledBbox(faceId) {
        const overlay = this.container.querySelector(`[data-face-id="${faceId}"]`);
        if (!overlay) return null;

        return {
            left: Math.round(overlay.offsetLeft * this.imageScale),
            top: Math.round(overlay.offsetTop * this.imageScale),
            right: Math.round((overlay.offsetLeft + overlay.offsetWidth) * this.imageScale),
            bottom: Math.round((overlay.offsetTop + overlay.offsetHeight) * this.imageScale)
        };
    }
}

// Export for use in templates
window.BoundingBoxEditor = BoundingBoxEditor;
