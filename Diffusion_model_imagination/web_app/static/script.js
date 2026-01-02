
document.addEventListener('DOMContentLoaded', () => {
    const strengthInput = document.getElementById('strength');
    const strengthVal = document.getElementById('strength-val');
    const guidanceInput = document.getElementById('guidance');
    const guidanceVal = document.getElementById('guidance-val');
    const imageUpload = document.getElementById('image-upload');
    const sourcePreview = document.getElementById('source-preview');
    const generateBtn = document.getElementById('generate-btn');
    const resultPreview = document.getElementById('result-preview');
    const loading = document.getElementById('loading');
    const saveBtn = document.getElementById('save-btn');

    // Update slider values
    strengthInput.addEventListener('input', () => strengthVal.textContent = strengthInput.value);
    guidanceInput.addEventListener('input', (e) => guidanceVal.textContent = e.target.value);

    // Handle Image Upload Preview
    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                sourcePreview.innerHTML = `<img src="${e.target.result}" alt="Source Image">`;
            };
            reader.readAsDataURL(file);
        }
    });

    // Handle Generate
    generateBtn.addEventListener('click', async () => {
        const file = imageUpload.files[0];
        const prompt = document.getElementById('prompt').value;
        const strength = strengthInput.value;
        const guidance = guidanceInput.value;

        if (!file) {
            alert("Please upload a source image.");
            return;
        }
        if (!prompt) {
            alert("Please enter a target prompt.");
            return;
        }



        // UI State
        generateBtn.disabled = true;
        loading.classList.remove('hidden');
        resultPreview.innerHTML = '';
        saveBtn.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', file);
        formData.append('prompt', prompt);
        formData.append('strength', strength);
        formData.append('guidance_scale', guidance);

        try {
            console.log("Sending request...");
            const response = await fetch('/generate', {
                method: 'POST',
                body: formData
            });

            console.log("Response status:", response.status);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || "Generation failed");
            }

            console.log("Parsing blob...");
            const blob = await response.blob();
            console.log("Blob size:", blob.size);

            if (blob.size === 0) {
                throw new Error("Received empty image blob");
            }

            const imageUrl = URL.createObjectURL(blob);
            console.log("Image URL:", imageUrl);

            // Safer render
            resultPreview.innerHTML = '';
            const img = document.createElement('img');
            img.src = imageUrl;
            img.alt = "Generated Image";
            img.onload = () => console.log("Image loaded successfully in DOM");
            img.onerror = (e) => {
                console.error("Image failed to load in DOM", e);
                alert("Image failed to render. See console.");
            };
            resultPreview.appendChild(img);

            document.getElementById('save-btn').classList.remove('hidden');
        } catch (error) {
            console.error("Generation Error:", error);
            alert(`Error: ${error.message}`);
            resultPreview.innerHTML = `<span>Error generating image: ${error.message}</span>`;
            document.getElementById('save-btn').classList.add('hidden');
        } finally {
            generateBtn.disabled = false;
            loading.classList.add('hidden');
        }
    });

    // Save Button Logic
    saveBtn.addEventListener('click', () => {
        const resultImg = resultPreview.querySelector('img');
        if (resultImg) {
            const link = document.createElement('a');
            link.href = resultImg.src;
            link.download = 'generated_image.png';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    });
});

// Zoom Logic
let zoomLevels = {
    'source-img': 1.0,
    'result-img': 1.0
};

function zoomImage(imgId, delta) {
    // Determine which image we are zooming
    // Since images are dynamically added, we target the <img> inside the preview div.
    // HTML onclick passes 'source-img' or 'result-img' as identifiers.
    // But wait, the <img> inside source-preview doesn't have an ID initially.
    // We should make sure we target the correct element.

    // Mapping identifiers to preview container IDs
    const containerId = imgId === 'source-img' ? 'source-preview' : 'result-preview';
    const container = document.getElementById(containerId);
    const img = container ? container.querySelector('img') : null;

    if (img) {
        if (!zoomLevels[imgId]) zoomLevels[imgId] = 1.0;
        zoomLevels[imgId] += delta;
        // Clamp zoom
        if (zoomLevels[imgId] < 0.1) zoomLevels[imgId] = 0.1;
        if (zoomLevels[imgId] > 5.0) zoomLevels[imgId] = 5.0;

        img.style.transform = `scale(${zoomLevels[imgId]})`;

        // Ensure container overflow is hidden (already in CSS)
    }
}

function resetZoom(imgId) {
    const containerId = imgId === 'source-img' ? 'source-preview' : 'result-preview';
    const container = document.getElementById(containerId);
    const img = container ? container.querySelector('img') : null;

    if (img) {
        zoomLevels[imgId] = 1.0;
        img.style.transform = `scale(1.0)`;
    }
}

// Attach to window so onclick works
window.zoomImage = zoomImage;
window.resetZoom = resetZoom;
