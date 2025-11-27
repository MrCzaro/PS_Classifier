function previewFile(event) {
    const fileInput = event.target;
    const file = fileInput && fileInput.files && fileInput.files[0];
    if (!file) {
        return;
    }
    const img = document.getElementById("preview-img");
    if (img && img.dataset && img.dataset.objectUrl) {
        try { URL.revokeObjectURL(img.dataset.objectUrl); } catch(e) {}
    }
    const objectUrl = URL.createObjectURL(file);
    img.src = objectUrl;
    const meta = document.getElementById("preview-meta");
    if (meta) meta.textContent = `${file.name} — ${(file.size/1024).toFixed(1)} KB`;
    img.dataset.objectUrl = objectUrl;
    img.classList.remove("hidden");
    const uploadBtn = document.getElementById("upload-btn");
    if (uploadBtn) uploadBtn.classList.remove("hidden");
}

document.addEventListener("DOMContentLoaded", function() {
    const dz = document.getElementById("drop-zone");
    const fileInput = document.getElementById("userfile");

    if (!dz || !fileInput) return;

    dz.addEventListener("click", function() { fileInput.click(); });
    fileInput.addEventListener("change", previewFile);

    dz.addEventListener("dragover", function(e) {
        e.preventDefault();
        dz.classList.add("bg-blue-100");
    });

    dz.addEventListener("dragleave", function(e) {
        dz.classList.remove("bg-blue-100");
    });

    dz.addEventListener("drop", function(e) {
        e.preventDefault();
        dz.classList.remove("bg-blue-100");
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
            try {
                fileInput.files = e.dataTransfer.files;
            } catch(err) {
                try {
                    const dt = new DataTransfer();
                    for (let i=0;i<e.dataTransfer.files.length;i++) dt.items.add(e.dataTransfer.files[i]);
                    fileInput.files = dt.files;
                } catch(e2) {
                    console.warn("Could not set fileInput.files programmatically", e2);
                }
            }
            previewFile({ target: fileInput });
        }
    });

    const img = document.getElementById("preview-img");
    if (img) {
        img.addEventListener("load", function() {
            if (img.dataset && img.dataset.objectUrl) {
                setTimeout(() => {
                    try { URL.revokeObjectURL(img.dataset.objectUrl); } catch(e){}
                    delete img.dataset.objectUrl;
                }, 5000);
            }
        });
    }
});
