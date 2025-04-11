// Hiển thị loading spinner khi form được submit
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form[method="POST"]');
    if (form && form.action.includes('anonymize')) {
        form.addEventListener('submit', function() {
            document.querySelector('.loading-overlay').style.display = 'flex';
        });
    }

    // Cập nhật tên file đã chọn
    const fileInput = document.querySelector('input[type="file"]');
    const fileLabel = document.querySelector('.file-input-label');
    if (fileInput && fileLabel) {
        fileInput.addEventListener('change', function() {
            fileLabel.textContent = this.files[0] ? this.files[0].name : 'Chưa chọn file';
        });
    }
});