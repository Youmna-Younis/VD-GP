// JavaScript
document.getElementById('chooseFileBtn').addEventListener('click', function() {
    document.getElementById('fileInput').click();
});

document.getElementById('fileInput').addEventListener('change', function() {
    // Do something with the selected file
    console.log('Selected file:', this.files[0]);
});
document.getElementById('fileInput').addEventListener('change', function() {
    const selectedFile = this.files[0];
    const videoElement = document.getElementById('myVideo');
    videoElement.src = URL.createObjectURL(selectedFile);
    videoElement.load(); // Load the new video source
});
