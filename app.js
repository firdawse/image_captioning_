// Select the file input, the model select dropdown, and the predict button
const fileInput = document.querySelector('input[type="file"]');
const modelSelect = document.querySelector('#model-select');
const predictButton = document.querySelector('#predict-btn');
const resultDiv = document.querySelector('#result');

predictButton.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file!');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelSelect.value);

    // Fetch data from the Flask server
    const response = await fetch('http://localhost:7000/predict', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        console.error('Server response:', response);
        return;
    }

    // Parse the JSON response
    const data = await response.json();

    // Display the response
    resultDiv.textContent = JSON.stringify(data, null, 2);
});

// Select the file input, the predict button, and the preview image

const previewImage = document.querySelector('#preview');

// Update the preview image when a file is selected
fileInput.addEventListener('change', function () {
    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result;
        }
        reader.readAsDataURL(file);
    }
});



