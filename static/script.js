// Simple frontend enhancements

document.addEventListener("DOMContentLoaded", function () {
    // Highlight active buttons on hover
    let buttons = document.querySelectorAll("button");
    buttons.forEach(btn => {
        btn.addEventListener("mouseover", () => {
            btn.style.transform = "scale(1.05)";
            btn.style.transition = "0.2s";
        });
        btn.addEventListener("mouseout", () => {
            btn.style.transform = "scale(1)";
        });
    });

    // Signup form validation
    let signupForm = document.querySelector("form[action='/signup']");
    if (signupForm) {
        signupForm.addEventListener("submit", function (e) {
            let password = signupForm.querySelector("input[name='password']").value;
            if (password.length < 6) {
                alert("Password must be at least 6 characters long!");
                e.preventDefault();
            }
        });
    }

    // Login form validation
    let loginForm = document.querySelector("form[action='/login']");
    if (loginForm) {
        loginForm.addEventListener("submit", function (e) {
            let email = loginForm.querySelector("input[name='email']").value;
            if (!email.includes("@")) {
                alert("Please enter a valid email address!");
                e.preventDefault();
            }
        });
    }

    // --- Gesture Recognition specific code ---
    const video = document.getElementById("video");
    const predictionDiv = document.getElementById("prediction");

    // Only run webcam and prediction logic if video element exists on the page
    if (video && predictionDiv) {
        // Start webcam
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(err => {
                console.error("Error accessing webcam: ", err);
                predictionDiv.innerText = "Error: Could not access webcam. Please allow camera access.";
            });

        // Capture frame and send to backend every 500ms
        setInterval(() => {
            if (video.readyState === video.HAVE_ENOUGH_DATA) { // Ensure video is ready
                const canvas = document.createElement("canvas");
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataUrl = canvas.toDataURL("image/jpeg", 0.7); // 0.7 quality for faster transfer

                fetch("/predict_frame", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image: dataUrl })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.error) {
                        predictionDiv.innerText = "Error: " + data.error;
                        predictionDiv.style.color = 'red';
                    } else {
                        // Display prediction only if confidence is above a threshold
                        // Or if no hand is detected
                        if (data.label === "No Hand Detected" || (data.confidence && data.confidence < 0.7)) { // Example threshold
                            predictionDiv.innerText = "Waiting for confident gesture...";
                            predictionDiv.style.color = 'orange';
                        } else {
                            predictionDiv.innerText = `Prediction: ${data.label} (${(data.confidence * 100).toFixed(1)}%)`;
                            predictionDiv.style.color = 'green';
                        }
                    }
                })
                .catch(err => {
                    console.error("Error fetching prediction: ", err);
                    predictionDiv.innerText = "Error connecting to server or network issue.";
                    predictionDiv.style.color = 'red';
                });
            }
        }, 500); // Send frame every 500ms
    }
});