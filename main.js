/* LOGIC FOR NORMAL BROWSER UTILITY */


/* LOGIC FOR UPDATING CANVAS ELEMENTS */

// let isRecording = false;

let isRecording = false;
const statusIndicator = document.getElementById("status-indicator");
const visualStatus = document.getElementById("visual-status");

document.getElementById("play").onclick = () => {
    if (!isRecording) {
        isRecording = true;
        statusIndicator.textContent = "● ON AIR";
        statusIndicator.className = "status-recording";
        visualStatus.textContent = "RECORDING";
        visualStatus.style.color = "var(--atc-red)";
        fetch("/start", { method: "POST" }).then(console.log("Transmission commenced!"));
    }
};
        document.getElementById("stop").onclick = () => {
            if (isRecording) {
                isRecording = false;
                statusIndicator.textContent = "● PROCESSING";
                statusIndicator.className = "status-active";
                visualStatus.textContent = "ANALYZING";
                visualStatus.style.color = "var(--atc-amber)";
                fetch("/stop")
                    .then(r => {
                        if (!r.ok) {
                            return r.text().then(text => {
                                try {
                                    const err = JSON.parse(text);
                                    return Promise.reject(err);
                                } catch (e) {
                                    return Promise.reject({error: `HTTP ${r.status}: ${text}`});
                                }
                            });
                        }
                        return r.json().catch(e => {
                            console.error("Failed to parse JSON response:", e);
                            return r.text().then(text => {
                                console.error("Response text:", text);
                                throw new Error("Invalid JSON response from server");
                            });
                        });
                    })
            .then(result => {
                console.log("Full result:", result);
                console.log("AI Supervisor Report in result:", result["ai_supervisor_report"]);

                // Check for error in response (only show if it's a critical error, not analysis_error)
                if (result.error && !result.success) {
                    alert("Error: " + result.error);
                    console.error("Recording error:", result.error);
                    return;
                }

                // If there's an analysis error but recording succeeded, just log it (don't show alert)
                if (result.analysis_error) {
                    console.warn("Audio analysis unavailable:", result.analysis_error);
                    console.log("Full result object:", JSON.stringify(result, null, 2));
                    document.getElementById("transcription").textContent = "ANALYSIS UNAVAILABLE - TRANSMISSION RECORDED";
                    document.getElementById("label").textContent = "---";
                    document.getElementById("ai-supervisor-report").textContent = "ANALYSIS UNAVAILABLE - TRANSMISSION RECORDED";
                    statusIndicator.textContent = "● STANDBY";
                    statusIndicator.className = "status-standby";
                    visualStatus.textContent = "ACTIVE";
                    visualStatus.style.color = "var(--atc-green)";
                    return;
                }

                // Check if required fields exist (but be lenient if analysis failed)
                if (!result["logits"]) {
                    console.warn("Missing logits in response");
                    return;
                }

                // Update transcription
                const transcriptionText = result["transcription"] || "NO TRANSMISSION DETECTED";
                document.getElementById("transcription").textContent = transcriptionText || "NO TRANSMISSION DETECTED";

                // Update AI Supervisor Report
                const aiReport = result["ai_supervisor_report"] || "AWAITING ANALYSIS...";
                console.log("AI Supervisor Report:", aiReport);
                const reportElement = document.getElementById("ai-supervisor-report");
                if (reportElement) {
                    reportElement.textContent = aiReport;
                } else {
                    console.error("AI supervisor report element not found!");
                }

                // Update emotion label with percentages
                if (result["logits"] && result["logits"][0]) {
                    const logits = result["logits"][0];
                    
                    // Convert logits to probabilities using softmax
                    const maxLogit = Math.max(...logits);
                    const expLogits = logits.map(logit => Math.exp(logit - maxLogit));
                    const sumExp = expLogits.reduce((a, b) => a + b, 0);
                    const probabilities = expLogits.map(exp => exp / sumExp);
                    
                    // Map emotions: [Neutral, Happy, Angry, Sad]
                    // Active = Happy only (index 1)
                    // Fatigued = Neutral (index 0) + Angry (index 2) + Sad (index 3)
                    const activeProb = probabilities[1]; // Happy only
                    const fatiguedProb = probabilities[0] + probabilities[2] + probabilities[3]; // Neutral + Angry + Sad
                    
                    // Format as percentages
                    const activePercent = (activeProb * 100).toFixed(1);
                    const fatiguedPercent = (fatiguedProb * 100).toFixed(1);
                    
                    document.getElementById("label").innerHTML = 
                        `ACTIVE: ${activePercent}%<br>FATIGUED: ${fatiguedPercent}%`;
                } else {
                    document.getElementById("label").textContent = "---";
                }
                
                // Update status
                statusIndicator.textContent = "● STANDBY";
                statusIndicator.className = "status-standby";
                visualStatus.textContent = "ACTIVE";
                visualStatus.style.color = "var(--atc-green)";

                // Only show spectrogram if we have data
                if (result["spectrogramImageData"] && result["spectrogramImageData"].length > 0 && 
                    result["spectrogramWidth"] > 0 && result["spectrogramHeight"] > 0) {
                    const container = document.getElementById("spectrogram-container");
                    
                    // Get container dimensions BEFORE creating canvas to prevent layout shift
                    const containerStyle = window.getComputedStyle(container);
                    const paddingLeft = parseFloat(containerStyle.paddingLeft) || 0;
                    const paddingRight = parseFloat(containerStyle.paddingRight) || 0;
                    const containerWidth = container.clientWidth || container.offsetWidth || 0;
                    const availableWidth = containerWidth - paddingLeft - paddingRight;
                    
                    // Get existing canvases and calculate their total width
                    const existingCanvases = container.querySelectorAll("canvas");
                    const existingTotalWidth = Array.from(existingCanvases).reduce((sum, canvas) => {
                        return sum + (parseFloat(canvas.style.width) || canvas.offsetWidth || 0);
                    }, 0);
                    
                    // Calculate width to maintain aspect ratio BEFORE creating canvas
                    const aspectRatio = result["spectrogramWidth"] / result["spectrogramHeight"];
                    const naturalWidth = 150 * aspectRatio;
                    
                    // Pre-calculate if we need to remove old canvases or scale down
                    const totalWidthWithNew = existingTotalWidth + naturalWidth;
                    let needsScaling = false;
                    let scaleFactor = 1;
                    
                    if (totalWidthWithNew > availableWidth && availableWidth > 0) {
                        if (existingCanvases.length > 0) {
                            // Remove leftmost canvas first
                            const leftmostCanvas = existingCanvases[0];
                            const leftmostWidth = parseFloat(leftmostCanvas.style.width) || leftmostCanvas.offsetWidth || 0;
                            const widthAfterRemoval = existingTotalWidth - leftmostWidth + naturalWidth;
                            
                            if (widthAfterRemoval > availableWidth) {
                                // Still too wide, need to scale
                                needsScaling = true;
                                scaleFactor = availableWidth / widthAfterRemoval;
                            } else {
                                // Just remove the leftmost one
                                leftmostCanvas.remove();
                            }
                        } else {
                            // No existing canvases, but new one is too wide - scale it
                            needsScaling = true;
                            scaleFactor = availableWidth / naturalWidth;
                        }
                    }
                    
                    // Now create the canvas with pre-calculated dimensions
                    const plotCanvas = document.createElement("canvas");
                    plotCanvas.width = result["spectrogramWidth"];
                    plotCanvas.height = result["spectrogramHeight"];
                    const ctx = plotCanvas.getContext("2d");

                    const imageData = ctx.createImageData(result["spectrogramWidth"], result["spectrogramHeight"]);
                    imageData.data.set(result["spectrogramImageData"]);

                    ctx.putImageData(imageData, 0, 0);

                    // Set dimensions immediately to prevent layout shift
                    const finalWidth = needsScaling ? (naturalWidth * scaleFactor) : naturalWidth;
                    plotCanvas.style.height = "150px";
                    plotCanvas.style.maxHeight = "150px";
                    plotCanvas.style.minHeight = "150px";
                    plotCanvas.style.width = finalWidth + "px";
                    plotCanvas.style.flexShrink = "0";
                    plotCanvas.style.flexGrow = "0";
                    plotCanvas.style.display = "block";

                    plotCanvas.style.animationName = "plotFade";
                    plotCanvas.style.animationDuration = "0.5s";

                    // If we need to scale existing canvases, do it before adding new one
                    if (needsScaling && existingCanvases.length > 0) {
                        existingCanvases.forEach(canvas => {
                            const currentWidth = parseFloat(canvas.style.width) || canvas.offsetWidth || 0;
                            canvas.style.width = (currentWidth * scaleFactor) + "px";
                        });
                    }
                    
                    // Add the new canvas to the right
                    container.appendChild(plotCanvas);
                }
            })
            .catch(error => {
                console.error("Error stopping recording:", error);
                console.error("Error details:", JSON.stringify(error, null, 2));
                
                // Don't show alert if it's just an analysis error - recording still succeeded
                if (error.analysis_error || (error.error && (error.error.toLowerCase().includes("torchcodec") || error.error.toLowerCase().includes("load")))) {
                    console.warn("Audio analysis failed, but recording was successful:", error.analysis_error || error.error);
                    document.getElementById("transcription").textContent = "(Analysis unavailable)";
                    document.getElementById("label").textContent = "(Analysis unavailable)";
                    return;
                }
                
                // Check if it's a network/fetch error
                if (error.message && error.message.includes("Load failed")) {
                    console.error("Network error - check server logs");
                    document.getElementById("transcription").textContent = "COMMUNICATION ERROR - CHECK CONSOLE";
                    document.getElementById("label").textContent = "---";
                    statusIndicator.textContent = "● ERROR";
                    statusIndicator.className = "status-standby";
                    visualStatus.textContent = "ERROR";
                    visualStatus.style.color = "var(--atc-red)";
                    return;
                }
                
                // Only show alert for critical errors
                alert("Error stopping recording: " + (error.error || error.message || "Unknown error"));
            });
    }
};

// recordButton.onclick = () => {
//     if (isRecording) {
//         fetch("/stop").then(r => r.json()).then(() => {
//             // do something
//         });
//         isRecording = false;
//     } else {
//         fetch("/start", { method: "POST" }).then(console.log("Started recording!"));
//         isRecording = true;
//     }
// };

/* LOGIC FOR WEBRTC VIDEO AND AUDIO STREAMING */

class WebRTCManager {
    /**
     * @type {WebRTCManager} Saved instance
     */
    static instance;

    /**
     * Construct a new WebRTC manager. Initializes the RTCPeerConnection object
     * and prepares the callbacks necessary to direct incoming video streams
     * to the corresponding <video> element.
     */
    constructor() {
        this.pc = new RTCPeerConnection({
            sdpSemantics: "unified-plan",
            iceServers: [
                { urls: ['stun:stun.l.google.com:19302']}
            ]
        });

        this.pc.addEventListener("track", event => {
            if (event.track.kind == "video") {
                document.getElementById("video").srcObject = event.streams[0];
            }
        });

        WebRTCManager.instance = this;
    }

    /**
     * Start the WebRTC video stream
     */
    async start() {

        const userMedia = await navigator.mediaDevices.getUserMedia({
            audio: true,
            video: true
        });

        const tracks = userMedia.getTracks();

        for (let i = 0; i < tracks.length; i ++) {
            this.pc.addTrack(tracks[i], userMedia);
        }

        const offer = await this.pc.createOffer();

        await this.pc.setLocalDescription(offer);

        await new Promise(resolve => {
            if (this.pc.iceGatheringState == "complete") {
                resolve()
            } else {
                const iceStateChangeListener = () => {
                    if (this.pc.iceGatheringState == "complete") {
                        this.pc.removeEventListener("icegatheringstatechange", iceStateChangeListener);
                        resolve();
                    }
                };
                this.pc.addEventListener("icegatheringstatechange", iceStateChangeListener);
            }
        });

        const description = this.pc.localDescription;
        
        console.log("Sending offer");

        const response = await fetch("/offer", {
            body: JSON.stringify({
                sdp: description.sdp,
                type: description.type
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });

        const responseJson = await response.json();

        await this.pc.setRemoteDescription(responseJson);
    }

    /**
     * Stop the WebRTC video stream
     */
    async stop() {
        if (this.pc.getTransceivers) {
            this.pc.getTransceivers().forEach((transceiver) => {
                if (transceiver.stop) {
                    transceiver.stop();
                }
            });
        }

        this.pc.getSenders().forEach(sender => sender.track.stop());

        setTimeout(() => this.pc.close(), 500);
    }
}

/* Let's light this candle! */
if (true) {
    (new WebRTCManager()).start().then(() => {
        console.log("Connected!");

        setTimeout(() => {
            const airplaneContainer = document.getElementById("airplane-container");
            const coverBackground = document.getElementById("cover-background");

            // Fade out the airplane and background
            airplaneContainer.style.opacity = "0%";
            coverBackground.style.opacity = "0%";

            setTimeout(() => {
                airplaneContainer.style.display = "none";
                coverBackground.style.display = "none";
            }, 1000);
        }, 2000); // Wait for animation to complete (2s)
    });
}