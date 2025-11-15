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
                console.log(result);

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

                // Update emotion label
                let maxLogit = -999;
                let maxLabel = "---";
                if (result["logits"] && result["logits"][0]) {
                    for (let i = 0; i < 4; i ++) {
                        if (result["logits"][0][i] > maxLogit) {
                            maxLogit = result["logits"][0][i];
                            maxLabel = result["labels"][i].toUpperCase();
                        }
                    }
                }

                document.getElementById("label").textContent = maxLabel || "---";
                
                // Update status
                statusIndicator.textContent = "● STANDBY";
                statusIndicator.className = "status-standby";
                visualStatus.textContent = "ACTIVE";
                visualStatus.style.color = "var(--atc-green)";

                // Only show spectrogram if we have data
                if (result["spectrogramImageData"] && result["spectrogramImageData"].length > 0 && 
                    result["spectrogramWidth"] > 0 && result["spectrogramHeight"] > 0) {
                    const plotCanvas = document.createElement("canvas");
                    plotCanvas.width = result["spectrogramWidth"];
                    plotCanvas.height = result["spectrogramHeight"];
                    const ctx = plotCanvas.getContext("2d");

                    const imageData = ctx.createImageData(result["spectrogramWidth"], result["spectrogramHeight"]);
                    imageData.data.set(result["spectrogramImageData"]);

                    ctx.putImageData(imageData, 0, 0);

                    const container = document.getElementById("spectrogram-container");

                    plotCanvas.style.width = plotCanvas.width + "px";
                    plotCanvas.style.height = plotCanvas.height + "px";

                    plotCanvas.style.animationName = "plotFade";
                    plotCanvas.style.animationDuration = "0.5s";

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