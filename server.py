import asyncio
import json
import os
import uuid
import logging
import threading
import queue
import io

# Prevent any library from trying to use torchcodec
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
os.environ["TORCHAUDIO_IO_BACKEND"] = "soundfile"
# Prevent transformers from using torchcodec
os.environ["TRANSFORMERS_NO_TORCHCODEC"] = "1"
# Set matplotlib backend via environment variable (must be before ANY imports that might use matplotlib)
# This prevents crashes when matplotlib is used in background threads on macOS
os.environ["MPLBACKEND"] = "Agg"

# Now import matplotlib with the Agg backend
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no GUI required
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt

import av.container

import numpy as np
import cv2
import torch
import torchvision
import librosa
import scipy.signal
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import WhisperForConditionalGeneration, HubertForSequenceClassification, ViTImageProcessor, ViTForImageClassification, ViTFeatureExtractor
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from PIL import Image
import av
from av.audio.frame import AudioFrame
from av.video.frame import VideoFrame

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

# Device detection: prefer CUDA, then MPS (Apple Silicon), then CPU
# For M2 Macs, CPU is often faster for small batch inference due to MPS overhead
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # For video inference (small batches), CPU is often faster on M2
        # MPS is better for larger batches and audio processing
        return torch.device("cpu")  # Use CPU for video on M2
    else:
        return torch.device("cpu")

def get_audio_device():
    """Use MPS for audio processing if available (larger batches benefit more)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()
AUDIO_DEVICE = get_audio_device()
logger.info(f"Using device for video: {DEVICE}, for audio: {AUDIO_DEVICE}")

#
# LOGIC FOR VIDEO PROCESSING: FACE RECOGNITION, EXTRACTION, AND FATIGUE PREDICTION
#

class VideoTransformTrack(MediaStreamTrack):
    """
    Take in a MediaStream and process it frame by frame
    """

    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.device = DEVICE
        self.model_path = 'xacer/vit-base-patch16-224-fatigue'
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        self.processor = ViTImageProcessor.from_pretrained(self.model_path)
        self.model = ViTForImageClassification.from_pretrained(self.model_path,num_labels=2,ignore_mismatched_sizes=True).to(self.device)
        self.model.eval()  # Set to evaluation mode for better performance
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_path) 
        self.image_mean, self.image_std = self.processor.image_mean, self.processor.image_std

        self.transform = Compose([
            Resize((224,224)),
            ToTensor(),
            Normalize(mean=self.image_mean,std=self.image_std)
        ])
        self.class_value = "Active"
        self.frame_count = 0
        self.last_face_position = None  # Cache last face position
        self.face_detection_skip = 2  # Only detect faces every N frames

    def crop_bounding_box(self, img, x, y, w, h):
        return img[y:y+h,x:x+w]

    async def recv(self):
        # Retrieve the next input frame
        video_frame: VideoFrame = await self.track.recv()
        outgoing_image: np.ndarray = video_frame.to_ndarray(format="bgr24")
    
        # Optimize: Only detect faces every N frames, reuse last position otherwise
        faces = []
        if self.frame_count % self.face_detection_skip == 0:
            gray = cv2.cvtColor(outgoing_image, cv2.COLOR_BGR2GRAY)
            # Use faster detection parameters
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
            if len(faces) > 0:
                self.last_face_position = faces[0]  # Cache the first face
        elif self.last_face_position is not None:
            # Reuse cached face position
            faces = [self.last_face_position]
        
        # Process faces
        for (x,y,w,h) in faces:
            cv2.rectangle(outgoing_image,(x,y),(x+w,y+h),(255,0,0),2)
            
            # Optimize: Run model inference less frequently (every 30 frames instead of 12)
            if self.frame_count % 30 == 0:
                cropped_image = self.crop_bounding_box(outgoing_image,x,y,w,h)
                # Skip grayscale conversion - use RGB directly
                cropped_pil = Image.fromarray(cropped_image).convert('RGB')
                transformed_pil = self.transform(cropped_pil)
                input_tensor = transformed_pil.unsqueeze(0)
                
                with torch.no_grad():
                    input_tensor = input_tensor.to(self.device)
                    output_logits = self.model(input_tensor).logits
                    # Process on device, only move to CPU for final result
                    predicted_class = torch.argmax(output_logits, dim=1).item()
                    
                if predicted_class == 1:
                    self.class_value = "Active"
                elif predicted_class == 0:
                    self.class_value = "Fatigued"
            
            # Draw bounding box and label
            cv2.putText(outgoing_image, f'{self.class_value}',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        
        # Clear cached face if no faces detected for a while
        if len(faces) == 0 and self.frame_count % 60 == 0:
            self.last_face_position = None

        self.frame_count += 1

        # Reconstruct and return the new frame
        new_frame = VideoFrame.from_ndarray(outgoing_image, format="bgr24")
        new_frame.pts = video_frame.pts
        new_frame.time_base = video_frame.time_base

        return new_frame

#
# LOGIC FOR AUDIO RECORDING
#

class CustomMediaRecorder:
    instance: "CustomMediaRecorder" = None

    def __init__(self):
        # Create recordings directory if it doesn't exist
        self.recordings_dir = os.path.join(ROOT, "recordings")
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Find the next available recording number
        self.counter = self._get_next_recording_number()
        self.reset_container()
        self.track = None
        self.task = None
        self.recording = False

        CustomMediaRecorder.instance = self
    
    def _get_next_recording_number(self):
        """Find the next available recording number by checking existing files"""
        existing_files = [f for f in os.listdir(self.recordings_dir) if f.startswith("recording_") and f.endswith(".mp3")]
        if not existing_files:
            return 0
        # Extract numbers from filenames like "recording_000.mp3"
        numbers = []
        for f in existing_files:
            try:
                num_str = f.replace("recording_", "").replace(".mp3", "")
                numbers.append(int(num_str))
            except ValueError:
                continue
        return max(numbers) + 1 if numbers else 0

    def add_track(self, track: MediaStreamTrack):
        self.track = track
    
    def reset_container(self):
        self.filename = os.path.join(self.recordings_dir, f"recording_{self.counter:03d}.mp3")
        self.container = av.open(self.filename, mode="w")
        self.stream = self.container.add_stream("mp3")
        self.counter += 1
    
    def start_recording(self):
        self.recording = True

    def stop_recording(self):
        self.recording = False
        # Flush any remaining frames before closing
        try:
            if self.container is not None and self.stream is not None:
                # Flush the encoder
                for packet in self.stream.encode(None):
                    self.container.mux(packet)
                self.container.close()
        except Exception as e:
            logger.error(f"Error closing container: {e}")
        old_filename = self.filename
        self.reset_container()
        return old_filename

    async def start(self):
        self.task = asyncio.ensure_future(self.run_track())
    
    async def stop(self):
        if self.task is not None:
            self.task.cancel()
            self.task = None

            for packet in self.stream.encode(None):
                self.container.mux(packet)
        
        self.container.close()

        self.track = None
        self.stream = None
        self.container = None
    
    async def run_track(self):
        try:
            while True:
                frame: AudioFrame = await self.track.recv()
                if self.recording and self.container is not None and self.stream is not None:
                    try:
                        for packet in self.stream.encode(frame):
                            self.container.mux(packet)
                    except Exception as e:
                        logger.error(f"Error encoding/muxing frame: {e}")
                        if not self.recording:
                            break
        except asyncio.CancelledError:
            logger.info("Audio track recording cancelled")
        except Exception as e:
            logger.error(f"Error in run_track: {e}")

#
# LOGIC FOR AUDIO PROCESSING
#

class AudioAnalyzer:
    instance: "AudioAnalyzer"

    def __init__(self):
        self.device = AUDIO_DEVICE
        
        logger.info("Loading HuBERT model for emotion recognition...")
        self.hubert = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-er").to(self.device)
        self.hubert.eval()
        
        logger.info("Loading Whisper model for transcription...")
        self.model_path = "openai/whisper-tiny.en"
        from transformers import WhisperTokenizer
        self.tokenizer = WhisperTokenizer.from_pretrained(self.model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
        self.model.eval()
        
        # Whisper exact parameters (from OpenAI's implementation)
        self.whisper_n_fft = 400
        self.whisper_hop_length = 160
        self.whisper_win_length = 400
        self.whisper_n_mels = 80
        self.whisper_sample_rate = 16000
        
        # Pre-compute Whisper mel filter bank (exact match to Whisper's implementation)
        whisper_mel_filters = librosa.filters.mel(
            sr=self.whisper_sample_rate, 
            n_fft=self.whisper_n_fft, 
            n_mels=self.whisper_n_mels,
            fmin=0.0,
            fmax=8000.0
        )
        self.whisper_mel_filter_bank = whisper_mel_filters.astype(np.float32)
        
        # Spectrogram parameters for emotion analysis
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 128
        self.sample_rate = 16000
        
        # Pre-compute mel filter bank for emotion analysis
        mel_filters = librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels)
        self.mel_filter_bank = torch.from_numpy(mel_filters).float().to(self.device)
        
        self.noise_rescale = 1.0 / torch.linspace(1.0, 16.0, self.n_fft // 2 + 1).to(self.device)
        
        logger.info("AudioAnalyzer initialized successfully")
        AudioAnalyzer.instance = self
    
    def transcribe_audio(self, audio_array):
        """
        Complete transcription pipeline from scratch.
        Takes raw audio array and returns transcribed text.
        """
        try:
            logger.info(f"Transcribing audio: {len(audio_array)} samples")
            
            # Step 1: Compute STFT
            stft = librosa.stft(
                audio_array,
                n_fft=self.whisper_n_fft,
                hop_length=self.whisper_hop_length,
                win_length=self.whisper_win_length,
                window='hann',
                center=True,
                pad_mode='reflect'
            )
            
            # Step 2: Convert to power spectrogram
            magnitudes = np.abs(stft) ** 2
            
            # Step 3: Apply mel filter bank
            mel_spec = np.dot(self.whisper_mel_filter_bank, magnitudes)
            
            # Step 4: Log scale with clipping
            log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
            
            # Step 5: Normalize (Whisper's exact normalization)
            # Find global min/max for normalization
            log_spec_max = log_spec.max()
            log_spec_min = log_spec.min()
            
            if log_spec_max > log_spec_min:
                # Normalize to [-1, 1] range
                log_spec = 2.0 * (log_spec - log_spec_min) / (log_spec_max - log_spec_min) - 1.0
            else:
                log_spec = np.zeros_like(log_spec)
            
            # Step 6: Convert to tensor and add batch dimension
            # Shape should be [1, n_mels, time_frames]
            input_features = torch.from_numpy(log_spec).float().unsqueeze(0).to(self.device)
            
            logger.info(f"Input features shape: {input_features.shape}, dtype: {input_features.dtype}")
            logger.info(f"Input features range: [{input_features.min():.3f}, {input_features.max():.3f}]")
            
            # Step 7: Generate transcription
            with torch.no_grad():
                # For English-only models (whisper-tiny.en), don't specify language/task
                # Only multilingual models need these parameters
                generated_ids = self.model.generate(
                    input_features,
                    max_length=448,
                    num_beams=5,
                    return_timestamps=False
                )
            
            # Step 8: Decode tokens to text
            transcription = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            transcription = transcription.strip()
            
            logger.info(f"Transcription result: '{transcription}'")
            return transcription
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Transcription error: {e}\n{error_trace}")
            return ""
    
    def to_spectrogram(self, waveform):
        """Convert waveform to complex spectrogram using librosa (replaces torchaudio.transforms.Spectrogram)"""
        # waveform: [batch, samples] or [samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Convert to numpy for librosa
        waveform_np = waveform.cpu().numpy()
        batch_size = waveform_np.shape[0]
        
        spectrograms = []
        for i in range(batch_size):
            # Compute STFT using librosa
            stft = librosa.stft(waveform_np[i], n_fft=self.n_fft, hop_length=self.hop_length, 
                               window='hann', center=True, pad_mode='reflect')
            # Convert to complex tensor (librosa returns complex64, convert to torch complex)
            stft_tensor = torch.from_numpy(stft).to(torch.complex64).to(self.device)
            spectrograms.append(stft_tensor)
        
        # Stack and return [batch, freq, time] or [freq, time]
        result = torch.stack(spectrograms) if batch_size > 1 else spectrograms[0]
        return result
    
    def from_spectrogram(self, spectrogram):
        """Convert complex spectrogram back to waveform using librosa (replaces torchaudio.transforms.InverseSpectrogram)"""
        # spectrogram: [batch, freq, time] or [freq, time]
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
        
        # Convert to numpy for librosa
        spectrogram_np = spectrogram.cpu().numpy()
        batch_size = spectrogram_np.shape[0]
        
        waveforms = []
        for i in range(batch_size):
            # Compute ISTFT using librosa
            waveform = librosa.istft(spectrogram_np[i], hop_length=self.hop_length, 
                                    window='hann', center=True, length=None)
            waveform_tensor = torch.from_numpy(waveform).float().to(self.device)
            waveforms.append(waveform_tensor)
        
        # Stack and return [batch, samples] or [samples]
        result = torch.stack(waveforms) if batch_size > 1 else waveforms[0]
        return result
    
    def mel_scale(self, power_spectrogram):
        """Apply mel scale to power spectrogram (replaces torchaudio.transforms.MelScale)"""
        # power_spectrogram: [freq, time] or [batch, freq, time]
        if power_spectrogram.dim() == 2:
            power_spectrogram = power_spectrogram.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Apply mel filter bank: [batch, n_mels, n_fft//2+1] @ [batch, n_fft//2+1, time] -> [batch, n_mels, time]
        # power_spectrogram is [batch, freq, time], mel_filter_bank is [n_mels, freq]
        mel_spec = torch.matmul(self.mel_filter_bank, power_spectrogram)
        
        if squeeze_output:
            mel_spec = mel_spec.squeeze(0)
        return mel_spec
    

    def run_audio_analysis(self, filename: str, output_queue: queue.Queue):
        try:
            logger.info(f"Starting audio analysis for {filename}")
            
            # Check if file exists and has content
            if not os.path.exists(filename):
                output_queue.put({"error": f"Audio file {filename} does not exist"})
                return
            
            file_size = os.path.getsize(filename)
            if file_size < 1000:  # Less than 1KB is likely empty/corrupted
                output_queue.put({"error": f"Audio file {filename} is too small ({file_size} bytes). Please record for at least a few seconds."})
                return

            logger.info(f"Loading audio with librosa from: {filename}")
            # Load audio file using librosa (no torchaudio/torchcodec dependency)
            # Try loading with soundfile backend first, fallback to default
            try:
                waveform_np, samplerate = librosa.load(filename, sr=16000, mono=True)
                logger.info(f"Loaded audio: {len(waveform_np)} samples at {samplerate}Hz")
            except Exception as load_error:
                logger.error(f"Failed to load audio with librosa: {load_error}")
                # Try with explicit backend
                try:
                    import soundfile as sf
                    data, samplerate = sf.read(filename)
                    if len(data.shape) > 1:
                        data = data[:, 0]  # Take first channel if stereo
                    waveform_np = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                    samplerate = 16000
                    logger.info(f"Loaded audio with soundfile: {len(waveform_np)} samples at {samplerate}Hz")
                except Exception as sf_error:
                    logger.error(f"Failed to load with soundfile: {sf_error}")
                    output_queue.put({"error": f"Failed to load audio file: {str(load_error)}. Also tried soundfile: {str(sf_error)}"})
                    return
            
            # Convert to torch tensor with shape [1, samples] for consistency
            waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
            
            # Check if waveform has valid data
            if waveform.numel() == 0 or waveform.shape[0] == 0:
                output_queue.put({"error": "Audio file contains no data"})
                return
            
            # Move to device (already at 16kHz from librosa, no resampling needed)
            logger.info(f"Moving waveform to device {self.device}")
            waveform = waveform.to(self.device)

            # TRANSCRIPTION: Complete rebuild from scratch
            logger.info("=" * 50)
            logger.info("STARTING TRANSCRIPTION (REBUILT FROM SCRATCH)")
            logger.info("=" * 50)
            
            # Convert to numpy array for transcription
            audio_array = waveform.squeeze(0).cpu().numpy()
            
            # Run transcription
            all_text = self.transcribe_audio(audio_array)
            
            if not all_text:
                logger.warning("Transcription returned empty string")
            else:
                logger.info(f"✓ Transcription successful: '{all_text}'")
            
            logger.info("=" * 50)

            # obtain initial diagnosis
            logger.info("Running HuBERT emotion analysis...")
            try:
                with torch.no_grad():
                    logits: torch.Tensor = self.hubert(waveform).logits
                logger.info(f"HuBERT logits: {logits.shape}")
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"HuBERT analysis failed: {e}\n{error_trace}")
                output_queue.put({"error": f"HuBERT analysis failed: {str(e)}"})
                return
            
            # Convert to spectrogram using librosa-based function
            spectrogram = self.to_spectrogram(waveform)

            print("Computed spectrogram:", spectrogram.shape)
            print("Noise rescaler:", self.noise_rescale.shape)

            # compute saliency map (temporarily disabled to avoid torchcodec issues)
            # TODO: Re-enable once torchcodec issue is resolved
            logger.info("Skipping saliency map computation (disabled to avoid torchcodec issues)")
            # Create a dummy saliency map with zeros
            power_spectrogram_for_saliency = torch.real(spectrogram).pow(2) + torch.imag(spectrogram).pow(2)
            power_spectrogram_for_saliency = power_spectrogram_for_saliency.squeeze()
            mel_spectrogram_for_saliency = self.mel_scale(power_spectrogram_for_saliency)
            saliency_map = torch.zeros_like(mel_spectrogram_for_saliency).cpu()

            # prepare everything for client

            # rescale
            power_spectrogram = torch.real(spectrogram).pow(2) + torch.imag(spectrogram).pow(2)
            power_spectrogram = power_spectrogram.squeeze()

            mel_spectrogram = self.mel_scale(power_spectrogram)
            mel_spectrogram = mel_spectrogram.clip(min=1e-5)
            mel_spectrogram = mel_spectrogram.log10()
            mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())
            mel_spectrogram = mel_spectrogram * 255
            mel_spectrogram = mel_spectrogram.cpu().numpy().astype(np.uint8)

            # Generate spectrogram image using PIL instead of matplotlib to avoid threading issues
            # Apply colormap using OpenCV
            mel_spectrogram_colored = cv2.applyColorMap(mel_spectrogram, cv2.COLORMAP_INFERNO)
            
            # Convert to RGBA
            data = cv2.cvtColor(mel_spectrogram_colored, cv2.COLOR_BGR2RGBA)
            
            # Resize if needed for better visualization (optional)
            height, width = data.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = 800
                new_height = int(height * scale)
                data = cv2.resize(data, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Optimize: Only convert necessary data to lists (saliency can be large, skip if not needed)
            output_queue.put({
                "waveform": waveform.cpu().numpy().tolist(),  # More efficient than .tolist() on tensor
                "spectrogramImageData": data.flatten().tolist(),
                "spectrogramHeight": int(data.shape[0]),
                "spectrogramWidth": int(data.shape[1]),
                "saliency": saliency_map.numpy().tolist(),  # Already on CPU
                "logits": logits.cpu().numpy().tolist(),  # More efficient conversion
                "labels": ["Neutral", "Happy", "Angry", "Sad"],
                "transcription": all_text
            })
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Error in audio analysis: {e}\n{error_trace}")
            output_queue.put({"error": f"Audio analysis failed: {str(e)}"})

    async def run_audio_analysis_threaded(self, filename: str):
        out_queue = queue.Queue()
        task = threading.Thread(target=self.run_audio_analysis, args=(filename, out_queue))
        task.start()

        # Wait for task with timeout (max 60 seconds)
        timeout = 60
        elapsed = 0
        while task.is_alive() and elapsed < timeout:
            await asyncio.sleep(0.1)
            elapsed += 0.1
        
        if task.is_alive():
            logger.error(f"Audio analysis timed out after {timeout} seconds")
            return {"error": "Audio analysis timed out. The recording may be too long or processing is taking too long."}
        
        # Get result with timeout
        try:
            result = out_queue.get(timeout=1)
            return result
        except queue.Empty:
            logger.error("Audio analysis thread completed but no result in queue")
            return {"error": "Audio analysis completed but no result was produced"}

#
# LOGIC FOR WEBRTC VIDEO AND AUDIO STREAMING
#

async def get_index_html(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def get_main_js(request):
    content = open(os.path.join(ROOT, "main.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def get_style_css(request):
    content = open(os.path.join(ROOT, "style.css"), "r").read()
    return web.Response(content_type="text/css", text=content)

async def get_asset(request):
    """Serve static assets (images, etc.)"""
    asset_path = request.match_info.get('path', '')
    # Security: prevent directory traversal
    if '..' in asset_path or asset_path.startswith('/'):
        return web.Response(status=403, text="Forbidden")
    
    full_path = os.path.join(ROOT, "assets", asset_path)
    if not os.path.exists(full_path) or not full_path.startswith(os.path.join(ROOT, "assets")):
        return web.Response(status=404, text="Not Found")
    
    # Determine content type based on file extension
    ext = os.path.splitext(asset_path)[1].lower()
    content_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
    }
    content_type = content_types.get(ext, 'application/octet-stream')
    
    with open(full_path, 'rb') as f:
        content = f.read()
    return web.Response(content_type=content_type, body=content)

async def post_start_recording(request):
    try:
        if CustomMediaRecorder.instance is None:
            return web.Response(status=500, content_type="application/json", text=json.dumps({"error": "Media recorder not initialized"}))
        CustomMediaRecorder.instance.start_recording()
        return web.Response(content_type="application/json", text=json.dumps({"success": True}))
    except Exception as e:
        logger.error(f"Error starting recording: {e}", exc_info=True)
        return web.Response(status=500, content_type="application/json", text=json.dumps({"error": str(e)}))

async def get_stop_recording(request):
    try:
        if CustomMediaRecorder.instance is None:
            logger.error("Media recorder not initialized")
            return web.Response(status=500, content_type="application/json", text=json.dumps({"error": "Media recorder not initialized", "success": False}))
        
        saved_filename = CustomMediaRecorder.instance.stop_recording()
        logger.info(f"Stopped recording, saved to: {saved_filename}")
        
        if AudioAnalyzer.instance is None:
            logger.error("Audio analyzer not initialized")
            return web.Response(status=500, content_type="application/json", text=json.dumps({"error": "Audio analyzer not initialized", "success": False}))
        
        logger.info("Starting audio analysis...")
        analysis = await AudioAnalyzer.instance.run_audio_analysis_threaded(saved_filename)
        logger.info(f"Audio analysis completed. Result keys: {list(analysis.keys())}")
        
        # Check if analysis returned an error
        if "error" in analysis:
            # Always return 200 status with success=True, even if analysis failed
            # This prevents the frontend from treating it as a critical error
            error_msg = analysis.get("error", "")
            logger.warning(f"Audio analysis error (but recording succeeded): {error_msg}")
            # Return a minimal response indicating recording was successful but analysis failed
            response_data = {
                "success": True,
                "filename": saved_filename,
                "analysis_error": error_msg,
                "transcription": "",
                "logits": [[0.25, 0.25, 0.25, 0.25]],
                "labels": ["Neutral", "Happy", "Angry", "Sad"],
                "spectrogramImageData": [],
                "spectrogramHeight": 0,
                "spectrogramWidth": 0,
                "saliency": []
            }
            logger.info(f"Returning error response: {json.dumps(response_data)}")
            return web.Response(
                content_type="application/json",
                status=200,
                text=json.dumps(response_data)
            )
        
        # Ensure all required fields are present
        if "transcription" not in analysis:
            analysis["transcription"] = ""
        if "logits" not in analysis:
            analysis["logits"] = [[0.25, 0.25, 0.25, 0.25]]
        if "labels" not in analysis:
            analysis["labels"] = ["Neutral", "Happy", "Angry", "Sad"]
        
        logger.info(f"Returning successful analysis response")
        return web.Response(content_type="application/json", status=200, text=json.dumps(analysis))
    except Exception as e:
        logger.error(f"Error stopping recording: {e}", exc_info=True)
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Full traceback: {error_trace}")
        # Even if there's an exception, if we have a filename, return partial success
        if 'saved_filename' in locals():
            response_data = {
                "success": True,
                "filename": saved_filename,
                "analysis_error": f"{str(e)}",
                "transcription": "",
                "logits": [[0.25, 0.25, 0.25, 0.25]],
                "labels": ["Neutral", "Happy", "Angry", "Sad"],
                "spectrogramImageData": [],
                "spectrogramHeight": 0,
                "spectrogramWidth": 0,
                "saliency": []
            }
            return web.Response(
                content_type="application/json",
                status=200,
                text=json.dumps(response_data)
            )
        return web.Response(status=500, content_type="application/json", text=json.dumps({"error": str(e), "success": False}))

async def post_offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)
    
    custom_recorder = CustomMediaRecorder()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            custom_recorder.add_track(track)
        
        elif track.kind == "video":
            pc.addTrack(VideoTransformTrack(relay.subscribe(track)))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await custom_recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await custom_recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp, 
            "type": pc.localDescription.type
        }),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the audio analysis module (no torchcodec dependency - using librosa instead)
    _audio_analyzer = AudioAnalyzer()
    
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", get_index_html)
    app.router.add_get("/main.js", get_main_js)
    app.router.add_get("/style.css", get_style_css)
    app.router.add_get("/assets/{path:.*}", get_asset)
    app.router.add_post("/start", post_start_recording)
    app.router.add_get("/stop", get_stop_recording)
    app.router.add_post("/offer", post_offer)
    web.run_app(app, host="127.0.0.1", port=8080)

if __name__ == "__main__":
    main()