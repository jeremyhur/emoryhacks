#!/bin/bash
# Startup script that sets FFmpeg library path for TorchCodec
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH

# Start the server
cd "$(dirname "$0")"
python3 server.py
