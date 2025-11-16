#!/usr/bin/env python3
"""
Test script to verify DigitalOcean Spaces upload is working
"""

import os
import sys
import boto3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_spaces_upload():
    """Test uploading a file to DigitalOcean Spaces"""
    
    # Get credentials from environment
    spaces_key = os.environ.get("SPACES_KEY")
    spaces_secret = os.environ.get("SPACES_SECRET")
    spaces_region = os.environ.get("SPACES_REGION", "nyc3")
    spaces_name = os.environ.get("SPACES_NAME", "your-space-name")
    
    logger.info("="*60)
    logger.info("Testing DigitalOcean Spaces Configuration")
    logger.info("="*60)
    
    # Check if credentials are set
    if not spaces_key or not spaces_secret:
        logger.error("❌ SPACES_KEY or SPACES_SECRET not set!")
        logger.error("   Set them with: export SPACES_KEY='...' and export SPACES_SECRET='...'")
        return False
    
    if spaces_name == "your-space-name":
        logger.error("❌ SPACES_NAME not configured (still using placeholder)")
        return False
    
    logger.info(f"✓ SPACES_KEY: {spaces_key[:10]}..." if len(spaces_key) > 10 else f"✓ SPACES_KEY: {spaces_key}")
    logger.info(f"✓ SPACES_SECRET: {'*' * 20}...")
    logger.info(f"✓ SPACES_REGION: {spaces_region}")
    logger.info(f"✓ SPACES_NAME: {spaces_name}")
    
    # Check if we have a spectrogram file to test with
    recordings_dir = os.path.join(os.path.dirname(__file__), "recordings")
    test_files = [f for f in os.listdir(recordings_dir) if f.endswith("_spectrogram.png")]
    
    if not test_files:
        logger.warning("No spectrogram files found to test with")
        return False
    
    test_file = os.path.join(recordings_dir, test_files[-1])  # Use the most recent one
    logger.info(f"\nTesting upload with: {test_file}")
    
    try:
        endpoint_url = f"https://{spaces_region}.digitaloceanspaces.com"
        logger.info(f"Endpoint: {endpoint_url}")
        
        session = boto3.session.Session()
        client = session.client('s3',
                                region_name=spaces_region,
                                endpoint_url=endpoint_url,
                                aws_access_key_id=spaces_key,
                                aws_secret_access_key=spaces_secret)
        
        # Test upload
        test_key = f"test_{os.path.basename(test_file)}"
        logger.info(f"Uploading to: {spaces_name}/{test_key}")
        
        client.upload_file(test_file,
                           spaces_name,
                           test_key,
                           ExtraArgs={'ACL': 'public-read'})
        
        url = f"https://{spaces_name}.{spaces_region}.digitaloceanspaces.com/{test_key}"
        logger.info(f"\n✅ SUCCESS! File uploaded successfully!")
        logger.info(f"   URL: {url}")
        
        # Try to verify the file is accessible
        try:
            import urllib.request
            response = urllib.request.urlopen(url, timeout=5)
            if response.status == 200:
                logger.info(f"✅ File is publicly accessible!")
            else:
                logger.warning(f"⚠️  File uploaded but returned status {response.status}")
        except Exception as e:
            logger.warning(f"⚠️  Could not verify public access: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ UPLOAD FAILED!")
        logger.error(f"   Error: {e}")
        logger.error(f"\nCommon issues:")
        logger.error(f"   1. Invalid SPACES_KEY or SPACES_SECRET")
        logger.error(f"   2. Wrong SPACES_REGION")
        logger.error(f"   3. Wrong SPACES_NAME")
        logger.error(f"   4. Space doesn't exist or you don't have access")
        return False

if __name__ == "__main__":
    success = test_spaces_upload()
    sys.exit(0 if success else 1)

