# DigitalOcean Spaces Configuration Guide

## Where to Find Your Spaces Credentials

### 1. **Spaces Name** (SPACES_NAME)
- Go to: https://cloud.digitalocean.com/spaces
- Your Spaces name is displayed in the list of Spaces
- It's the name you gave your Space when you created it
- Example: If your Space is called `my-towerguard-space`, that's your SPACES_NAME

### 2. **Spaces Region** (SPACES_REGION)
- In the same Spaces list, look at the region column
- Common regions:
  - `nyc3` - New York
  - `sfo3` - San Francisco
  - `ams3` - Amsterdam
  - `sgp1` - Singapore
  - `fra1` - Frankfurt
- You can also see it in your Space's URL: `https://your-space-name.nyc3.digitaloceanspaces.com`
- The region is the part between the space name and `.digitaloceanspaces.com`

### 3. **Access Key** (SPACES_KEY)
- Go to: https://cloud.digitalocean.com/account/api/spaces
- Click "Generate New Key"
- Give it a name (e.g., "TowerGuard App")
- Copy the **Access Key** - this is your `SPACES_KEY`

### 4. **Secret Key** (SPACES_SECRET)
- On the same page where you generated the key
- Copy the **Secret Key** - this is your `SPACES_SECRET`
- ⚠️ **Important**: You can only see the secret key once when you first create it!
- If you lose it, you'll need to generate a new key pair

## Setting Environment Variables

### Option 1: Export in Terminal (Temporary)
```bash
export SPACES_KEY="your-access-key-here"
export SPACES_SECRET="your-secret-key-here"
export SPACES_REGION="nyc3"  # Replace with your region
export SPACES_NAME="your-space-name"  # Replace with your space name
```

### Option 2: Add to start.sh (Persistent)
Edit your `start.sh` file and add the exports before starting the server:
```bash
export SPACES_KEY="your-access-key-here"
export SPACES_SECRET="your-secret-key-here"
export SPACES_REGION="nyc3"
export SPACES_NAME="your-space-name"
```

### Option 3: Use .env file (Recommended for Production)
Create a `.env` file in your project root:
```bash
SPACES_KEY=your-access-key-here
SPACES_SECRET=your-secret-key-here
SPACES_REGION=nyc3
SPACES_NAME=your-space-name
```

Then load it in your script (you may need to install python-dotenv):
```bash
pip install python-dotenv
```

## Quick Verification

After setting the variables, you can test the connection by making a recording. Check your server logs for:
- ✅ "Successfully uploaded spectrogram to Spaces: https://..."
- ❌ "SPACES_NAME not configured" or "SPACES_KEY or SPACES_SECRET not set"

## Example Configuration

If your Space is:
- **Name**: `towerguard-audio`
- **Region**: `nyc3`
- **URL**: `https://towerguard-audio.nyc3.digitaloceanspaces.com`

Then your environment variables should be:
```bash
export SPACES_NAME="towerguard-audio"
export SPACES_REGION="nyc3"
export SPACES_KEY="DO00ABCDEF1234567890"
export SPACES_SECRET="abcdef1234567890abcdef1234567890abcdef1234567890"
```

