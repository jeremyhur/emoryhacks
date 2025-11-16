# How to Create a DigitalOcean Spaces Bucket

## Step-by-Step Guide

### 1. Go to DigitalOcean Spaces
- Navigate to: https://cloud.digitalocean.com/spaces
- Or click "Spaces" in the left sidebar of your DigitalOcean dashboard

### 2. Create a New Space
- Click the **"Create Spaces Bucket"** button (usually green/blue button in the top right)

### 3. Configure Your Space

#### **Choose a Datacenter Region**
- Select a region closest to you or your users
- Common options:
  - **New York (nyc3)** - Good for US East Coast
  - **San Francisco (sfo3)** - Good for US West Coast
  - **Amsterdam (ams3)** - Good for Europe
  - **Singapore (sgp1)** - Good for Asia
  - **Frankfurt (fra1)** - Good for Central Europe
- **Note the region code** (e.g., `nyc3`, `sfo3`) - you'll need this for `SPACES_REGION`

#### **Choose a Unique Name**
- Enter a unique name for your Space (e.g., `towerguard-audio`, `towerguard-recordings`)
- The name must be:
  - All lowercase
  - No spaces (use hyphens or underscores)
  - Globally unique across all DigitalOcean Spaces
- **This is your `SPACES_NAME`**

#### **File Listing (Optional)**
- Choose whether to make file listings public or private
- For this use case, you can leave it as **"Public"** since we're setting individual file ACLs

#### **CDN (Optional)**
- You can enable CDN for faster file delivery
- For now, you can skip this (you can enable it later)

### 4. Create the Space
- Click **"Create a Spaces Bucket"** button
- Wait a few seconds for it to be created

### 5. Get Your Access Keys

After creating the Space, you need to create access keys:

1. Go to: https://cloud.digitalocean.com/account/api/spaces
2. Click **"Generate New Key"**
3. Give it a name (e.g., "TowerGuard App")
4. Click **"Generate Key"**
5. **IMPORTANT**: Copy both:
   - **Access Key** → This is your `SPACES_KEY`
   - **Secret Key** → This is your `SPACES_SECRET`
   - ⚠️ You can only see the secret key once! Save it securely.

### 6. Configure Your Environment Variables

Once you have all the information:

```bash
export SPACES_NAME="your-space-name"        # The name you chose in step 3
export SPACES_REGION="nyc3"                 # The region code from step 3
export SPACES_KEY="your-access-key"         # From step 5
export SPACES_SECRET="your-secret-key"      # From step 5
```

### 7. Test Your Configuration

After setting the variables, make a test recording in your app and check the server logs. You should see:
- ✅ "Successfully uploaded spectrogram to Spaces: https://..."

## Example

If you create a Space with:
- **Name**: `towerguard-recordings`
- **Region**: New York (`nyc3`)

Your Space URL will be: `https://towerguard-recordings.nyc3.digitaloceanspaces.com`

And your environment variables would be:
```bash
export SPACES_NAME="towerguard-recordings"
export SPACES_REGION="nyc3"
export SPACES_KEY="DO00ABCDEF1234567890"
export SPACES_SECRET="abcdef1234567890abcdef1234567890abcdef1234567890"
```

## Pricing Note

DigitalOcean Spaces pricing:
- **$5/month** for 250 GB storage
- **$0.02/GB** for additional storage
- **$0.01/GB** for outbound data transfer
- First 1 GB of outbound transfer is free per month

For audio files and spectrogram images, this should be very affordable unless you're storing massive amounts of data.

