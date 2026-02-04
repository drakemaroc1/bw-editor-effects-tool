# BW Editor Effects Tool

Generate AI-powered video clips for real estate listings.

## Features

- **Frame Generation**: Nano Banana Pro (fal.ai) for styled frames
- **Video Generation**: VEO 3.1 Fast (Google Vertex AI) for animation
- **10 Effect Types**: Day→Night, Virtual Staging, Furniture Float, etc.
- **5 Camera Movements**: Orbit, Dolly, Crane, Push, Pull

## Setup

### Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud Deployment

Add these secrets in Streamlit Cloud dashboard:

```toml
FAL_KEY = "your-fal-api-key"

[GOOGLE_SERVICE_ACCOUNT]
type = "service_account"
project_id = "your-project"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
```

## Usage

1. Upload listing images
2. Assign effects to each image
3. Click Generate
4. Download clips for editing
