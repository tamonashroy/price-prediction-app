from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
import requests
import re
import urllib.parse

app = FastAPI()

# Your Facebook App credentials
APP_ID = "1302012748155479"
APP_SECRET = "51f63774b1096e1856a4d4eea458847b"

# Function to get App Token
def get_access_token():
    token_url = "https://graph.facebook.com/oauth/access_token"
    params = {
        "client_id": APP_ID,
        "client_secret": APP_SECRET,
        "grant_type": "client_credentials"
    }

    r = requests.get(token_url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]

# Extract video ID from Facebook URL
def extract_video_id(fb_url: str) -> str:
    fb_url = urllib.parse.unquote(fb_url)
    # Common patterns in Facebook URLs
    patterns = [
        r"/videos/(\d+)",                        # /videos/1234567890
        r"v=(\d+)",                              # ?v=1234567890
        r"/(\d+)/?$"                             # trailing number
    ]
    for pattern in patterns:
        match = re.search(pattern, fb_url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract video ID from URL.")

# Proxy endpoint accepting full video URL
@app.get("/proxy")
def proxy_video(url: str = Query(..., description="Facebook video URL")):
    try:
        # Extract video ID
        video_id = extract_video_id(url)
        access_token = get_access_token()

        # Get video source from Graph API
        graph_url = f"https://graph.facebook.com/v19.0/{video_id}"
        params = {
            "fields": "source",
            "access_token": access_token
        }
        r = requests.get(graph_url, params=params)
        r.raise_for_status()
        source_url = r.json().get("source")

        if not source_url:
            raise HTTPException(status_code=404, detail="Video source not available")

        # Stream the video content
        stream = requests.get(source_url, stream=True, timeout=10)
        if stream.status_code != 200:
            raise HTTPException(status_code=stream.status_code, detail="Unable to stream video")

        content_type = stream.headers.get("Content-Type", "video/mp4")
        return StreamingResponse(stream.raw, media_type=content_type)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
