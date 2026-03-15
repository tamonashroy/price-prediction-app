# streamlit_facebook_oembed.py

import streamlit as st
import requests

# Enter your app credentials
APP_ID = "1239430984039292"
APP_SECRET = "7f74e8dfde86d414a5db2a4ae8b2164e"
ACCESS_TOKEN = f"{APP_ID}|{APP_SECRET}"

st.title("📺 Facebook Video Embed via oEmbed")

video_url = st.text_input("Enter Facebook Video URL:")

if video_url:
    try:
        oembed_url = "https://graph.facebook.com/v23.0/oembed_video"
        params = {
            "url": video_url,
            "access_token": ACCESS_TOKEN
        }

        response = requests.get(oembed_url, params=params)
        data = response.json()

        if "html" in data:
            st.markdown("### Video Preview")
            st.components.v1.html(data["html"], height=data.get("height", 315), scrolling=False)
        else:
            st.error(f"Error: {data.get('error', {}).get('message', 'Unknown error')}")
    except Exception as e:
        st.error(f"Failed to fetch video embed: {str(e)}")
