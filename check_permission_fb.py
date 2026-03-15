import requests

# Replace with your valid user access token with user_videos permission
ACCESS_TOKEN = "EAAXafweZC2y4BPA7KAB2G46jd51ZBfeCsgDxem8Fi3j1i3ZBQASOsKKcdnipb9ejTfdkIqvKJjCHos92v1fIZCE9ZBlrNj3czQ1WNhTio4iOyYJTFG9qrruPYyS8HX12rtviDLq6XZCMEBT97OrJ86tA7zJbJ5h1IuDqtFZCNAcSMLZAcnOSJQoaInHeiJuDydIFENYba5wNivODkFCWUl2PKv9IV26VuYaIoAZDZD"

def get_user_videos():
    url = f"https://graph.facebook.com/v19.0/me/videos"
    params = {
        "access_token": ACCESS_TOKEN,
        "fields": "id,description,created_time,length,permalink_url,thumbnails"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        videos = data.get("data", [])
        if not videos:
            print("✅ No videos found for this user.")
        else:
            for video in videos:
                print("🎞️ Video ID:", video.get("id"))
                print("📝 Description:", video.get("description"))
                print("📅 Created:", video.get("created_time"))
                print("🔗 URL:", video.get("permalink_url"))
                print("------")
    else:
        print("❌ Error:", response.status_code)
        print(response.json())

if __name__ == "__main__":
    get_user_videos()
