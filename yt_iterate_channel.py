from googleapiclient.discovery import build

api_key = 'AIzaSyAZr4GF3wouWtBamx5xw0Fom9_ZJQiq9CE'
youtube = build('youtube', 'v3', developerKey=api_key)

# Get the uploads playlist id
channel_response = youtube.channels().list(
    part='contentDetails',
    id='@lexfridman'
).execute()
uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

# Get videos from the uploads playlist
playlist_items_response = youtube.playlistItems().list(
    part='snippet',
    playlistId=uploads_playlist_id,
    maxResults=50  # max allowed per request
).execute()

for item in playlist_items_response['items']:
    video_title = item['snippet']['title']
    video_id = item['snippet']['resourceId']['videoId']
    print(f'{video_title}: https://www.youtube.com/watch?v={video_id}')
