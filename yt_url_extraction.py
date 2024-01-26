import requests
import json
import re
def yt_url_extraction(link:str):
    video_id = link[link.find('v=')+2:]
    API_KEY = 'AIzaSyAZr4GF3wouWtBamx5xw0Fom9_ZJQiq9CE'  # Replace with your API key
    VIDEO_ID = video_id     # Replace with the desired YouTube video ID
    # The URL for the YouTube Data API request
    url = f"https://www.googleapis.com/youtube/v3/videos?id={VIDEO_ID}&key={API_KEY}&part=snippet"

    response = requests.get(url)
    data = response.json()
    #print(data)
    #with open('yt_data.json','w') as f:
    #    json.dump(data,f,indent=4)
    # Check if items exist in the returned data
    if data.get('items'):
        video_data = data['items'][0]['snippet']

        title = video_data['title']
        description = video_data['description']
        youtube_channel = video_data["channelTitle"]
        tags = video_data["tags"]
        
        
        #print(f"Title: {title}")
        #print(f"Description: {description}")
        #print(f"Youtube channel: {youtube_channel}")
        #print(f"Tags: {tags}")
    else:
        #print("No data found for this video ID.")
        pass
    timestamps_all = {}
    timestamps = re.findall(r'(?:\d:)\d+.*',description)
    #print(timestamps)
    # extracting hour, minutes, seconds of time stamp and seperating timestamp info
    for timestamp in timestamps:
        time = re.match(r'\d+:(?!-)[^-]*:?\d+ ',timestamp)
        time = time.group()
        timestamp_info = timestamp[str(timestamp).find(time)+len(str(time)):]
        hms = time.split(':')
        if len(hms) == 3:
            hours,minutes,seconds = hms
        elif len(hms) == 2:
            minutes,seconds = hms
            hours = '0'
        elif len(hms) == 1:
            seconds = hms
            hours,minutes = '0','0'
        else:
            "Time tag was not found!"
        #print(f"Hours: {hours}, minutes: {minutes}, seconds: {seconds}")
        timestamps_all[time] = timestamp_info    
    print(timestamps_all)
    return {'youtube_channel':youtube_channel,'title':title,'tags':tags,'timestamps':timestamps_all,'link':link}

if __name__ == "__main__":
    print(yt_url_extraction('https://www.youtube.com/watch?v=co_MeKSnyAo&t=24s&ab_channel=LexFridman'))