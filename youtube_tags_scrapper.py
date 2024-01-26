from bs4 import BeautifulSoup
import requests


response = requests.get('https://www.youtube.com/watch?v=co_MeKSnyAo&t=24s&ab_channel=LexFridman')
content = response.text
'''
with open('video.txt','w',encoding='utf-8')as f:
    f.write(content)
    print("Done")
    
#print(content)
'''
with open('video.txt', 'r', encoding='utf-8') as f:
    content = f.read()

soup = BeautifulSoup(content, 'lxml')

tags_containers = soup.find_all("div",class_="yt-core-attributed-string--link-inherit-color")

print(tags_containers)