import requests
from bs4 import BeautifulSoup

URL = "http://quotes.toscrape.com/"

# Send a GET request to the website
response = requests.get(URL)
soup = BeautifulSoup(response.content, 'lxml')

# Find all quote containers
quote_containers = soup.find_all("div", class_="quote")

for container in quote_containers:
    # Extract quote
    quote = container.find("span", class_="text").text
    # Extract author
    author = container.find("small", class_="author").text
    
    print(f"Quote: {quote}")
    print(f"Author: {author}")
    print("-----")