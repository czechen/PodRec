import requests
import re

# Define the endpoint and parameters
endpoint = "https://en.wikipedia.org/w/api.php"
params = {
    "action": "query",
    "format": "json",
    "titles": "Jared_Kushner",
    "prop": "revisions",
    "rvprop": "content"
}

# Send the request
response = requests.get(endpoint, params=params)
data = response.json()

# Extract the page content
page_content = list(data['query']['pages'].values())[0]['revisions'][0]['*']

# Find the birth date using a regex
birth_date_match = re.search(r'\b(?:Born|birth_date)\s*=\s*([^\n]*)', page_content)
if birth_date_match:
    birth_date = birth_date_match.group(1)
    print(f'Birth date: {birth_date}')
else:
    print('Birth date not found')