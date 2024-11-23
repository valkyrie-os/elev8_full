import requests
import json

# API endpoint
url = "http://localhost:8001/analyze/mert.incesu"

# Make the POST request
response = requests.get(url)

# Check if request was successful
if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, indent=2))
else:
    print(f"Error: {response.status_code}")
    print(response.text)