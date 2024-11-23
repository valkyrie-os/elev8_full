import requests
import json

# API endpoint
url = "http://localhost:8002/analyze"

# Sample data
data = {
    "github_data": {
        "commits": [
            {
                "hash": "abc123",
                "author": "John Doe",
                "date": "2024-03-20",
                "message": "Update user authentication",
                "changes": {
                    "files_changed": ["auth.py", "users.py"],
                    "insertions": 50,
                    "deletions": 20
                },
                "analysis": {
                    "rating": "high",
                    "analysis": "Significant improvement to authentication system"
                }
            }
        ]
    },
    "slack_data": {
        "user_statistics": {
            "total_messages": 150,
            "unique_channels": 5,
            "date_range": {
                "start": "2024-01-01",
                "end": "2024-03-20"
            },
            "messages": [
                {
                    "channel": "team-dev",
                    "user": "john.doe",
                    "timestamp": "2024-03-20T10:00:00Z",
                    "text": "Deployed the new authentication system"
                }
            ]
        }
    }
}

# Make the POST request
response = requests.post(url, json=data)

# Check if request was successful
if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, indent=2))
else:
    print(f"Error: {response.status_code}")
    print(response.text)