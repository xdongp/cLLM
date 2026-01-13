#!/usr/bin/env python3
import urllib.request
import urllib.parse
import json

# Test the /generate interface
print("Testing /generate API with input 'hello'...")

url = "http://localhost:8080/generate"
headers = {
    "Content-Type": "application/json"
}

# Prepare the data
request_data = {
    "prompt": "hello",
    "max_tokens": 50,
    "temperature": 0.7
}

# Convert to JSON
json_data = json.dumps(request_data).encode('utf-8')

try:
    # Create request
    req = urllib.request.Request(url, data=json_data, headers=headers, method='POST')
    
    # Send request
    with urllib.request.urlopen(req) as response:
        # Read response
        status_code = response.status
        response_body = response.read().decode('utf-8')
        
        print(f"Status Code: {status_code}")
        print(f"Response Body: {response_body}")
        
        # Try to parse JSON
        try:
            json_response = json.loads(response_body)
            print("\nJSON parsed successfully")
            if "choices" in json_response and json_response["choices"]:
                generated_text = json_response["choices"][0]["text"]
                print(f"Generated Text: {generated_text}")
        except json.JSONDecodeError:
            print("\nResponse is not valid JSON")
            
except urllib.error.URLError as e:
    print(f"Error: Could not connect to server - {e.reason}")
