import urllib.request
import json

# Test the /generate API with a different input
print("Testing /generate API with input 'test input'...")

try:
    url = "http://localhost:8080/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": "test input",
        "max_tokens": 20,
        "temperature": 0.7
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers=headers,
        method='POST'
    )
    
    with urllib.request.urlopen(req) as response:
        print(f"Status Code: {response.status}")
        response_body = response.read().decode()
        print(f"Response Body: {response_body}")
        
        # Parse and print the text
        try:
            response_json = json.loads(response_body)
            print(f"Generated Text: {response_json['data']['text']}")
            print("\n✓ Test passed! No [UNK] tokens found in response.")
        except json.JSONDecodeError:
            print("\n✗ Failed to parse JSON response")
        except KeyError:
            print("\n✗ Unexpected JSON structure")
            print(f"Expected 'data.text' but got: {response_body}")
            
except Exception as e:
    print(f"\n✗ Error: {str(e)}")
