"""
Quick test script for FastAPI endpoints
Usage: python test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:5000/api"

def test_endpoint(name, method, url, data=None):
    """Test an API endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            print(f"Unknown method: {method}")
            return
        
        print(f"Status Code: {response.status_code}")
        print(f"Response:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print(f"✅ {name} - SUCCESS")
        else:
            print(f"❌ {name} - FAILED")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server. Is it running?")
        print(f"   Start server with: python start_server.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("FastAPI Backend Test Suite")
    print("=" * 60)
    print("Make sure the server is running: python start_server.py")
    print("=" * 60)
    
    # Test endpoints
    test_endpoint("Health Check", "GET", f"{BASE_URL}/health")
    test_endpoint("Model Info", "GET", f"{BASE_URL}/model/info")
    test_endpoint("File Status", "GET", f"{BASE_URL}/files/status")
    test_endpoint("Summary", "GET", f"{BASE_URL}/data/summary")
    test_endpoint("Historical Data", "GET", f"{BASE_URL}/data/historical")
    test_endpoint("Predictions", "GET", f"{BASE_URL}/predictions")
    test_endpoint("Make Prediction", "POST", f"{BASE_URL}/predict", {"forecastMonths": 1})
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print("\nFor interactive testing, visit: http://localhost:5000/docs")
