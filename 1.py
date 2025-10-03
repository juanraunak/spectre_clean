#!/usr/bin/env python3
import requests
import json

# Hardcoded API key and CSE ID
GOOGLE_API_KEY_HARDCODE = "AIzaSyAohBAGNUxv_QpPXoMjvAXRipIqdhb1DY4"
GOOGLE_CSE_ID_HARDCODE  = "9539617f2a9e14131"

def google_custom_search(query, api_key=GOOGLE_API_KEY_HARDCODE, cse_id=GOOGLE_CSE_ID_HARDCODE):
    """Perform a Google Custom Search query"""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": 5  # Number of results (max 10 per request)
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json()
        return results.get("items", [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

if __name__ == "__main__":
    # Test query
    query = "Manipal Fintech"
    results = google_custom_search(query)

    print(f"\nTop results for query: {query}\n")
    for i, item in enumerate(results, start=1):
        title = item.get("title")
        link = item.get("link")
        snippet = item.get("snippet")
        print(f"{i}. {title}\n   {link}\n   {snippet}\n")
