#!/usr/bin/env python3
import argparse
import json
import requests


def main():
    parser = argparse.ArgumentParser(description="Send a single transaction to the Fraud Detection API")
    parser.add_argument("--url", default="http://127.0.0.1:5000/predict", help="API /predict URL")
    parser.add_argument("--payload", help="Path to JSON file with transaction features (30 fields)")
    args = parser.parse_args()

    with open(args.payload, "r") as f:
        data = json.load(f)

    r = requests.post(args.url, json=data, timeout=10)
    r.raise_for_status()
    print(r.json())


if __name__ == "__main__":
    main()
