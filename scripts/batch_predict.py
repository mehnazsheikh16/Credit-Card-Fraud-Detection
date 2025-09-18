#!/usr/bin/env python3
import argparse
import requests


def main():
    parser = argparse.ArgumentParser(description="Send a CSV of transactions to the Fraud Detection API for batch scoring")
    parser.add_argument("--url", default="http://127.0.0.1:5000/batch_predict", help="API /batch_predict URL")
    parser.add_argument("--csv", required=True, help="Path to CSV with required columns")
    parser.add_argument("--out", default="batch_predictions.csv", help="Where to save the returned predictions CSV")
    args = parser.parse_args()

    with open(args.csv, "rb") as f:
        files = {"file": (args.csv, f, "text/csv")}
        r = requests.post(args.url, files=files, timeout=30)
        r.raise_for_status()
        with open(args.out, "wb") as out:
            out.write(r.content)
    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
