import requests
import argparse


def main(feature_list):
    url = "http://localhost:8000/predict"

    payload = {"features": feature_list}

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()["prediction"]

    raise Exception("Invalid response", response.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("pass features list")
    parser.add_argument(
        "--feat_list",
        type=float,
        nargs="+",
        required=True,
    )
    args = parser.parse_args()
    print(main(args.feat_list))
