import os
import time
import requests
import pandas as pd
from tqdm import tqdm

MAPBOX_TOKEN = "pk.eyJ1IjoicHJhYmhhdDA4IiwiYSI6ImNtamVwOXh2djBnenIzZnM4ZHA1eGI3YWkifQ.vzo4cGwR7qpds-V8HcVPSQ"

IMAGE_SIZE = 224
ZOOM_LEVELS = [16, 18]

TRAIN_FILE = "data/raw/train.xlsx"
TEST_FILE  = "data/raw/test.xlsx"

IMAGE_ROOT = "data/images"

REQUEST_TIMEOUT = 15
SLEEP_SECONDS = 0.25  

def build_url(lat, lon, zoom):
    return (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"{lon},{lat},{zoom}/{IMAGE_SIZE}x{IMAGE_SIZE}"
        f"?access_token={MAPBOX_TOKEN}"
    )

def download_one(lat, lon, zoom, save_path):
    if os.path.exists(save_path):
        return "exists"

    url = build_url(lat, lon, zoom)

    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        return "ok"
    except Exception:
        return "fail"


def download_split(df, split_name):
    for zoom in ZOOM_LEVELS:
        out_dir = os.path.join(IMAGE_ROOT, f"zoom{zoom}", split_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nDownloading {split_name} images | zoom{zoom}")
        stats = {"ok": 0, "exists": 0, "fail": 0}

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"{split_name} | zoom{zoom}",
            unit="img"
        ):
            pid = row["id"]
            lat = row["lat"]
            lon = row["long"]

            save_path = os.path.join(out_dir, f"{pid}.png")
            status = download_one(lat, lon, zoom, save_path)
            stats[status] += 1

            time.sleep(SLEEP_SECONDS)

        print(f"zoom{zoom} summary:", stats)

if __name__ == "__main__":
    train_df = pd.read_excel(TRAIN_FILE)
    test_df  = pd.read_excel(TEST_FILE)

    download_split(train_df, "train")
    download_split(test_df, "test")

    print("\nAll downloads completed.")
