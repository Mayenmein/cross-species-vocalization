from pathlib import Path
import requests
import freesound
from tqdm import tqdm


def init_client(api_key: str):
    client = freesound.FreesoundClient()
    client.set_token(api_key, "token")
    return client

     
def download_file(url: str, dest: Path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def fetch_sounds(client, query: str, out_dir: Path, limit=5):
    out_dir.mkdir(parents=True, exist_ok=True)

    results = client.search(
        query=query,
        filter="duration:[1 TO 30]",
        fields="id,name,previews",
        page_size=limit,
    )

    paths = []
    for s in results:
        url = s.previews.preview_hq_ogg  # correct format
        fname = f"{s.id}_{s.name[:40].replace(' ','_')}.ogg"
        fname = "".join(c for c in fname if c.isalnum() or c in "._-")
        path = out_dir / fname

        if not path.exists():
            download_file(url, path)

        paths.append(path)

    return paths


def main():
    API_KEY = "tj67wdsD1C0CWbtDUPREV4s84r0XwcwEXMCmytG9"  # or use env
    root = Path(__file__).resolve().parent.parent.parent
    base = root / "data" / "raw" / "freesound"

    client = init_client(API_KEY)

    queries = ["whale song", "bird song", "engine failure"]

    for q in queries:
        fetch_sounds(client, q, base / q.replace(" ", "_"), limit=5)


if __name__ == "__main__":
    main()