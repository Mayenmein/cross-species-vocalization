from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import hashlib


def sha256(file_path: Path, chunk_size=8192) -> str:
    """Compute SHA256 checksum (streaming, memory-safe)"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)

    existing_size = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}

    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()

        total_size = int(r.headers.get("Content-Length", 0))
        total = total_size + existing_size if total_size else None

        mode = "ab" if existing_size else "wb"

        with open(dest, mode) as f, tqdm(
            total=total,
            initial=existing_size,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def verify(file_path: Path, expected_hash: str | None):
    """Verify checksum if provided"""
    if not expected_hash:
        print(" No checksum provided, skipping verification")
        return True

    computed = sha256(file_path)
    if computed != expected_hash:
        print("Checksum mismatch → corrupted download")
        file_path.unlink(missing_ok=True)
        return False

    print("Checksum verified")
    return True


def extract(zip_path: Path, extract_to: Path):
    
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


def main():
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"

    # ESC-50 does NOT publish official checksum → leave None
    expected_sha256 = None

    root = Path(__file__).resolve().parent.parent.parent
    data_dir = root / "data"
    zip_path = data_dir / "raw" / "ESC-50" / "esc50.zip"
    extract_dir = data_dir / "raw" / "ESC-50" / "extracted"
    

    if extract_dir.exists():
        extract(zip_path, extract_dir)
    else:
        download(url, zip_path)
        extract(zip_path, extract_dir)

    if not verify(zip_path, expected_sha256):
        return

    audio_dir = extract_dir / "ESC-50-master" / "audio"
    if audio_dir.exists():
        print(f"Done: {len(list(audio_dir.glob('*.wav')))} files")
    else:
        print("Unexpected structure")


if __name__ == "__main__":
    main()