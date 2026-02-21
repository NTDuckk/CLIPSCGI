import json
import csv
from pathlib import Path
from typing import Dict

def load_captions(path: str) -> Dict[str, str]:
    """Load offline captions mapping.

    Supported formats:
      - JSON: either {key: caption, ...} or [{"key":..., "caption":...}, ...]
      - CSV : columns must include a key column and a caption/text column.
             Common names: key,image,filename,path ; caption,text,desc

    Keys can be full image paths or basenames. The dataset will try both.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Caption file not found: {path}")
    suffix = p.suffix.lower()

    if suffix in [".json"]:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        if isinstance(data, list):
            out = {}
            for row in data:
                if not isinstance(row, dict):
                    continue
                # guess columns
                key = row.get("key") or row.get("image") or row.get("path") or row.get("filename") or row.get("img")
                cap = row.get("caption") or row.get("text") or row.get("desc") or row.get("description")
                if key is None or cap is None:
                    continue
                out[str(key)] = str(cap)
            return out
        raise ValueError("Unsupported JSON structure for captions.")
    elif suffix in [".csv", ".tsv"]:
        delimiter = "," if suffix == ".csv" else "\t"
        out = {}
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            # infer columns
            key_cols = ["key","image","path","filename","img"]
            cap_cols = ["caption","text","desc","description"]
            for row in reader:
                k = None
                for c in key_cols:
                    if c in row and row[c]:
                        k = row[c]; break
                v = None
                for c in cap_cols:
                    if c in row and row[c]:
                        v = row[c]; break
                if k is None or v is None:
                    continue
                out[str(k)] = str(v)
        return out
    else:
        raise ValueError(f"Unsupported caption file type: {suffix}")