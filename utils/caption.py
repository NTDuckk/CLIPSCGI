import json, os

def truncate_words(text: str, max_words: int = 45) -> str:
    if text is None:
        return ""
    words = str(text).strip().split()
    return " ".join(words[:max_words])

def load_caption_maps(path: str, max_words: int = 45):
    """
    Hỗ trợ:
    - JSON list: [ {"image_path":..., "caption":...}, ... ]
    - JSON single record: {"image_path":..., "caption":...}
    - JSONL: mỗi dòng 1 record như trên
    Trả về:
      cap_by_img: dict[str, str]  (key = basename + full image_path)
      cap_by_pid: dict[int, str]  (fallback, 1 caption/ID)
    """
    cap_by_img = {}
    cap_by_pid = {}

    def add_record(ip: str, cap: str):
        if not ip:
            return
        cap = truncate_words(cap, max_words)
        cap_by_img[ip] = cap
        cap_by_img[os.path.basename(ip)] = cap

        # fallback theo pid (label) nếu cần
        base = os.path.basename(ip)
        try:
            pid = int(base.split("_")[0])
            cap_by_pid.setdefault(pid, cap)  # giữ caption đầu tiên của pid
        except Exception:
            pass

    # Try parse as JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and ("image_path" in data and "caption" in data):
            add_record(data.get("image_path"), data.get("caption", ""))
            return cap_by_img, cap_by_pid

        if isinstance(data, list):
            for r in data:
                if isinstance(r, dict):
                    add_record(r.get("image_path"), r.get("caption", ""))
            return cap_by_img, cap_by_pid

        # Nếu là dict mapping kiểu {"0002.jpg":"..."} thì cũng handle
        if isinstance(data, dict):
            for k, v in data.items():
                add_record(k, v)
            return cap_by_img, cap_by_pid

    except Exception:
        pass

    # Fallback parse as JSONL
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, dict):
                add_record(r.get("image_path"), r.get("caption", ""))

    return cap_by_img, cap_by_pid