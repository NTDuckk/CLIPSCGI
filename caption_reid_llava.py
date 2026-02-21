import argparse, json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig


def truncate_words(text: str, max_words: int = 45) -> str:
    words = text.strip().split()
    return " ".join(words[:max_words])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--ext", nargs="+", default=[".jpg", ".jpeg", ".png", ".webp"])
    args = ap.parse_args()

    instruction = (
        "Describe the most obvious person only. Ignore background. "
        "Use at most 45 words and follow this template:\n"
        "A [age] [gender] is wearing [upper clothes]. "
        "[gender] is wearing [lower clothes]. "
        "The [gender] is wearing [shoes] and carrying [others/accessories]. "
        "The [gender] has [hair length] and [wearing glasses or not]."
    )

    # ---- FIX: dùng BitsAndBytesConfig (ổn định hơn trên nhiều version) ----
    quant_cfg = None
    if args.load_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    # HF docs khuyên left padding khi batch generate
    processor.tokenizer.padding_side = "left"
    # ---------------------------------------------------------------

    data_root = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = []
    exts = set([e.lower() for e in args.ext])
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            image_paths.append(p)
    image_paths.sort()

    done = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(obj["image_path"])
                except Exception:
                    pass

    with out_path.open("a", encoding="utf-8") as f:
        for img_path in tqdm(image_paths, desc="Captioning"):
            rel = str(img_path.relative_to(data_root))
            if rel in done:
                continue

            image = Image.open(img_path).convert("RGB")
            prompt = f"USER: <image>\n{instruction}\nASSISTANT:"

            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

            decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            caption = decoded.split("ASSISTANT:")[-1].strip()
            caption = truncate_words(caption, 45)

            f.write(json.dumps({"image_path": rel, "caption": caption}, ensure_ascii=False) + "\n")
            f.flush()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()