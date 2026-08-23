"""Measure CLIP zero-shot body-region accuracy against filename ground truth.

The filename-labelled subset of DermNet is the only labelled data available
here, so it is what decides whether CLIP may be trusted on the unlabelled
remainder — and where the accept/reject thresholds sit.
"""
import glob
import os
import sys

import torch
import open_clip
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_prep"))
import body_region as BR

MODEL_NAME, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"


def load():
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    prompts = [p for _, p in BR.CLIP_PROMPTS]
    with torch.no_grad():
        text = model.encode_text(tokenizer(prompts))
        text /= text.norm(dim=-1, keepdim=True)
    return model, preprocess, text


def score(model, preprocess, text_features, path):
    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(image)
        features /= features.norm(dim=-1, keepdim=True)
        probs = (100.0 * features @ text_features.T).softmax(dim=-1)[0]
    intimate = sum(float(probs[i]) for i, (kind, _) in enumerate(BR.CLIP_PROMPTS) if kind == BR.INTIMATE)
    return intimate


def main():
    model, preprocess, text_features = load()
    truth = []
    for cls in ("Acne", "Eczema"):
        for path in sorted(glob.glob(f"data_prep/downloads/dermnet/{cls}/*")):
            if not path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            verdict, detail = BR.classify_filename(path)
            if verdict in (BR.INTIMATE, BR.SAFE):
                truth.append((path, verdict, detail))

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    intimate = [t for t in truth if t[1] == BR.INTIMATE]
    safe = [t for t in truth if t[1] == BR.SAFE][:limit]
    sample = intimate + safe
    print(f"dogrulama kumesi: {len(intimate)} intimate + {len(safe)} safe = {len(sample)}")

    scores = []
    for index, (path, verdict, detail) in enumerate(sample):
        try:
            scores.append((score(model, preprocess, text_features, path), verdict, os.path.basename(path), detail))
        except Exception as error:
            print("atlandi", path, error)
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(sample)}", flush=True)

    for threshold in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        tp = sum(1 for s, v, *_ in scores if v == BR.INTIMATE and s >= threshold)
        fn = sum(1 for s, v, *_ in scores if v == BR.INTIMATE and s < threshold)
        fp = sum(1 for s, v, *_ in scores if v == BR.SAFE and s >= threshold)
        tn = sum(1 for s, v, *_ in scores if v == BR.SAFE and s < threshold)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        print(f"esik {threshold:.2f}: recall={recall:.3f} precision={precision:.3f}  TP={tp} FN={fn} FP={fp} TN={tn}")

    print("\nen yuksek skorlu SAFE ornekleri (yanlis pozitif adaylari):")
    for s, v, name, detail in sorted([x for x in scores if x[1] == BR.SAFE], key=lambda x: -x[0])[:10]:
        print(f"   {s:.3f}  {name[:44]:<46} ({detail})")
    print("\nen dusuk skorlu INTIMATE ornekleri (kacanlar):")
    for s, v, name, detail in sorted([x for x in scores if x[1] == BR.INTIMATE], key=lambda x: x[0])[:10]:
        print(f"   {s:.3f}  {name[:44]:<46} ({detail})")


if __name__ == "__main__":
    main()
