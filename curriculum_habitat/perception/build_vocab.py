"""Build the object_id -> CLIP-name-vector cache used by the graph encoder.

The graph's `object_id` field indexes into a category vocabulary. Here we collect every
category that appears across the 10 scenes, add the 6 ObjectNav goal categories, and
store (a) the category->id map and (b) a frozen [V, 512] matrix of CLIP text embeddings.

Outputs:
    scene_graphs/vocab/categories.json         {category: object_id}
    scene_graphs/vocab/clip_text_embeddings.pt  float tensor [V, 512] (row = object_id)

Run: python -m curriculum_habitat.perception.build_vocab
"""

import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SEM_DIR = REPO_ROOT / "scene_graphs" / "semantics"
OUT_DIR = REPO_ROOT / "scene_graphs" / "vocab"

# ObjectNav goal categories (canonical). Always present so the builder can look up the
# episode target by name even if the raw annotation label differs (couch vs sofa, ...).
GOAL_CATEGORIES = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
CLIP_MODEL = "openai/clip-vit-base-patch32"


def collect_categories():
    cats = set(GOAL_CATEGORIES)
    for path in sorted(SEM_DIR.glob("*.json")):
        for obj in json.loads(path.read_text())["objects"]:
            cats.add(obj["category"].strip().lower())
    return sorted(cats)


def clip_text_embeddings(categories):
    from transformers import CLIPModel, CLIPTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in categories]
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(prompts), 128):
            batch = tokenizer(prompts[i:i + 128], padding=True, return_tensors="pt").to(device)
            feats = model.get_text_features(**batch)
            embeddings.append(torch.nn.functional.normalize(feats, dim=-1).cpu())
    return torch.cat(embeddings, dim=0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    categories = collect_categories()
    vocab = {c: i for i, c in enumerate(categories)}
    embeddings = clip_text_embeddings(categories)

    (OUT_DIR / "categories.json").write_text(json.dumps(vocab, indent=2))
    torch.save(embeddings, OUT_DIR / "clip_text_embeddings.pt")
    print(f"vocab size: {len(vocab)}  |  CLIP table: {tuple(embeddings.shape)}")
    print("goal ids:", {c: vocab[c] for c in GOAL_CATEGORIES})
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}/categories.json + clip_text_embeddings.pt")


if __name__ == "__main__":
    main()
