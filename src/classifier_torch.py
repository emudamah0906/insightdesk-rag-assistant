"""
Intent classifier in PyTorch.

WHY this lives in the project: before RAG answers a ticket, you route it. A small
supervised model classifies intent (billing/technical/account/shipping/complaint) so
high-priority intents go to humans and FAQ-style ones go to the RAG bot.

Pipeline: text -> bag-of-words vector -> MLP -> softmax over intents.
Covers: Dataset/tensors, nn.Module, CrossEntropyLoss, Adam, train/val split, early-ish
stopping, evaluation. Deliberately tiny so it trains in seconds on CPU.

Setup:  pip install torch
Run:    python src/classifier_torch.py
"""
from __future__ import annotations
import sys, pathlib, csv, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from embeddings import tokenize

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "tickets.csv"


def load():
    rows = list(csv.DictReader(open(DATA)))
    texts = [r["text"] for r in rows]
    labels = [r["intent"] for r in rows]
    return texts, labels


def build_vocab(texts):
    vocab = {}
    for t in texts:
        for tok in tokenize(t):
            vocab.setdefault(tok, len(vocab))
    return vocab


def vectorize(text, vocab):
    v = [0.0] * len(vocab)
    for tok in tokenize(text):
        if tok in vocab:
            v[vocab[tok]] += 1.0
    return v


def main():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed. Run:  pip install torch\n"
              "(The rest of the project runs without it.)")
        return

    random.seed(0)
    torch.manual_seed(0)
    texts, labels = load()
    classes = sorted(set(labels))
    cls_idx = {c: i for i, c in enumerate(classes)}
    vocab = build_vocab(texts)

    X = torch.tensor([vectorize(t, vocab) for t in texts], dtype=torch.float32)
    y = torch.tensor([cls_idx[l] for l in labels], dtype=torch.long)

    # stratified-ish shuffle split
    idx = list(range(len(texts)))
    random.shuffle(idx)
    split = int(0.75 * len(idx))
    tr, va = idx[:split], idx[split:]
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

    model = nn.Sequential(
        nn.Linear(len(vocab), 32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, len(classes)),
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for epoch in range(1, 61):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                acc = (model(Xva).argmax(1) == yva).float().mean().item()
            print(f"epoch {epoch:>2}  train_loss={loss.item():.3f}  val_acc={acc:.0%}")
            if acc >= best_acc:
                best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nBest validation accuracy: {best_acc:.0%}  (tiny dataset — illustrative)")

    # inference demo
    model.load_state_dict(best_state)
    model.eval()
    demos = ["my card was charged twice", "the api returns a 401 error",
             "please delete my account", "where is my sensor shipment"]
    print("\nPredictions:")
    with torch.no_grad():
        for d in demos:
            probs = model(torch.tensor([vectorize(d, vocab)], dtype=torch.float32)).softmax(1)[0]
            pred = classes[int(probs.argmax())]
            print(f"  {d:36} -> {pred:10} ({probs.max():.0%})")


if __name__ == "__main__":
    main()
