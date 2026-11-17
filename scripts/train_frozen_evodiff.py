from collections import Counter
from itertools import zip_longest
from pathlib import Path
import argparse
import csv
import gzip
import json
import random

import numpy as np
import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    assert torch.cuda.is_available()
    args.out.mkdir(parents=True, exist_ok=True)

    ids, proteins, groups, lengths, labels = [], [], [], [], []

    with gzip.open(args.source / "sequences.csv.gz", "rt") as f1, \
         gzip.open(args.source / "rosetta_numeric.csv.gz", "rt") as f2:

        sequences = csv.DictReader(f1)
        targets = csv.DictReader(f2)

        for a, b in zip_longest(sequences, targets):
            assert a is not None and b is not None
            pos = int(a["row_position"])
            assert pos == int(b["row_position"])

            ids.append(pos)
            proteins.append(a["pdb_fn"])
            groups.append(a["role"] + "/" + a["split"])
            lengths.append(len(a["sequence"]))
            labels.append([
                float(b["fa_atr"]),
                float(b["fa_sol"]),
            ])

    assert len(ids) == 274000
    assert len(set(ids)) == len(ids)
    assert Counter(groups) == {
        "train/train": 240000,
        "train/val": 12000,
        "train/test": 12000,
        "holdout/test": 10000,
    }
    assert np.isfinite(labels).all()

    lookup = {pos: i for i, pos in enumerate(ids)}
    features = np.empty((len(ids), 1024), dtype=np.float16)
    seen = np.zeros(len(ids), dtype=bool)
    chunks = sorted(args.embeddings.glob("*.npz"))
    assert len(chunks) == 70

    for path in chunks:
        with np.load(path) as chunk:
            positions = chunk["row_position"]
            indices = np.fromiter(
                (lookup[int(pos)] for pos in positions),
                dtype=np.int64,
                count=len(positions),
            )

            assert not seen[indices].any()
            assert chunk["embeddings"].shape == (len(indices), 1024)
            assert np.isfinite(chunk["embeddings"]).all()
            assert all(
                groups[i].split("/")[1] == str(s)
                for i, s in zip(indices, chunk["split"])
            )

            features[indices] = chunk["embeddings"]
            seen[indices] = True

    assert seen.all()

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Loaded:", len(ids), "embeddings", flush=True)

    x = torch.from_numpy(features).to(
        device=device, dtype=torch.float32
    )

    raw = np.asarray(labels, dtype=np.float32)
    lengths_array = np.asarray(lengths, dtype=np.float32)

    y = torch.from_numpy(
        raw / lengths_array[:, None]
    ).to(device)

    train_indices = torch.as_tensor(
        [i for i, g in enumerate(groups) if g == "train/train"],
        dtype=torch.long,
        device=device,
    )

    val_indices = torch.as_tensor(
        [i for i, g in enumerate(groups) if g == "train/val"],
        dtype=torch.long,
        device=device,
    )

    x_mean = x[train_indices].mean(dim=0)
    x_std = x[train_indices].std(dim=0).clamp_min(1e-6)
    y_mean = y[train_indices].mean(dim=0)
    y_std = y[train_indices].std(dim=0).clamp_min(1e-6)

    x.sub_(x_mean).div_(x_std)
    y.sub_(y_mean).div_(y_std)

    model = nn.Sequential(
        nn.Linear(1024, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Linear(128, 2),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.001,
    )

    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    patience = 6

    with open(args.out / "history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            order = train_indices[
                torch.randperm(len(train_indices), device=device)
            ]

            total_loss = 0.0

            for batch in order.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x[batch])
                loss = loss_fn(prediction, y[batch])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)

            train_loss = total_loss / len(train_indices)

            model.eval()
            with torch.inference_mode():
                val_loss = loss_fn(
                    model(x[val_indices]),
                    y[val_indices],
                ).item()

            writer.writerow([epoch, train_loss, val_loss])
            f.flush()

            print(
                f"Epoch {epoch}: "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f}",
                flush=True,
            )

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_epoch = epoch

                torch.save({
                    "model_state": model.state_dict(),
                    "x_mean": x_mean.cpu(),
                    "x_std": x_std.cpu(),
                    "y_mean": y_mean.cpu(),
                    "y_std": y_std.cpu(),
                    "best_epoch": best_epoch,
                }, args.out / "best_model.pt")

            elif epoch - best_epoch >= patience:
                break

    checkpoint = torch.load(
        args.out / "best_model.pt",
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    evaluation = [
        i for i, g in enumerate(groups)
        if g != "train/train"
    ]

    predictions = np.empty(
        (len(evaluation), 2), dtype=np.float32
    )

    with torch.inference_mode():
        for start in range(0, len(evaluation), 8192):
            indices = evaluation[start:start + 8192]
            normalized = model(x[indices])
            prediction = (
                normalized * y_std + y_mean
            ).cpu().numpy()

            predictions[start:start + len(indices)] = (
                prediction * lengths_array[indices, None]
            )

    with gzip.open(
        args.out / "predictions.csv.gz", "wt", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "row_position", "pdb_fn", "group",
            "true_fa_atr", "pred_fa_atr",
            "true_fa_sol", "pred_fa_sol",
        ])

        for j, i in enumerate(evaluation):
            writer.writerow([
                ids[i], proteins[i], groups[i],
                raw[i, 0], predictions[j, 0],
                raw[i, 1], predictions[j, 1],
            ])

    metrics = []

    for protein in sorted(set(proteins)):
        for group in (
            "train/val",
            "train/test",
            "holdout/test",
        ):
            selected = [
                j for j, i in enumerate(evaluation)
                if proteins[i] == protein and groups[i] == group
            ]

            if not selected:
                continue

            true = raw[np.asarray(evaluation)[selected]]
            predicted = predictions[selected]

            for col, name in enumerate(("fa_atr", "fa_sol")):
                difference = true[:, col] - predicted[:, col]
                denominator = np.sum(
                    (true[:, col] - true[:, col].mean()) ** 2
                )

                r2 = (
                    float(
                        1 - np.sum(difference ** 2) / denominator
                    )
                    if denominator > 0 else None
                )

                metrics.append([
                    protein, group, name, len(selected),
                    float(np.mean(np.abs(difference))),
                    float(np.sqrt(np.mean(difference ** 2))),
                    r2,
                ])

    with open(args.out / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pdb_fn", "group", "property",
            "n", "MAE", "RMSE", "R2",
        ])
        writer.writerows(metrics)

    with open(args.out / "summary.json", "w") as f:
        json.dump({
            "model": "Frozen EvoDiff OA_DM_38M + MLP",
            "training_variants": len(train_indices),
            "validation_variants": len(val_indices),
            "best_epoch": best_epoch,
            "best_val_mse_standardized": best_val,
            "evaluation_rows": len(evaluation),
            "targets": ["fa_atr", "fa_sol"],
            "target_scaling": "per-residue, train-only mean/std",
            "feature_scaling": "train-only mean/std",
        }, f, indent=2)

    (args.out / "SUCCESS").touch()
    print("FROZEN EVODIFF BASELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
