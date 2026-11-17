from pathlib import Path
from collections import defaultdict
import argparse
import csv
import gzip
import json
import os
import tempfile

import numpy as np
import torch
from evodiff.pretrained import OA_DM_38M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)

    with gzip.open(source, "rt") as f:
        for row in csv.DictReader(f):
            grouped[row["pdb_fn"]].append((
                int(row["row_position"]),
                row["split"],
                row["sequence"],
            ))

    if args.limit:
        remaining = args.limit
        for protein in sorted(grouped):
            records = grouped[protein]
            grouped[protein] = records[:remaining]
            remaining -= len(grouped[protein])
            if remaining <= 0:
                break
        grouped = {k: v for k, v in grouped.items() if v}

    assert torch.cuda.is_available(), "GPU unavailable"

    model, _, tokenizer, _ = OA_DM_38M()
    model = model.eval().cuda()

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Proteins:", len(grouped), flush=True)

    completed = 0
    total = sum(len(v) for v in grouped.values())

    for protein in sorted(grouped):
        records = sorted(grouped[protein])

        for start in range(0, len(records), 4096):
            subset = records[start:start + 4096]
            dest = output / (
                f"{protein.replace('.pdb', '')}_{start:06d}.npz"
            )

            if dest.exists():
                with np.load(dest) as existing:
                    assert len(existing["row_position"]) == len(subset)
                    assert np.array_equal(
                        existing["row_position"],
                        [r[0] for r in subset],
                    )
                completed += len(subset)
                continue

            vectors = []

            for offset in range(0, len(subset), args.batch_size):
                batch = subset[offset:offset + args.batch_size]
                lengths = {len(r[2]) for r in batch}
                assert len(lengths) == 1

                tokens = torch.tensor(
                    [
                        [tokenizer.a_to_i[aa] for aa in r[2]]
                        for r in batch
                    ],
                    dtype=torch.long,
                    device="cuda",
                )

                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                    ):
                        hidden = model.last_norm(
                            model.embedder(tokens, None)
                        )
                        pooled = hidden.mean(dim=1)

                vectors.append(
                    pooled.float().cpu().numpy().astype(np.float16)
                )

            embeddings = np.concatenate(vectors, axis=0)
            assert np.isfinite(embeddings).all()
            assert embeddings.shape == (len(subset), 1024)

            with tempfile.NamedTemporaryFile(
                dir=output,
                prefix=".partial_",
                suffix=".npz",
                delete=False,
            ) as f:
                temporary = f.name
                np.savez_compressed(
                    f,
                    embeddings=embeddings,
                    row_position=np.array(
                        [r[0] for r in subset], dtype=np.int32
                    ),
                    split=np.array(
                        [r[1] for r in subset]
                    ),
                )

            os.replace(temporary, dest)
            completed += len(subset)

            print(
                f"{protein}: {min(start + len(subset), len(records)):,}"
                f"/{len(records):,} | total {completed:,}/{total:,}",
                flush=True,
            )

    assert completed == total

    with open(output / "summary.json", "w") as f:
        json.dump({
            "model": "OA_DM_38M",
            "pooling": "mean of final normalized ByteNet hidden states",
            "dtype": "float16",
            "dimension": 1024,
            "sequences": completed,
            "proteins": len(grouped),
            "batch_size": args.batch_size,
            "torch_version": torch.__version__,
        }, f, indent=2)

    (output / "SUCCESS").touch()
    print("EXTRACTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
