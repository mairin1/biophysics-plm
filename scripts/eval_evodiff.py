import argparse
from pathlib import Path
import torch
from torch import nn
from train_evodiff_biophysics import load_baseline_model
import csv
from sklearn.model_selection import train_test_split


class Embedder():
    def __init__(self, model_path=None) -> None:
        super().__init__()
        model, self.tokenizer = load_baseline_model()
        if model_path:
            state = torch.load(model_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state['model_state_dict'])
        self.embedder = model.embedder.requires_grad_(False)
        self.last_norm = model.last_norm.requires_grad_(False)

    def embed(self, x):
        embeddings = self.last_norm(self.embedder(x, None))
        return embeddings.mean(dim=1).float()
    

class PGData():
    def __init__(self, paths, tokenizer):
        self.tokens = []
        self.labels = []
        for path in paths:
            files = Path(path).glob("*.csv")
            for file in files:
                with open(file, 'r') as read_obj:
                    csv_reader = csv.reader(read_obj)
                    list_of_csv = list(csv_reader)
                    for i, row in enumerate(list_of_csv):
                        if i != 0: # skip columnn names
                            label = int(row[-1]) if len(row[-1]) == 1 else (0 if row[-1] == 'Benign' else 1)
                            self.tokens.append(
                                [tokenizer.a_to_i[a] for a in row[-2]],
                            )
                            self.labels.append(label)
        
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {'tokens': self.tokens[idx], 'label': self.labels[idx]}


def regression(tokens, labels, train_data, test_data, embedder, seed):
    # simple regression to isolate effects of embeddings as much as possible
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True) # model location
    parser.add_argument('--source', type=Path, required=True) # inputs to evaluate
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=False, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    assert (torch.cuda.is_available())
    model_pt = Embedder(args.model) # Post-trained embeddings
    model_default = Embedder() # Baseline model
    # Load data
    data = PGData(['data/protein_gym/indel', 'data/protein/sub'], model_default.tokenizer)
    # TODO: could have a better split method based on structure/sequence
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    


if __name__ == '__main__':
    main()