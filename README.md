# µHP

Lightweight hyperparameters handling

```py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from muhp import MuHP
from tqdm import tqdm

HP = MuHP(
    name="my_experiment",
    config=dict(
        KEY=1234,
        BATCH_SIZE=32,
        LEARNING_RATE=1e-2,
        N_EPOCHS=5,
        OPTIMIZER=optim.Adam,
    ),
)

torch.manual_seed(HP.KEY)

# config can still be later on modified, before starting actual training
HP.OPTIMIZER = optim.SGD

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=HP.BATCH_SIZE,
    shuffle=True,
)

model = nn.Sequential(
    nn.Flatten(), nn.Linear(28 * 28, 48), nn.ReLU(), nn.Linear(48, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = HP.OPTIMIZER(model.parameters(), lr=HP.LEARNING_RATE)

for epoch in (pbar := tqdm(HP.lapsed(range(HP.N_EPOCHS)))):
    total_correct, total_samples = 0, 0

    # HP.LEARNING_RATE = 1. # Disallowed
    for x, y in train_loader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    HP.log("loss", loss.detach())
    HP.log("accuracy", total_correct / total_samples)

    pbar.set_postfix(HP.lapse_metrics)
```

will result in the outputs:
```
muhp
└── my_experiment
    ├── config.json
    ├── metrics_lapse_final.npz
    ├── gitdiff.patch
    └── _completed_sentinel
```

with:

- `config.json`:
```json
{
  "KEY": 1234,
  "BATCH_SIZE": 32,
  "LEARNING_RATE": 0.01,
  "N_EPOCHS": 5,
  "OPTIMIZER": "SGD"
}
```

- `metrics_lapse_final.npz`:
```json
{
    "loss": [0.59373546, 0.1961537, 0.31670108, 0.45723966, 0.1811399 ],
    "accuracy": [0.81868333, 0.8942, 0.90448333, 0.90948333, 0.91303333]
}
```

- `gitdiff.patch`: providing you run your simulation in a folder with a `.git` directory, the last commit hash and diffs are also stored to be able to retrieve exactly the code used for running.
```diff
# current-commit: 5040a6ba3026f29ea2981a25e0c9d87c73cb5828

diff --git a/src/muhp/__init__.py b/src/muhp/__init__.py
index cd57eb4..2697d2e 100644
--- a/src/muhp/__init__.py
+++ b/src/muhp/__init__.py
@@ -46,7 +46,7 @@ class MuHP:
                 .decode("ascii")
                 .strip()
             )
-            git_diff = subprocess.check_output(["git", "diff"]).decode("ascii").strip()
+            git_diff = subprocess.check_output(["git", "diff"]).decode("ascii")
 
             with open(self.path / "gitdiff.patch", "w") as f:
                 f.write("# current-commit: " + git_commit_hash + "\n\n")

```

- `_completed_sentinel`: empty file used to assess the completion of a run, for error recovery.