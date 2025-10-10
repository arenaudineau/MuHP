import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from muhp import MuHP

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

# config can still be later on modified, before starting actual training or in-between
HP.config = HP.config | dict(OPTIMIZER=optim.SGD)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=HP.BATCH_SIZE,
    shuffle=True,
)

model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 48), nn.Linear(48, 10))

criterion = nn.CrossEntropyLoss()
optimizer = HP.OPTIMIZER(model.parameters(), lr=HP.LEARNING_RATE)

for epoch in HP.lapsed(range(HP.N_EPOCHS)):
    total_correct, total_samples = 0, 0

    # HP.config = HP.config | dict(LEARNING_RATE=1)  # Disallowed
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
    print(
        f"Epoch {epoch+1}, loss = {loss.item():.4f}, acc = {total_correct/total_samples*100:.2f}"
    )
