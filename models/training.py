# Initialize the model
input_dim = 1  # Features (e.g., sales data)
output_dim = 1  # Prediction
model = TimeSeriesTransformer(input_dim=input_dim, output_dim=output_dim).to("cuda")

# Initialize trainer
trainer = TimeSeriesTransformerTrainer(model, learning_rate=1e-4, scaler=torch.cuda.amp.GradScaler())

# Dummy DataLoader (baad mei replace kr denge)
from torch.utils.data import DataLoader, TensorDataset

sequence_length = 30
num_samples = 1000

X = torch.rand((num_samples, sequence_length, input_dim)).to("cuda")
y = torch.rand((num_samples, output_dim)).to("cuda")

train_loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# Training Loop
for epoch in range(20):
    train_loss = 0
    for batch, targets in train_loader:
        loss = trainer.train_step(batch, targets)
        train_loss += loss
    
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}")
