import torch
import torch.nn as nn
import torch.optim as optim
import time
from nn_live import Visualizer

# 1. Define a simple Multi-Layer Perceptron (MLP)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 4)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def main():
    print("Initializing Model...")
    model = SimpleNet()
    
    # 2. Attach the Visualizer
    # This will automatically start the server and open your browser!
    viz = Visualizer(model, port=8000, open_browser=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("\nStarting simulated training loop...")
    print("Go to http://127.0.0.1:8000 if your browser didn't open automatically.")
    print("Press Ctrl+C to stop.\n")
    
    try:
        # 3. Training Loop
        for epoch in range(1000):
            # Create some random dummy data
            inputs = torch.randn(1, 8)  # Batch size 1, 8 features
            targets = torch.randn(1, 4)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # 4. Push updates to the visualizer
            viz.step()
            
            # Slow down training slightly so we can see the animation
            time.sleep(0.5)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
                
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
    finally:
        viz.cleanup()

if __name__ == "__main__":
    main()
