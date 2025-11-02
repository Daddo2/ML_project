import torch
from torch import nn
import torch.nn.functional as F


#print(torch.__version__)
#print(torch.version.cuda)  # mostra la versione di CUDA collegata
#print(torch.cuda.is_available())  # dice se PyTorch può usare la GPU

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Test the DQN network
    state_dim = 1  # Example state dimension
    action_dim = 4  # Example action dimension
    model = DQN(state_dim, action_dim)

    sample_input = torch.randn(1, state_dim)  # Batch size of 1
    output = model(sample_input)
    print("Output shape:", output.shape)  # Should be [1, action_dim]
    print("Output:", output)