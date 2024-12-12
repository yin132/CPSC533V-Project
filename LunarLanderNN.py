import torch

# Pytorch MiniModel: https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
class LunarLanderNN(torch.nn.Module):
    def __init__(self):
        super(LunarLanderNN, self).__init__()

        self.linear1 = torch.nn.Linear(8, 32)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(32, 32)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(32, 4)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x
    
    # From A2
    def select_action(self, state):
        self.eval()
        x = self.forward(state)
        self.train()

        return x.max(1)[1].view(1, 1).to(torch.long)
    
    def load_chromosome(self, C):
        with torch.no_grad():
            curr = 0
            self.linear1.weight = torch.nn.Parameter(torch.tensor(C[:curr + 8*32].reshape((32, 8)), dtype=torch.float32))
            curr += 8 * 32
            self.linear1.bias = torch.nn.Parameter(torch.tensor(C[curr:curr + 32], dtype=torch.float32))
            curr += 32
            self.linear2.weight = torch.nn.Parameter(torch.tensor(C[curr:curr + 32*32].reshape((32, 32)), dtype=torch.float32))
            curr += 32 * 32
            self.linear2.bias = torch.nn.Parameter(torch.tensor(C[curr:curr + 32], dtype=torch.float32))
            curr += 32
            self.linear3.weight = torch.nn.Parameter(torch.tensor(C[curr:curr + 32*4].reshape((4, 32)), dtype=torch.float32))
            curr += 32 * 4
            self.linear3.bias = torch.nn.Parameter(torch.tensor(C[curr:curr + 4], dtype=torch.float32))