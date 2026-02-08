import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class DeepShipNet(nn.Module):
    def __init__(self, board_size=10, num_res_blocks=5, num_channels=64):
        super().__init__()
        self.board_size = board_size
        
        # Input: 3 Channels 
        # 1. Hits (1=Hit, 0=Empty)
        # 2. Misses (1=Miss, 0=Empty)
        # 3. Probability Heatmap (0.0 - 1.0 from Math Agent)
        self.start_block = ConvBlock(3, num_channels)
        
        # Body: Residual Tower
        self.backbone = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(num_res_blocks)]
        )
        
        # Head 1: Policy (Where to shoot?)
        # Outputs a probability for every square on the board (100 squares)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1), # Reduce filters
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size)
            # No Softmax here! We use CrossEntropyLoss which includes Softmax
        )
        
        # Head 2: Value (Am I winning?)
        # Outputs a single number between -1 (Lose) and 1 (Win)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.start_block(x)
        for res_block in self.backbone:
            x = res_block(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value