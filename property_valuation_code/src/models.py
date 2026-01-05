import torch
import torch.nn as nn
import torchvision.models as models
class CNNImageEncoder(nn.Module):
    def __init__(self, out_dim=32, freeze_backbone=True):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return x
class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
class LateFusionModel(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()

        self.image_encoder = CNNImageEncoder(out_dim=32)
        self.tabular_encoder = TabularMLP(tabular_dim, hidden_dim=64)

        self.regressor = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img, tabular):
        img_emb = self.image_encoder(img)
        tab_emb = self.tabular_encoder(tabular)

        fused = torch.cat([tab_emb, img_emb], dim=1)
        return self.regressor(fused).squeeze(1)
class CNNResidualRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = CNNImageEncoder(out_dim=32)

        self.head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, img):
        emb = self.image_encoder(img)
        return self.head(emb).squeeze(1)
class DualZoomResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder16 = CNNImageEncoder(out_dim=32)
        self.encoder18 = CNNImageEncoder(out_dim=8)

        self.head = nn.Sequential(
            nn.Linear(32 + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, img16, img18):
        emb16 = self.encoder16(img16)
        emb18 = self.encoder18(img18)

        fused = torch.cat([emb16, 0.3 * emb18], dim=1)
        return self.head(fused).squeeze(1)


class WeightedLateFusionModel(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()

        self.image_encoder = CNNImageEncoder(out_dim=32)
        self.tabular_encoder = TabularMLP(tabular_dim, hidden_dim=64)

        # Learnable scalar weight for image branch
        self.alpha = nn.Parameter(torch.tensor(0.2))

        self.regressor = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img, tabular):
        img_emb = self.image_encoder(img)
        tab_emb = self.tabular_encoder(tabular)

        # Apply learnable weight
        fused = torch.cat([tab_emb, self.alpha * img_emb], dim=1)
        return self.regressor(fused).squeeze(1)