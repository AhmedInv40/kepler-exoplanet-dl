import torch
from src.models.resnet import ResNetTabular
from src.models.ft_transformer import FTTransformer


def test_resnet_forward_shape():
    model = ResNetTabular(in_features=51, n_classes=3)
    x = torch.randn(4, 51)
    out = model(x)
    assert out.shape == (4, 3)


def test_ft_transformer_forward_shape():
    model = FTTransformer(in_features=51, n_classes=3)
    x = torch.randn(4, 51)
    out = model(x)
    assert out.shape == (4, 3)
