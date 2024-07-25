# Let us now functionize the model selection
import torchvision
from torch import nn

def create_model(model_name: str,
                 out_features: int=3,
                 device: str=device):
    assert model_name == "effnetb2" or model_name == "effnetv2_s", "Model name should be effnetb2 or effnetv2_s"
    if model_name == "effnetb2":
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights).to(device)
        dropout = 0.3
        in_features = 1408
    elif model_name == "effnetv2_s":
        weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        model = torchvision.models.efficientnet_v2_s(weights=weights).to(device)
        dropout = 0.2
        in_features = 1280

    # Freeze the base layer of the models
    for param in model.parameters():
        param.requires_grad = False

    # Update the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features,out_features=out_features)
    ).to(device)

    # set the model name
    model.name = model_name
    print(f"[INFO] Creating {model_name} feature extractor model...")
    return model
