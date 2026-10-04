import torch, torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet18
from PIL import Image

CLASSES = ["angry", "happy", "neutral", "sad"]
FER_MEAN, FER_STD = 0.4924, 0.2532
IMG_SIZE_RESNET = 224
DEVICE = torch.device("cpu")

class EmotionCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(128 * 6 * 6, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

_cnn_transform = T.Compose([T.Resize((48, 48)), T.ToTensor(), T.Normalize([FER_MEAN], [FER_STD])])
_resnet_transform = T.Compose([
    T.Resize((IMG_SIZE_RESNET, IMG_SIZE_RESNET)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_models = {}

def _load_cnn():
    m = EmotionCNN()
    m.load_state_dict(torch.load("models/best_cnn.pth", map_location=DEVICE))
    m.eval()
    return m

def _load_resnet():
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load("models/best_resnet.pth", map_location=DEVICE))
    m.eval()
    return m

def get_model(name):
    if name not in _models:
        _models[name] = {"cnn": _load_cnn, "resnet": _load_resnet}[name]()
    return _models[name]

def predict(pil_image, model_name="cnn"):
    img = pil_image.convert("L")
    model = get_model(model_name)
    transform = _cnn_transform if model_name == "cnn" else _resnet_transform
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    return {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}