# extract image feature via DETR
import torch
from PIL import Image
import torchvision.transforms as T

# running following code, to download DETR offical code and model
# see https://github.com/facebookresearch/detr
model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
model.eval()

# load an image
im = Image.open('flickr30k/flickr30k-images/36979.jpg')

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)

# propagate through the model
outputs = model(img)

# outputs return 6 decoder layers' features
# we get the lastest layer's feature
print(outputs[-1].shape)
