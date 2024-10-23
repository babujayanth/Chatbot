import torch
from torchvision import models, transforms
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("example.jpg")
img_tensor = preprocess(img).unsqueeze(0)

with torch.no_grad():
    img_features = resnet(img_tensor)

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Example prompt (image caption or object description)
prompt = "This image shows a busy market scene with people buying vegetables."

inputs = tokenizer.encode(prompt, return_tensors='pt')
outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
print(outputs)

for i, output in enumerate(outputs):
    question = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Generated Question {i + 1}: {question}")
