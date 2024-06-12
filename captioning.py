import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer

model_name = "nlpconnect/vit-gpt2-image-captioning"
image_captioning_model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = feature_extractor(images=image, return_tensors="pt").pixel_values
    return image

image_path = 'image.jpg'  # Replace with the path to your image
image = preprocess_image(image_path)


def generate_caption(image, model, tokenizer, max_length=16):
    output_ids = model.generate(image, max_length=max_length, num_beams=5, early_stopping=True)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

caption = generate_caption(image, image_captioning_model, tokenizer)
print("Generated Caption:", caption)


def display_image_with_caption(image_path, caption):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(caption)
    plt.axis('off')
    plt.show()

display_image_with_caption(image_path, caption)
