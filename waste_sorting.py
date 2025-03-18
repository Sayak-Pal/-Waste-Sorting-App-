import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
import gradio as gr

# Load model and image processor
model_name = "watersplash/waste-classification"  # Change to a valid model
model = AutoModelForImageClassification.from_pretrained(model_name)
image_processor = AutoImageProcessor.from_pretrained(model_name)

# Define preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define multi-label prediction function
def predict_waste(image):
    image = Image.fromarray(image)  # Convert NumPy array to PIL image
    input_tensor = preprocess_image(image)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_tensor)

    # Apply sigmoid activation for multi-label classification
    probabilities = torch.sigmoid(outputs.logits)[0]  # Convert logits to probabilities

    # Set a threshold to select labels (e.g., >= 50%)
    threshold = 0.5
    predicted_labels = [label for idx, label in model.config.id2label.items() if probabilities[idx] >= threshold]
    confidence_scores = [f"{probabilities[idx] * 100:.2f}%" for idx in range(len(probabilities)) if probabilities[idx] >= threshold]

    if predicted_labels:
        result = "\n".join([f"{label}: {score}" for label, score in zip(predicted_labels, confidence_scores)])
    else:
        result = "No clear classification (confidence below threshold)"

    return result

# Create Gradio interface
interface = gr.Interface(
    fn=predict_waste,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Multi-Label Waste Sorting App",
    description="Upload an image of waste. The model will classify it into multiple waste categories with confidence scores."
)

# Launch the app
interface.launch()
