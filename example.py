# import torch
# from vit_rgts.main import VitRGTS

# v = VitRGTS(
#     image_size = 256,
#     patch_size = 32,
#     num_classes = 1000,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

# img = torch.randn(1, 3, 256, 256)
# print(f'Input image shape: {img}') # Input image shape: torch.Size([1, 3, 256, 256])

# preds = v(img) # (1, 1000)
# print(f"Output tensors shape: {preds}") # Output tensors shape: torch.Size([1, 1000])
# print(preds.shape)



import warnings
# Silence specific FutureWarning coming from PyTorch's sdp_kernel deprecation
# This prevents the noisy deprecation message during training. If you prefer
# to see future warnings (recommended for debugging), remove this filter and
# either upgrade PyTorch or update code that calls the deprecated API.
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.backends\.cuda\.sdp_kernel.*",
    category=FutureWarning,
)




# import torch
# import torch.nn.functional as F  # We need this for the softmax function
# from vit_rgts.main import VitRGTS

# # 1. Initialize Model
# v = VitRGTS(
#     image_size = 256,
#     patch_size = 32,
#     num_classes = 1000,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
# # Set model to evaluation mode (turns off dropout, etc.)
# v.eval() 

# # 2. Create a random dummy image
# img = torch.randn(1, 3, 256, 256)

# # --- FIX 1: Print the .shape, not the whole tensor ---
# print(f'Input image shape: {img.shape}') 

# # 3. Get predictions
# # Use torch.no_grad() for inference to save memory
# with torch.no_grad():
#     preds = v(img) # This is a tensor of raw scores (logits)

# # --- FIX 2: Print the .shape, not the whole tensor ---
# print(f"Output tensor shape: {preds.shape}")

# # 4. --- NEW: Get a truly understandable output ---

# # Convert the raw scores (logits) into probabilities (from 0% to 100%)
# # F.softmax turns the 1000 scores into 1000 probabilities that add up to 1.0
# probabilities = F.softmax(preds, dim=1)

# # Find the highest probability and its class index
# # torch.max() returns two things: the (max_value, max_index)
# confidence, predicted_index = torch.max(probabilities, 1)

# print("\n" + "="*30)
# print("   Understandable Prediction")
# print("="*30)
# # .item() extracts the single number from the tensor
# print(f"Predicted Class Index: {predicted_index.item()}")
# print(f"Confidence in prediction: {confidence.item() * 100:.2f}%")



import torch
import torch.nn.functional as F
from vit_rgts.main import VitRGTS

# --- 1. DEFINE THE MODEL ---
# It MUST match the parameters from train.py
# Your train.py used image_size=224, patch_size=14, and num_classes=10

model = VitRGTS(
    image_size = 224,    # <-- Must match train.py
    patch_size = 14,    # <-- Must match train.py
    num_classes = 10,     # <-- Must be 10 for CIFAR-10
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# --- 2. LOAD THE TRAINED WEIGHTS ---
# This loads the "brain" you saved in train.py
try:
    model.load_state_dict(torch.load("mega_vit_model.pth"))
    print("✅ Successfully loaded trained model 'mega_vit_model.pth'")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Did you run train.py first to create 'mega_vit_model.pth'?")
    exit()

# Set model to evaluation mode (turns off dropout, etc.)
model.eval() 

# --- 3. PREPARE A DUMMY IMAGE ---
# Create a random dummy image.
# It MUST match the image size the model was trained on (224x224).
img = torch.randn(1, 3, 224, 224)
print(f"Input image shape: {img.shape}")

# --- 4. GET PREDICTION ---
with torch.no_grad(): # No gradients needed for inference
    preds = model(img) # Output will be shape [1, 10]

print(f"Output tensor shape: {preds.shape}") # This will now be [1, 10]

# --- 5. INTERPRET THE RESULT ---
probabilities = F.softmax(preds, dim=1)
confidence, predicted_index = torch.max(probabilities, 1)

# CIFAR-10 class names in the correct order
class_names = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

# Get the name of the predicted class
predicted_class_name = class_names[predicted_index.item()]

print("\n" + "="*30)
print("   Prediction from TRAINED Model")
print("="*30)
print(f"Predicted Class Index: {predicted_index.item()}")
print(f"Predicted Class Name: {predicted_class_name}")
print(f"Confidence in prediction: {confidence.item() * 100:.2f}%")