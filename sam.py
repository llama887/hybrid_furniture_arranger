from ultralytics import SAM

# Load a model
model = SAM("./models/sam2.1_b.pt")

# Display model information (optional)
model.info()

# Run inference
results = model("flux-quantized-output.png", save=True)
