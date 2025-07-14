import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import zipfile
import numpy as np
import cv2
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import gradio as gr

# ---------------------------
# Model and Utility Functions
# ---------------------------
IMG_H = 256
IMG_W = 256
smooth = 1e-15

# Custom Dice coefficient and loss (needed for model loading)
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# ---------------------------
# Load Multiple Models
# ---------------------------
# Update these paths with your actual .keras files.
MODELS = {
    "UNet3+ Original": "./model/original.keras",
    "ResNet50": "./model/resnet50.keras",
    "MobileNet": "./model/mobilenet.keras",
    "MobileNetV2": "./model/mobilenetv2.keras",
    "DenseNet121": "./model/densenet121.keras",
    "VGG16": "./model/vgg16.keras"
}

# Pre-load the models into a dictionary for quick inference.
loaded_models = {}
for model_name, model_path in MODELS.items():
    try:
        loaded_models[model_name] = tf.keras.models.load_model(
            model_path,
            custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef}
        )
        print(f"Loaded {model_name} from {model_path}")
    except Exception as e:
        print(f"Error loading {model_name} from {model_path}: {e}")

# ---------------------------
# Prediction Function
# ---------------------------
def predict_mask_image(image: np.ndarray, model) -> np.ndarray:
    """
    Preprocess the image, predict the mask with the given model,
    and resize the mask to the original image dimensions.
    """
    original_h, original_w = image.shape[:2]
    # Resize and normalize
    x = cv2.resize(image, (IMG_W, IMG_H))
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    # Predict the mask
    pred = model.predict(x, verbose=0)[0]
    # (Optional) Threshold if needed:
    # pred = (pred > 0.5).astype(np.float32)
    pred_mask = cv2.resize(pred, (original_w, original_h))
    mask = (pred_mask * 255).astype(np.uint8)
    return mask

# ---------------------------
# FastAPI Application
# ---------------------------
app = FastAPI()

@app.post("/predict_multi")
async def predict_multi(
    file: UploadFile = File(...),
    models: str = Form(...)  # Comma-separated model names
):
    """
    Accepts an image file and a comma-separated list of model names.
    Returns a ZIP file containing a JPEG for each model's predicted mask.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Unable to decode image"})

    selected_models = [m.strip() for m in models.split(",") if m.strip() in loaded_models]
    if not selected_models:
        return JSONResponse(status_code=400, content={"error": "No valid model names provided"})

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for model_name in selected_models:
            model = loaded_models[model_name]
            mask = predict_mask_image(image, model)
            success, buffer = cv2.imencode(".jpg", mask)
            if not success:
                continue
            mask_bytes = buffer.tobytes()
            zip_file.writestr(f"{model_name}.jpg", mask_bytes)

    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/x-zip-compressed", headers={
        "Content-Disposition": "attachment; filename=predicted_masks.zip"
    })

@app.post("/predict")
async def predict_single(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    """
    Accepts an image file and a single model name.
    Returns the predicted mask as a JPEG image.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Unable to decode image"})

    if model_name not in loaded_models:
        return JSONResponse(status_code=400, content={"error": f"Model '{model_name}' not found"})

    model = loaded_models[model_name]
    mask = predict_mask_image(image, model)
    success, buffer = cv2.imencode(".jpg", mask)
    if not success:
        return JSONResponse(status_code=500, content={"error": "Failed to encode mask image"})

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

# ---------------------------
# Gradio Interface (Blocks Layout)
# ---------------------------
def gradio_predict_multi(image: np.ndarray, selected_models: list) -> list:
    """
    For each selected model, predict the mask and annotate the mask with the model name.
    Since Gradio sends images in RGB format, we convert to BGR before processing
    and then convert the result back to RGB.
    """
    if image is None:
        return []
    # Convert RGB to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    outputs = []
    for model_name in selected_models:
        model = loaded_models.get(model_name)
        if model is None:
            continue
        mask = predict_mask_image(image_bgr, model)
        # Convert the grayscale mask to RGB
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        # Center the model name on the image
        text = model_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x_center = (mask_rgb.shape[1] - text_width) // 2
        y_position = text_height + 10  # 10 pixels from the top
        cv2.putText(mask_rgb, text, (x_center, y_position), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
        outputs.append(mask_rgb)
    return outputs

# Create a Blocks layout with custom CSS for full-width gallery
with gr.Blocks(css=".output-gallery { width: 100%; }", title="Polyps Segmentation App") as demo:
    gr.Markdown("# Polyps Segmentation")
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Input Image", type="numpy")
        with gr.Column(scale=1):
            model_select = gr.Dropdown(
                choices=list(loaded_models.keys()),
                label="Select Models",
                multiselect=True,
                value=list(loaded_models.keys())
            )
    predict_button = gr.Button("Predict")
    # The gallery will display the predicted masks in the lower half.
    output_gallery = gr.Gallery(label="Predicted Masks", elem_classes="output-gallery").style(grid=[3])
    predict_button.click(fn=gradio_predict_multi, inputs=[image_input, model_select], outputs=output_gallery)

# Mount the Gradio app on the FastAPI app at the "/gradio" path.
app = gr.mount_gradio_app(app, demo, path="/gradio")

# ---------------------------
# Run the Server
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



# The FastAPI endpoint at http://localhost:8000/predict
# The Gradio UI at http://localhost:8000/gradio