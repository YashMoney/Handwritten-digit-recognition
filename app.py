# app.py
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps, ImageColor # Added ImageColor
from streamlit_drawable_canvas import st_canvas
import pandas as pd # For displaying probabilities

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    layout="wide",
    page_title="Pro Digit Recognizer",
    page_icon="✏️" # You can use an emoji or a URL to an image
)

# --- Global Configuration ---
MODEL_PATH = 'mnist_cnn_model.h5'
IMAGE_SIZE = (28, 28)
CANVAS_SIZE = 280
STROKE_WIDTH = 20 # Default stroke width for drawing

# --- Model Loading (Cached) ---
@st.cache_resource
def load_keras_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Ensure 'mnist_cnn_model.h5' is in the app directory.")
        return None

model = load_keras_model(MODEL_PATH)

# --- Image Preprocessing ---
def preprocess_image(image_data, target_size):
    if isinstance(image_data, np.ndarray): # From canvas, already an array
        pil_image = Image.fromarray(image_data.astype('uint8'), 'RGBA').convert('L')
    elif isinstance(image_data, Image.Image): # From file upload (PIL Image)
        pil_image = image_data.convert('L') # Ensure it's grayscale
    else:
        st.error("Unsupported image data type for preprocessing.")
        return None

    pil_image_resized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(pil_image_resized) / 255.0
    return img_array.reshape(1, target_size[0], target_size[1], 1)

# --- UI Functions ---
def display_prediction_probabilities(probabilities, class_names):
    """Displays prediction probabilities as a bar chart."""
    prob_df = pd.DataFrame({
        'Digit': class_names,
        'Probability': probabilities.flatten() # Ensure it's a 1D array
    })
    prob_df = prob_df.sort_values(by='Probability', ascending=False)
    st.write("📊 **Prediction Probabilities:**")
    st.bar_chart(prob_df.set_index('Digit'), height=250)

# --- Main Application ---
st.markdown("<h1 style='text-align: center;'>✏️ Advanced Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Upload or draw a digit for AI-powered recognition.</p>", unsafe_allow_html=True)
st.markdown("---")


if model is None:
    st.warning("🚦 Model not loaded. Functionality will be limited. Please check console for errors.")
    st.stop()

# --- Sidebar for Controls and Information ---
with st.sidebar:
    st.header("⚙️ Controls & Info")
    app_mode = st.selectbox("Choose Input Mode", ["✍️ Draw Digit", "🖼️ Upload Image"])

    if app_mode == "✍️ Draw Digit":
        st.subheader("Canvas Settings")
        stroke_color = st.color_picker("Digit Color", "#FFFFFF") # White default
        bg_color = st.color_picker("Background Color", "#000000") # Black default
        # Allow stroke width to be adjusted. Key it to re-render canvas if changed.
        current_stroke_width = st.slider("Stroke Width", 5, 50, STROKE_WIDTH, key="stroke_width_slider")

    st.subheader("ℹ️ About")
    st.info(
        "This app uses a Convolutional Neural Network (CNN) "
        "trained on the MNIST dataset to recognize handwritten digits."
    )
    with st.expander("🤖 Model Details"):
        st.caption(f"Model: {MODEL_PATH}")
        st.caption(f"Input Shape: {model.input_shape}")
        st.caption(f"Output Shape: {model.output_shape}")
        # st.text(model.summary()) # Can be too verbose for sidebar

# --- Main Area based on Mode ---
if app_mode == "🖼️ Upload Image":
    st.header("🖼️ Upload Image Mode")
    uploaded_file = st.file_uploader(
        "Choose an image file (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Upload an image containing a single handwritten digit."
    )

    if uploaded_file is not None:
        col_img, col_pred = st.columns([1, 2]) # Adjusted column widths
        with col_img:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col_pred:
            with st.spinner("🧠 Analyzing uploaded image..."):
                # For uploaded images, assume user might upload black-on-white or white-on-black
                # A more robust approach would be to detect background or offer an invert toggle.
                # For now, we preprocess directly. The model expects light digit on dark bg.
                # If many uploads are black on white, consider adding an `ImageOps.invert()` step here.
                processed_image = preprocess_image(image, IMAGE_SIZE)


            if processed_image is not None:
                prediction = model.predict(processed_image)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)

                st.success(f"🔍 **Predicted Digit: {predicted_digit}** (Confidence: {confidence*100:.2f}%)")
                display_prediction_probabilities(prediction, [str(i) for i in range(10)])

                with st.expander("🔬 View Processed Image (28x28)"):
                    st.image(processed_image.reshape(IMAGE_SIZE), caption="Model Input", width=140)
            else:
                st.error("Failed to process uploaded image.")

elif app_mode == "✍️ Draw Digit":
    st.header("✍️ Draw Digit Mode")
    st.caption(f"Draw with '{stroke_color}' on a '{bg_color}' background. Stroke width: {current_stroke_width}px")

    col_canvas, col_results = st.columns([2,2]) # Canvas on left, results on right

    with col_canvas:
        # Use current_stroke_width from slider by referencing its key for dynamic update
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)", # Keep transparent fill
            stroke_width=current_stroke_width, # Use the slider value
            stroke_color=stroke_color,
            background_color=bg_color,
            height=CANVAS_SIZE + 20, # Slightly larger to accommodate thicker strokes
            width=CANVAS_SIZE + 20,
            drawing_mode="freedraw",
            key="digit_canvas_main", # Stable key for the canvas
            display_toolbar=True,
            update_streamlit=True # Important for live updates
        )
        # The canvas toolbar provides undo, redo, and eraser. A programmatic clear button
        # is harder to implement correctly with st_canvas without session_state and re-keying.

    with col_results:
        if canvas_result.image_data is not None and np.sum(canvas_result.image_data[:,:,:3]) > 5000: # Heuristic for drawing
            with st.spinner("✏️ Recognizing your masterpiece..."):
                drawn_image_data_np = canvas_result.image_data # This is an RGBA NumPy array
                pil_drawn_img = Image.fromarray(drawn_image_data_np.astype('uint8'), 'RGBA').convert('L')

                # Check if inversion is needed based on canvas colors.
                try:
                    bg_rgb = ImageColor.getrgb(bg_color)
                    stroke_rgb = ImageColor.getrgb(stroke_color)

                    avg_bg_intensity = np.mean(bg_rgb)
                    avg_stroke_intensity = np.mean(stroke_rgb)

                    if avg_bg_intensity > avg_stroke_intensity: # e.g., user drew black digit on white canvas
                        pil_drawn_img = ImageOps.invert(pil_drawn_img)
                        st.caption("Note: Colors auto-inverted to match model (light digit, dark background).")
                except ValueError as e:
                    st.warning(f"Could not parse color strings for inversion check: {e}. Using image as is.")

                processed_drawn_image = preprocess_image(pil_drawn_img, IMAGE_SIZE)


            if processed_drawn_image is not None:
                prediction_drawn = model.predict(processed_drawn_image)
                predicted_digit_drawn = np.argmax(prediction_drawn)
                confidence_drawn = np.max(prediction_drawn)

                st.success(f"💡 **Predicted Digit: {predicted_digit_drawn}** (Confidence: {confidence_drawn*100:.2f}%)")
                display_prediction_probabilities(prediction_drawn, [str(i) for i in range(10)])

                with st.expander("🔬 View Processed Drawing (28x28)"):
                    st.image(processed_drawn_image.reshape(IMAGE_SIZE), caption="Model Input", width=140)
            else:
                st.error("Could not process the drawn image.")
        elif canvas_result.image_data is not None:
            st.info("Keep drawing! The AI is waiting to see your digit.")
        else:
            st.info("Draw a digit on the canvas to the left.")


st.markdown("---")
st.markdown("<p style='text-align: center; font-size: small;'>Powered by Streamlit & TensorFlow/Keras</p>", unsafe_allow_html=True)