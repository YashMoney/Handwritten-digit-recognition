# Handwritten Digit Recognizer  MNIST

This project is an interactive web application built with Streamlit that allows users to recognize handwritten digits. Users can either upload an image of a digit or draw a digit directly on a canvas within the app. The recognition is performed by a Convolutional Neural Network (CNN) trained on the famous MNIST dataset.


## Features

*   **Dual Input Modes:**
    *   **Image Upload:** Upload JPG, PNG, or JPEG files containing a handwritten digit.
    *   **Drawable Canvas:** Draw a digit directly in the browser using an interactive canvas.
*   **Real-time Prediction:** The model predicts the digit as you draw or after an image is uploaded.
*   **Customizable Canvas:**
    *   Adjust stroke (digit) color.
    *   Adjust background color.
    *   Adjust stroke width for drawing.
*   **Intelligent Color Inversion:** Automatically inverts drawn digits if the background is lighter than the stroke, to match the MNIST model's expected input format (light digit on dark background).
*   **Detailed Feedback:**
    *   Displays the predicted digit and the model's confidence level.
    *   Shows a bar chart of prediction probabilities for all 10 digits (0-9).
    *   Option to view the 28x28 grayscale image that is fed into the model.
*   **User-Friendly Interface:** Clean layout with a sidebar for controls and information, and clear instructions.
*   **Powered by AI:** Uses a TensorFlow/Keras CNN model.



## Technical Stack

*   **Backend/ML Model:**
    *   Python
    *   TensorFlow / Keras (for building and training the CNN model)
    *   NumPy (for numerical operations)
    *   Pillow (PIL) (for image processing)
*   **Frontend/Web Application:**
    *   Streamlit (for creating the interactive web UI)
    *   `streamlit-drawable-canvas` (for the drawing canvas component)
    *   Pandas (for structuring data for charts)
*   **Dataset:**
    *   MNIST (for training the digit recognition model)


## Setup and Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone [your-repository-url]
    cd [repository-name]
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    ```
    Activate the environment:
    *   Windows: `.\.venv\Scripts\activate`
    *   macOS/Linux: `source .venv/bin/activate`

3.  **Install Dependencies:**
    Make sure you have `pip` installed. Then, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    Or, install them individually:
    ```bash
    pip install streamlit tensorflow Pillow numpy pandas streamlit-drawable-canvas
    ```
    *(Note: If `requirements.txt` is not provided, list the individual install command.)*

4.  **Download/Ensure Model File:**
    Make sure the pre-trained model file `mnist_cnn_model.h5` is present in the root directory of the project. If you have a `train_mnist_cnn.py` script, you can run it to generate the model file (this may take some time depending on your hardware).

## How to Run

1.  Ensure your virtual environment is activated (if you created one).
2.  Navigate to the project's root directory in your terminal.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  The application will open in your default web browser.

## Model Training (Optional)

If you wish to train the model yourself (or if `mnist_cnn_model.h5` is not provided):

1.  Ensure you have TensorFlow and other necessary libraries installed (`pip install tensorflow scikit-learn matplotlib`).
2.  Run the training script (e.g., `train_mnist_cnn.py` - *you'll need to provide this script based on your Kaggle notebook code*):
    ```bash
    python train_mnist_cnn.py
    ```
    This will generate the `mnist_cnn_model.h5` file.

The CNN architecture used typically includes:
*   Input Layer (28x28x1 for grayscale MNIST images)
*   Convolutional Layers (e.g., Conv2D with ReLU activation)
*   Max Pooling Layers (MaxPooling2D)
*   Flatten Layer
*   Dense (Fully Connected) Layers (with ReLU activation)
*   Output Layer (Dense with Softmax activation for 10 classes)

The model is compiled with 'sparse_categorical_crossentropy' loss and the 'Adam' optimizer.

## Future Enhancements

*   **Batch Prediction:** Allow uploading multiple images or a ZIP file for batch processing.
*   **Advanced Drawing Tools:** More sophisticated drawing tools (e.g., different brush shapes).
*   **Model Explainability:** Integrate LIME or SHAP to visualize what parts of the image the model focuses on.
*   **Alternative Models:** Option to switch between different trained models (e.g., a simpler MLP vs. CNN).
*   **Deployment:** Deploy the application to a cloud platform like Streamlit Community Cloud, Heroku, or AWS.
*   **Error Handling:** More robust error handling for diverse image inputs.
*   **Clear Canvas Button:** Implement a more robust "Clear Canvas" functionality that fully resets the canvas state.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Commit your changes (`git commit -m 'Add some YourFeature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if you add one). If not, you can just state "This project is open source."

---

**Note:**
*   Replace `[your-repository-url]` and `[repository-name]` if you host this on GitHub/GitLab.
*   Create a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated virtual environment *after* installing all necessary packages. This makes it easy for others to install dependencies.
*   If you have a separate training script (`train_mnist_cnn.py` from your Kaggle code), include it in the repository and reference it in the "Model Training" section.
*   Consider adding a `LICENSE` file (e.g., MIT License is common for open-source projects).
*   The screenshot part is important for a good README.

