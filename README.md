
# website:https://fruitdetectionweb.streamlit.app/
# How to Run the Application

## 1. Clone the Repository
Clone the project to your local machine:
```bash
git clone <repository-url>
cd <repository-folder>
```

## 2. Set Up the Environment
Install the necessary libraries:
```bash
pip install -r requirements.txt
```

## 3. Prepare the YOLO Model
Download your YOLO model weights and save them in the specified path. Update the `model_path` in the code to point to the `.pt` file:
```python
model_path = 'path/to/your/best.pt'
```

## 4. Run the Application
Start the application with:
```bash
python app.py
```

---

# Using the Application

## Launch the Application
- A GUI window will open with a button to select an image.

## Select an Image
- Click the **"Select Image"** button and choose an image file (`.jpg`, `.jpeg`, `.png`).

## View Detection Results
- The image will be displayed with detected objects (fruits) highlighted.
- Detection details, including the count of each type of fruit, will be shown on the interface.

---

# Notes
- Ensure the image dimensions are appropriate for YOLO (preferably square, e.g., 640x640).
- Adjust confidence thresholds and other parameters in the code if needed.
