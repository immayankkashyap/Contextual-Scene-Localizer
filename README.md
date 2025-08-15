# Contextual Scene Localizer

This project is part of **AIMS 2K28 Recruitments** and aims to build a system that can **find and crop a specific part of a dense image** using a **natural language query**. For example, given *"a man selling vegetables"* and a busy scene, the model should return the cropped region showing that.

---

## Models Tried & Why I Chose Current Architecture

At first, I explored:

- **MDETR** – A multimodal detection model that jointly reasons with text and image. I kept hitting runtime and dependency errors that I couldn’t fix reliably.
- **MDETR + SAM** – Using MDETR for coarse localization and SAM for fine segmentation. Unfortunately, integration errors and version mismatches kept causing failures.
- **GroundingDINO** – A text-prompt–based open-set detector. It showed promise, but persistent bugs during dataset preprocessing and model loading prevented progress.

Because of these recurring technical issues, I switched to a simpler, custom **BERT + ResNet50** architecture. This choice made the pipeline easier to implement, debug, and control, with fewer external dependencies.

---

## Dataset Note

Initially, I was using the `Flickr30k Entities` dataset but switched to training on the `RefCOCO` dataset.  
The reason for switching was complications with importing and formatting the Flickr30k dataset. RefCOCO already provides images with bounding boxes linked to natural language descriptions, making it more suitable for this project.

---

## Approach Overview

1. **Text Understanding** – Encode the query using a BERT text encoder (`bert-base-uncased`).
2. **Image Understanding** – Extract image features using a ResNet50 backbone.
3. **Feature Fusion & Detection Head** – Merge text and image features to predict bounding box coordinates.
4. **Cropping Output** – Return the cropped image area corresponding to the query.

---

## Usage Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/immayankkashyap/Contextual-Scene-Localizer.git
cd Contextual-Scene-Localizer
```

### 2. Install Dependencies

Install using pip or your preferred method:

```bash
pip install torch transformers datasets opencv-python pycocotools albumentations
```

### 3. Run the Notebook

Open and run the Jupyter notebook:

```bash
jupyter notebook main.ipynb
```

Or convert to a script:

```bash
python main.ipynb
```

### 4. Predict on a New Image

Example pattern to follow:

```python
from your_module import predict_single_image

image_path = "path/to/image.jpg"
query = "a man selling vegetables"
cropped = predict_single_image(image_path, query)
cv2.imwrite("cropped.jpg", cropped)
```

---

## Results

After training and testing, I observed:

- For **simple queries** in less cluttered images, the model could detect and crop the correct region most of the time.
- In **medium-density scenes**, the model sometimes predicted slightly off-centre bounding boxes, but the target object/action was still visible in the crop.
- In **very dense scenes** (like markets or railway stations), performance dropped. The model sometimes confused similar objects or actions and returned the wrong region.
- Queries that were **clear and descriptive** (e.g., “front bowl with carrots in it”) performed better than vague ones (e.g., “bowl behind the others”).
- The model handled **objects** better than **complex actions** — recognizing a “bowl” was easier than “little girl playing”.
- Training on a limited dataset meant it didn’t generalize well to activities it hadn’t seen before.
- Inference speed was quite fast compared to the earlier heavy models I tried, making it usable even on smaller hardware.

Overall, while the accuracy isn’t perfect, the model works decently for a prototype and can be improved further with more training data and fine-tuning.

---

## Notebook Documentation

The detailed step-by-step documentation—including setup, methodology, errors faced, and blockers—is written inside the project’s notebook (`main.ipynb`). Sections include:
- Introduction
- Approaches Tried & Why I Switched
- Methodology
- Results
- Blockers
- References

---

## Blockers & Future Work

- Limited dataset coverage for very specific queries.
- Colab and Kaggle hardware limits slowed or interrupted long training runs.
- Bounding box accuracy drops in very dense or complex images.

**Future improvements might include:**
- Extending dataset or using data augmentation
- Adding fine-grained action recognition
- Exploring model ensembling
- Improving bounding box precision through multi-stage refinement

---

## References

- [RefCOCO dataset](https://github.com/lichengunc/refer)
- Hugging Face Transformers documentation
- PyTorch documentation
