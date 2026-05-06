# Assignment #4: Heartbeat to Heatmap

**DS-3002 Data Mining** – Spring 2026  
**FAST-NUCES, BSDS**  
**Student ID:** i232568

---

## Project Title

**Unsupervised Learning, Ensemble Methods, and Neural Networks on Heart Disease & Handwritten Digit Data**

---

## Dataset Download Instructions

### Heart Disease Dataset (UCI Cleveland)
The dataset is **not included** in this repository due to file size / policy.  
You can download it automatically by running the scripts, or manually:

**Manual download:**  
1. Visit [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)  
2. Download the `processed.cleveland.data` file  
3. Place it in the **project root folder** (same level as `run_all.py`)

**Automatic download (recommended):**  
The script `run_all.py` will download the dataset automatically if it is missing.

### MNIST Handwritten Digits
MNIST is loaded directly from `tensorflow.keras.datasets` – no manual download required.

---

## How to Run the Code

### 1. Set up the environment

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt