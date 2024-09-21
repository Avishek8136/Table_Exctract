
# Omkara Extractor

## Overview

**Omkara Extractor** is a Streamlit-based web application designed for extracting textual data and tables from images and PDF files using Optical Character Recognition (OCR) and layout detection models. It leverages **PaddleOCR** for text extraction and **LayoutParser** for layout detection, specifically targeting tables within the documents. The extracted data can then be downloaded as an Excel file for further use.

## Features

- **Image and PDF Upload**: Supports uploading of images (PNG, JPG, JPEG) and PDF files.
- **OCR and Layout Detection**: Uses PaddleOCR for detecting and extracting text, and LayoutParser to identify the layout structure, including tables.
- **Table Extraction**: Automatically detects tables in the document and extracts the corresponding text.
- **Excel Export**: Exports the extracted text and table data to an Excel file for easy download.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/omkara-extractor.git
   ```

2. Navigate to the project directory:

   ```bash
   cd omkara-extractor
   ```

3. Create a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Upload an image (PNG, JPG, JPEG) or a PDF file for text and table extraction.

3. Once processed, you can view the extracted table in the app and download the results as an Excel file.

## File Structure

```
├── app.py               # Main Streamlit application
├── requirements.txt     # Dependencies for the project
├── README.md            # Project documentation
└── .gitignore           # Files and directories to be ignored by Git
```

## Dependencies

The main dependencies of the project include:

- **Streamlit**: For building the web interface.
- **PaddleOCR**: For performing Optical Character Recognition (OCR) on images and PDFs.
- **LayoutParser**: For detecting and extracting table layouts.
- **OpenCV**: For image processing.
- **TensorFlow**: For Non-Max Suppression used in text extraction.
- **pdf2image**: For converting PDF pages into images.

You can find the full list of dependencies in the `requirements.txt` file.


