# 🔬 Advanced Colony Counter

![Project Banner](banner_image_url)

## 📄 Overview

The **Advanced Colony Counter** is a web-based application designed for automated counting of bacterial colonies in Petri dishes. Leveraging the power of **YOLO** (You Only Look Once) for object detection and **Gradio** for an interactive user interface, this tool offers both automated detection and manual correction capabilities to ensure accurate and reliable colony counts.

## 🚀 Features

- **Automated Colony Detection**: Utilizes the YOLO model to detect and count bacterial colonies within uploaded images of Petri dishes.
- **Manual Correction**: Allows users to add or remove colony points manually to correct any discrepancies in automatic detection.
- **Petri Dish Recognition**: Identifies the Petri dish in the image, highlighting only its outline while excluding the entire color for clarity.
- **Interactive Interface**: Built with Gradio, providing an intuitive and user-friendly interface for seamless operation.
- **Customizable Analysis Settings**: Users can adjust various parameters such as input size, IOU threshold, confidence threshold, circularity threshold, and size percentiles to fine-tune the detection process.
- **Real-Time Visualization**: Displays the analysis results with clearly marked colony counts, distinguishing between automatically detected and manually added colonies.

## 🛠️ Installation

### Prerequisites

- **Python 3.7+**
- **CUDA-enabled GPU** (optional, for faster processing)
- **Pip** package manager

### Clone the Repository

```bash
git clone https://github.com/yourusername/advanced-colony-counter.git
cd advanced-colony-counter
