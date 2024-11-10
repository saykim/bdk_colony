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
```
# 🔬 고급 콜로니 카운터 (Advanced Colony Counter)

![프로젝트 배너](banner_image_url)

## 📄 개요

**고급 콜로니 카운터**는 페트리 접시 이미지에서 박테리아 콜로니를 자동으로 감지하고 카운트하는 웹 기반 애플리케이션입니다. **YOLO**(You Only Look Once) 객체 탐지 모델과 **Gradio**를 활용하여 인터랙티브한 사용자 인터페이스를 제공하며, 자동 감지와 수동 수정 기능을 통해 정확하고 신뢰할 수 있는 콜로니 카운팅을 지원합니다.

## 🚀 주요 기능

- **자동 콜로니 감지**: YOLO 모델을 사용하여 업로드된 페트리 접시 이미지 내 박테리아 콜로니를 자동으로 감지하고 카운트합니다.
- **수동 수정 기능**: 사용자가 콜로니 포인트를 직접 추가하거나 제거하여 자동 감지의 오류를 보정할 수 있습니다.
- **페트리 접시 인식**: 이미지 내 페트리 접시를 식별하여 외곽선만 표시하고 전체 색상은 제외함으로써 콜로니와 페트리 접시를 명확히 구분합니다.
- **인터랙티브 인터페이스**: Gradio를 사용하여 직관적이고 사용자 친화적인 웹 인터페이스를 제공합니다.
- **분석 설정 커스터마이징**: 입력 크기, IOU 임계값, 신뢰도 임계값, 원형도 임계값, 크기 백분위수 등의 다양한 파라미터를 조정하여 감지 과정을 미세 조정할 수 있습니다.
- **실시간 시각화**: 자동 감지된 콜로니와 수동으로 추가된 콜로니를 구분하여 명확하게 표시하고, 총 콜로니 수를 실시간으로 업데이트합니다.

## 🛠️ 설치 방법

### 필수 조건

- **Python 3.7 이상**
- **CUDA 지원 GPU** (선택 사항, 더 빠른 처리 속도를 위해)
- **Pip** 패키지 관리자

### 저장소 클론

```bash
git clone https://github.com/yourusername/advanced-colony-counter.git
cd advanced-colony-counter
```





