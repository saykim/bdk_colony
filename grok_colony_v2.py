import os
from datetime import datetime
from ultralytics import YOLO
import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import time

# AI 기반 분할 모델 로드 (FastSAM-x 사용)
model_path = 'weights/FastSAM-x.pt'
model = YOLO(model_path)

# 디바이스 설정 (CUDA, MPS, CPU 순으로 사용 가능)
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# 이미지 전처리 히스토리 관리 클래스
class ImagePreprocessHistory:
    def __init__(self):
        self.original_image = None
        self.current_image = None
        self.history = []
        self.current_index = -1

    def set_original(self, image):
        if image is None:
            return None
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        self.original_image = image
        self.current_image = image
        self.history = [image]
        self.current_index = 0
        return image

    def add_state(self, image):
        if image is None:
            return None
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        self.history = self.history[:self.current_index + 1]
        self.history.append(image)
        self.current_index = len(self.history) - 1
        self.current_image = image
        return image

    def reset(self):
        if self.original_image is not None:
            self.current_image = self.original_image
            self.current_index = 0
            return self.original_image
        return None

    def undo(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.current_image = self.history[self.current_index]
            return self.current_image
        return self.current_image

image_history = ImagePreprocessHistory()

# Colony 카운터 클래스 (자동/수동 구분 및 결과 이미지에 통합 숫자 표시)
class ColonyCounter:
    def __init__(self):
        self.manual_points = []  # 수동 포인트
        self.auto_points = []    # 자동 포인트
        self.current_image = None
        self.auto_detected_count = 0
        self.remove_mode = False
        self.original_image = None

    def reset(self):
        self.manual_points = []
        self.auto_points = []
        self.current_image = None
        self.auto_detected_count = 0
        self.remove_mode = False
        self.original_image = None

    def set_original_image(self, image):
        self.original_image = np.array(image)

    def toggle_remove_mode(self):
        self.remove_mode = not self.remove_mode
        img_with_points = self.draw_points()
        mode_text = "🔴 REMOVE MODE" if self.remove_mode else "🟢 ADD MODE"
        return img_with_points, mode_text

    def add_or_remove_point(self, image, evt: gr.SelectData):
        if self.current_image is None and image is not None:
            self.current_image = np.array(image)
        x, y = evt.index
        if self.remove_mode:
            closest_auto = self.find_closest_point(x, y, self.auto_points)
            closest_manual = self.find_closest_point(x, y, self.manual_points)
            if closest_auto is not None:
                self.auto_points.pop(closest_auto)
                self.auto_detected_count = len(self.auto_points)
            elif closest_manual is not None:
                self.manual_points.pop(closest_manual)
        else:
            self.manual_points.append((x, y))
        img_with_points = self.draw_points()
        return img_with_points, self.get_count_text()

    def find_closest_point(self, x, y, points, threshold=20):
        if not points:
            return None
        distances = [(np.sqrt((x - px) ** 2 + (y - py) ** 2), idx) for idx, (px, py) in enumerate(points)]
        closest_dist, closest_idx = min(distances, key=lambda item: item[0])
        return closest_idx if closest_dist < threshold else None

    def remove_last_point(self, image):
        if self.manual_points:
            self.manual_points.pop()
        img_with_points = self.draw_points()
        return img_with_points, self.get_count_text()

    def get_count_text(self):
        total_count = self.auto_detected_count + len(self.manual_points)
        return (f"전체 CFU 수: {total_count}\n"
                f"🤖 자동 감지된 CFU: {self.auto_detected_count}\n"
                f"👆 수동 추가된 CFU: {len(self.manual_points)}")

    def draw_points(self):
        if self.current_image is None:
            return None
        img_with_points = self.current_image.copy()
        overlay = np.zeros_like(img_with_points)
        square_size = 30

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 1
        outline_thickness = 3

        # 자동 감지된 포인트 표시 (녹색 원)
        for idx, (x, y) in enumerate(self.auto_points, 1):
            cv2.circle(img_with_points, (x, y), 5, (0, 255, 0), -1)
            text = str(idx)
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = int(x - text_width / 2)
            text_y = int(y - 10)
            
            # 검은색 외곽선 추가
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cv2.putText(img_with_points, text, 
                          (text_x + dx, text_y + dy), 
                          font, font_scale, (0, 0, 0), outline_thickness)
            
            # 파란색 텍스트
            cv2.putText(img_with_points, text, (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)

        # 수동 추가된 포인트 표시 (빨간 사각형)
        for idx, (x, y) in enumerate(self.manual_points, len(self.auto_points) + 1):
            pt1 = (int(x - square_size / 2), int(y - square_size / 2))
            pt2 = (int(x + square_size / 2), int(y + square_size / 2))
            cv2.rectangle(overlay, pt1, pt2, (0, 0, 255), -1)
            text = str(idx)
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = int(x - text_width / 2)
            text_y = int(y - 10)
            
            # 하얀색 외곽선 추가
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cv2.putText(img_with_points, text, 
                          (text_x + dx, text_y + dy), 
                          font, font_scale, (255, 255, 255), outline_thickness)
            
            # 파란색 텍스트
            cv2.putText(img_with_points, text, (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)

        cv2.addWeighted(overlay, 0.4, img_with_points, 1.0, 0, img_with_points)

        # 전체 카운트 표시 (왼쪽 하단)
        total_count = self.auto_detected_count + len(self.manual_points)
        count_text = f"Total: {total_count}"
        text_size = cv2.getTextSize(count_text, font, 1.5, 3)[0]
        margin = 20
        cv2.rectangle(img_with_points, 
                      (10, img_with_points.shape[0] - text_size[1] - margin * 2),
                      (text_size[0] + margin * 2, img_with_points.shape[0]),
                      (0, 0, 0), -1)
        cv2.putText(img_with_points, count_text, 
                    (margin, img_with_points.shape[0] - margin),
                    font, 1.5, (255, 255, 255), 3)

        return img_with_points

# 이미지 전처리 함수 (fastsam_prd_v4_ok에서 가져옴)
def preprocess_image(input_image, to_grayscale, binary_threshold, edge_detection, sharpen):
    if input_image is None:
        return None
    img = np.array(input_image)
    if to_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if binary_threshold > 0:
        _, img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), binary_threshold, 255, cv2.THRESH_BINARY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if edge_detection:
        img = cv2.Canny(img, 100, 200)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if sharpen > 0:
        kernel = np.array([[0, -sharpen, 0], [-sharpen, 1 + 4*sharpen, -sharpen], [0, -sharpen, 0]])
        img = cv2.filter2D(img, -1, kernel)
    return Image.fromarray(img)

# grok_colony_v1의 검출 로직 유지
def segment_and_count_colonies(
    input_image,
    conf_threshold=0.25,
    iou_threshold=0.7,
    circularity_threshold=0.8,
    draw_contours=True,
    mask_random_color=True,
    input_size=1024,
    better_quality=False,
    min_area_percentile=1,
    max_area_percentile=99,
    use_dish_filtering=False,  # 배양 접시 필터링 활성화 옵션
    dish_overlap_threshold=0.5,  # 배양 접시와 겹쳐야 하는 최소 비율
    progress=gr.Progress()
):
    if input_image is None:
        return None, "이미지를 선택해주세요."
    
    progress(0.1, desc="초기화 중...")
    counter.reset()
    counter.set_original_image(input_image)
    
    # 원본 이미지 크기 저장
    original_width, original_height = input_image.size
    
    # 이미지 크기 조정 (비율 유지)
    w, h = input_image.size
    scale = input_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    input_resized = input_image.resize((new_w, new_h))
    image_np = np.array(input_resized)

    progress(0.3, desc="AI 분석 중...")
    results = model.predict(
        image_np,
        device=device,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=input_size,
        retina_masks=True
    )

    if not results[0].masks:
        return image_np, "CFU가 감지되지 않았습니다."

    progress(0.6, desc="결과 처리 중...")
    processed_image = image_np.copy()
    counter.auto_points = []

    # 배양 접시 필터링 처리
    dish_mask = None
    dish_idx = -1
    if use_dish_filtering:
        progress(0.65, desc="배양 접시 감지 중...")
        # 가장 큰 마스크를 배양 접시로 식별
        annotations = results[0].masks.data
        if len(annotations) > 0:
            areas = [np.sum(ann.cpu().numpy()) for ann in annotations]
            dish_idx = np.argmax(areas)
            dish_annotation = annotations[dish_idx].cpu().numpy()
            dish_mask = dish_annotation > 0
            
            # 배양 접시 윤곽선 시각화 (파란색)
            dish_contours, _ = cv2.findContours(dish_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(processed_image, dish_contours, -1, (0, 0, 255), 2)

    # 마스크 면적 계산 및 필터링
    all_masks = []
    all_areas = []
    
    for idx, mask in enumerate(results[0].masks.data):
        mask_np = mask.cpu().numpy()
        
        # 배양 접시 필터링 적용
        if use_dish_filtering and dish_mask is not None:
            # 배양 접시인 경우 건너뛰기
            if idx == dish_idx:
                continue
                
            # 배양 접시 내부 여부 확인 (마스크 교집합)
            overlap_ratio = np.sum(mask_np * dish_mask) / np.sum(mask_np)
            if overlap_ratio < dish_overlap_threshold:
                continue  # 배양 접시와 충분히 겹치지 않는 경우 건너뛰기
        
        contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            all_areas.append(area)
            all_masks.append((mask_np, contour, area))
    
    # 면적 기준 필터링
    if all_areas:
        min_area = np.percentile(all_areas, min_area_percentile)
        max_area = np.percentile(all_areas, max_area_percentile)
    else:
        min_area = 0
        max_area = float('inf')
    
    # 향상된 품질 설정 적용
    if better_quality:
        # 더 정교한 윤곽선 그리기 (두께 조정)
        contour_thickness = 2
        # 더 선명한 색상 사용
        color_min = 120 if better_quality else 100
        color_max = 255 if better_quality else 200
    else:
        contour_thickness = 2
        color_min = 100
        color_max = 200
    
    # 필터링된 마스크 처리
    for mask_np, contour, area in all_masks:
        if min_area <= area <= max_area:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity >= circularity_threshold:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    counter.auto_points.append((cX, cY))
                    if draw_contours:
                        color = tuple(np.random.randint(color_min, color_max, 3).tolist()) if mask_random_color else (0, 255, 0)
                        cv2.drawContours(processed_image, [contour], -1, color, contour_thickness)

    counter.auto_detected_count = len(counter.auto_points)
    counter.current_image = processed_image

    progress(1.0, desc="완료!")
    img_with_points = counter.draw_points()
    
    # 결과 이미지를 원본 크기로 다시 조정
    img_with_points_pil = Image.fromarray(img_with_points)
    try:
        # PIL 9.0.0 이상
        resampling_filter = Image.Resampling.LANCZOS
    except AttributeError:
        # PIL 9.0.0 미만
        resampling_filter = Image.LANCZOS
    img_with_points_resized = img_with_points_pil.resize((original_width, original_height), resampling_filter)
    
    return np.array(img_with_points_resized), counter.get_count_text()

# 결과 저장 함수 (원본과 결과 이미지 저장)
def save_results(original_image, processed_image):
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_path = os.path.join(output_dir, f"original_{unique_id}.png")
    processed_path = os.path.join(output_dir, f"카운팅완료_{unique_id}.png")
    original_image.save(original_path)
    Image.fromarray(processed_image).save(processed_path)
    return f"저장 완료:\n- 원본: {original_path}\n- 결과: {processed_path}"

# UI 디자인 (fastsam_prd_v4_ok 기반)
css = """
body { font-family: Arial, sans-serif; background-color: #f0f0f0; }
.button-primary { 
    background-color: #4CAF50; 
    color: white; 
    border: none; 
    padding: 10px 20px; 
    border-radius: 5px; 
    transition: all 0.3s ease;
}
.button-primary:hover { 
    background-color: #45a049; 
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.button-secondary {
    background-color: #f1f1f1;
    color: #333;
    border: 1px solid #ddd;
    padding: 8px 16px;
    border-radius: 4px;
    transition: all 0.3s ease;
}
.button-secondary:hover {
    background-color: #e0e0e0;
    transform: translateY(-1px);
}
.result-text { 
    background: #fff; 
    border-left: 5px solid #4CAF50; 
    padding: 10px; 
    border-radius: 0 4px 4px 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.accordion-header {
    font-weight: bold;
    color: #333;
}
.section-title {
    font-size: 1.2em;
    font-weight: bold;
    color: #2E7D32;
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 2px solid #4CAF50;
}
.filtering-active {
    background-color: #e8f5e9;
    color: #2E7D32;
    padding: 3px 6px;
    border-radius: 4px;
    font-weight: bold;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { opacity: 0.7; }
    50% { opacity: 1; }
    100% { opacity: 0.7; }
}
"""

counter = ColonyCounter()

with gr.Blocks(theme=gr.themes.Default(), css=css) as demo:
    gr.Markdown("""
    # 🔬 Colony Counter
    ## AI 기반 CFU 자동 감지 및 수동 편집
    
    이 애플리케이션은 FastSAM AI 모델을 사용하여 배양접시의 CFU(Colony Forming Units)를 자동으로 감지하고 카운팅합니다.
    사용자는 자동 감지 결과를 수동으로 편집할 수도 있습니다.
    
    **새로운 기능**: 배양 접시 필터링 기능을 사용하면 가장 큰 마스크를 배양 접시로 식별하고, 접시 내부의 CFU만 카운팅합니다.
    이를 통해 배양 접시 외부의 노이즈나 오감지를 효과적으로 제거할 수 있습니다.
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("<div class='section-title'>📁 이미지 업로드</div>")
            input_image = gr.Image(type="pil", label="이미지 업로드")
            
            with gr.Accordion("🛠️ 이미지 전처리", open=False, elem_classes="accordion-header"):
                gr.Markdown("이미지를 분석하기 전에 전처리 옵션을 적용하여 결과를 개선할 수 있습니다.")
                to_grayscale = gr.Checkbox(label="흑백 변환", value=False, info="이미지를 흑백으로 변환합니다.")
                binary_threshold = gr.Slider(0, 255, 0, label="바이너리 임계값", info="0보다 크면 이진화를 적용합니다.")
                edge_detection = gr.Checkbox(label="에지 검출", value=False, info="이미지의 에지를 검출합니다.")
                sharpen = gr.Slider(0, 1, 0, label="샤픈 강도", info="이미지의 선명도를 높입니다.")
                
                with gr.Row():
                    preprocess_btn = gr.Button("전처리 적용", variant="secondary", elem_classes="button-secondary")
                    reset_btn = gr.Button("초기화", variant="secondary", elem_classes="button-secondary")
                    undo_btn = gr.Button("실행 취소", variant="secondary", elem_classes="button-secondary")
            
            # FastSAM 분석 설정 추가
            with gr.Accordion("⚙️ 분석 설정", open=True) as analysis_accordion:
                accordion_label = gr.Textbox(visible=False, value="⚙️ 분석 설정")
                
                with gr.Tab("일반"):
                    input_size_slider = gr.Slider(
                        512, 2048, 1024,
                        step=64,
                        label="입력 크기",
                        info="크기가 클수록 정확도가 높아지지만 속도는 느려집니다."
                    )
                    better_quality_checkbox = gr.Checkbox(
                        label="향상된 품질",
                        value=False,
                        info="속도를 희생하고 출력 품질을 향상시킵니다."
                    )
                    withContours_checkbox = gr.Checkbox(
                        label="윤곽선 표시",
                        value=True,
                        info="CFU의 경계를 표시합니다."
                    )
                    mask_random_color_checkbox = gr.Checkbox(
                        label="랜덤 색상",
                        value=True,
                        info="CFU 마스크에 랜덤 색상을 적용합니다."
                    )

                with gr.Tab("AI 탐지"):
                    iou_threshold_slider = gr.Slider(
                        0.1, 0.9, 0.7,
                        step=0.1,
                        label="IOU 임계값",
                        info="높을수록 겹침 기준이 엄격해집니다."
                    )
                    conf_threshold_slider = gr.Slider(
                        0.1, 0.9, 0.25,
                        step=0.05,
                        label="신뢰도 임계값",
                        info="높을수록 더 신뢰할 수 있는 탐지만 표시됩니다."
                    )
                    circularity_threshold_slider = gr.Slider(
                        0.0, 1.0, 0.8,
                        step=0.01,
                        label="원형도 임계값",
                        info="대략적으로 원형인 CFU만 감지합니다 (1 = 완벽한 원)."
                    )

                with gr.Tab("크기 필터"):
                    min_area_percentile_slider = gr.Slider(
                        0, 10, 1,
                        step=1,
                        label="최소 크기 백분위수",
                        info="더 작은 CFU를 필터링합니다 (1은 가장 작은 1% 필터링)."
                    )
                    max_area_percentile_slider = gr.Slider(
                        90, 100, 99,
                        step=1,
                        label="최대 크기 백분위수",
                        info="더 큰 객체를 필터링합니다 (99는 가장 큰 1% 필터링)."
                    )

                with gr.Tab("배양 접시 필터링"):
                    gr.Markdown("""
                    배양 접시 필터링은 이미지에서 가장 큰 마스크를 배양 접시로 식별하고, 해당 배양 접시 내부에 있는 CFU만 카운팅합니다.
                    이 기능은 다음과 같은 경우에 유용합니다:
                    - 배경에 노이즈가 많은 이미지
                    - 배양 접시 외부의 오탐지를 제거하고 싶을 때
                    - 여러 접시가 한 이미지에 있는 경우 가장 큰 접시만 분석하고 싶을 때
                    
                    활성화하면 배양 접시 윤곽선이 파란색으로 표시됩니다.
                    """)
                    use_dish_filtering_checkbox = gr.Checkbox(
                        label="배양 접시 필터링 사용",
                        value=True,
                        info="가장 큰 영역을 배양 접시로 인식하고 내부 CFU만 감지합니다."
                    )
                    dish_overlap_threshold_slider = gr.Slider(
                        0.1, 1.0, 0.5,
                        step=0.1,
                        label="배양 접시 겹침 임계값",
                        info="CFU가 배양 접시와 최소한 이 비율만큼 겹쳐야 합니다 (0.5 = 50%)."
                    )
                
            analyze_btn = gr.Button("🔍 분석 시작", variant="primary", elem_classes="button-primary")
        
        with gr.Column():
            gr.Markdown("<div class='section-title'>📊 분석 결과</div>")
            output_image = gr.Image(type="numpy", interactive=True, label="결과 이미지")
            count_text = gr.Textbox(label="카운트 결과", elem_classes="result-text")
            
            gr.Markdown("### 수동 편집")
            gr.Markdown("결과 이미지를 클릭하여 포인트를 추가하거나 제거할 수 있습니다.")
            
            with gr.Row():
                remove_mode_btn = gr.Button("🔄 모드 전환", variant="secondary", elem_classes="button-secondary")
                mode_text = gr.Textbox(value="🟢 ADD MODE", interactive=False)
                remove_last_btn = gr.Button("↩️ 최근 포인트 삭제", variant="secondary", elem_classes="button-secondary")
            
            save_btn = gr.Button("💾 결과 저장", variant="primary", elem_classes="button-primary")
            save_output = gr.Textbox(label="저장 결과", interactive=False, elem_classes="result-text")

    # 이벤트 핸들러
    input_image.upload(
        lambda img: (image_history.set_original(img), "이미지 업로드 완료"),
        inputs=input_image,
        outputs=[input_image, count_text]
    )

    preprocess_btn.click(
        preprocess_image,
        inputs=[input_image, to_grayscale, binary_threshold, edge_detection, sharpen],
        outputs=input_image
    ).then(
        lambda img: image_history.add_state(img),
        inputs=input_image,
        outputs=input_image
    )

    reset_btn.click(
        lambda: image_history.reset(),
        inputs=[],
        outputs=input_image
    )

    undo_btn.click(
        lambda: image_history.undo(),
        inputs=[],
        outputs=input_image
    )

    analyze_btn.click(
        segment_and_count_colonies,
        inputs=[
            input_image,
            conf_threshold_slider,
            iou_threshold_slider,
            circularity_threshold_slider,
            withContours_checkbox,
            mask_random_color_checkbox,
            input_size_slider,
            better_quality_checkbox,
            min_area_percentile_slider,
            max_area_percentile_slider,
            use_dish_filtering_checkbox,
            dish_overlap_threshold_slider
        ],
        outputs=[output_image, count_text]
    )

    output_image.select(
        counter.add_or_remove_point,
        inputs=[output_image],
        outputs=[output_image, count_text]
    )

    remove_mode_btn.click(
        counter.toggle_remove_mode,
        inputs=[],
        outputs=[output_image, mode_text]
    )

    remove_last_btn.click(
        counter.remove_last_point,
        inputs=[output_image],
        outputs=[output_image, count_text]
    )

    save_btn.click(
        save_results,
        inputs=[input_image, output_image],
        outputs=save_output
    )

if __name__ == "__main__":
    demo.launch()