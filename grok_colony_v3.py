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

# 타입 힌팅 오류 해결을 위한 코드 (실제 실행에는 영향 없음)
# pylint: disable=no-member
# mypy: ignore-errors

# AI 기반 분할 모델 로드
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

# 이미지 전처리 함수
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

# grok_colony_v1의 검출 로직
def segment_and_count_colonies(
    input_image,
    conf_threshold=0.25,
    iou_threshold=0.7,
    circularity_threshold=0.8,
    better_quality=False,
    min_area_percentile=1,
    max_area_percentile=99,
    use_dish_filtering=False,
    dish_overlap_threshold=0.5,
    progress=gr.Progress(),
    draw_contours=True,
    mask_random_color=True,
    input_size=1024
):
    """
    FastSAM 모델을 사용하여 이미지에서 콜로니를 감지하고 카운트하는 함수
    
    Args:
        input_image: 입력 이미지
        conf_threshold: 신뢰도 임계값 (0.0~1.0)
        iou_threshold: IOU 임계값 (0.0~1.0)
        circularity_threshold: 원형 임계값 (0.0~1.0)
        better_quality: 향상된 시각화 품질 사용 여부
        min_area_percentile: 최소 면적 퍼센타일 (0~100)
        max_area_percentile: 최대 면적 퍼센타일 (0~100)
        use_dish_filtering: 배양 접시 필터링 사용 여부
        dish_overlap_threshold: 배양 접시 겹침 임계값 (0.0~1.0)
        progress: gradio 진행 상태 표시 객체
        draw_contours: 윤곽선 그리기 여부
        mask_random_color: 마스크 랜덤 색상 사용 여부
        input_size: 입력 이미지 크기 조정 값
        
    Returns:
        tuple: (처리된 이미지, 카운트 결과 텍스트)
    """
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
        contour_thickness = 3
        # 더 선명한 색상 사용
        color_min = 120
        color_max = 255
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
    
    # PIL 버전에 따른 리샘플링 필터 설정
    try:
        resampling_filter = Image.Resampling.LANCZOS
    except AttributeError:
        # PIL 9.0.0 미만 버전 호환성
        try:
            resampling_filter = Image.LANCZOS
        except AttributeError:
            # 더 오래된 PIL 버전 호환성
            resampling_filter = Image.ANTIALIAS
    
    img_with_points_resized = img_with_points_pil.resize((original_width, original_height), resampling_filter)
    
    return np.array(img_with_points_resized), counter.get_count_text()

# 결과 저장 함수
def save_results(original_image, processed_image):
    """
    분석 결과를 파일로 저장하는 함수
    
    Args:
        original_image: 원본 이미지
        processed_image: 처리된 결과 이미지
        
    Returns:
        str: 저장 결과 메시지
    """
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_path = os.path.join(output_dir, f"original_{unique_id}.png")
    processed_path = os.path.join(output_dir, f"processed_{unique_id}.png")
    original_image.save(original_path)
    Image.fromarray(processed_image).save(processed_path)
    return f"저장 완료:\n- 원본: {original_path}\n- 결과: {processed_path}"

# UI 디자인 (fastsam_prd_v4_ok 기반)
css = """
body { font-family: Arial, sans-serif; background-color: #f0f0f0; }
.button-primary { background-color: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; }
.button-primary:hover { background-color: #45a049; }
.button-secondary { background-color: #808080; color: white; border: none; padding: 8px 16px; border-radius: 5px; }
.button-secondary:hover { background-color: #707070; }
.result-text { background: #fff; border-left: 5px solid #4CAF50; padding: 10px; }
.section-title { font-size: 18px; font-weight: bold; color: #4CAF50; margin-bottom: 10px; }
"""

counter = ColonyCounter()

with gr.Blocks(theme=gr.themes.Default(), css=css) as demo:
    gr.Markdown("# 🔬 Colony Counter\nAI 기반 CFU 자동 감지 및 수동 편집")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("<div class='section-title'>📁 이미지 업로드</div>")
            input_image = gr.Image(type="pil", label="이미지 업로드")
            
            with gr.Accordion("🛠️ 이미지 전처리", open=False):
                to_grayscale = gr.Checkbox(label="흑백 변환", value=False)
                binary_threshold = gr.Slider(0, 255, 0, label="바이너리 임계값")
                edge_detection = gr.Checkbox(label="에지 검출", value=False)
                sharpen = gr.Slider(0, 1, 0, label="샤픈 강도")
                preprocess_btn = gr.Button("전처리 적용", variant="secondary", elem_classes="button-secondary")
                
                with gr.Row():
                    reset_btn = gr.Button("초기화", variant="secondary", elem_classes="button-secondary")
                    undo_btn = gr.Button("실행 취소", variant="secondary", elem_classes="button-secondary")
            
            analysis_setting = gr.Accordion("⚙️ 분석 설정", open=False)
            
            with analysis_setting:
                with gr.Group():
                    gr.Markdown("### FastSAM 파라미터")
                    conf_threshold = gr.Slider(0.0, 1.0, 0.25, label="신뢰도 임계값", step=0.01)
                    iou_threshold = gr.Slider(0.0, 1.0, 0.7, label="IOU 임계값", step=0.01)
                    circularity_threshold = gr.Slider(0.0, 1.0, 0.8, label="원형 임계값", step=0.01)
                    better_quality = gr.Checkbox(label="향상된 시각화 품질", value=True)
                    
                with gr.Group():
                    gr.Markdown("### 필터링 옵션")
                    min_area_percentile = gr.Slider(0, 20, 1, label="최소 면적 퍼센타일")
                    max_area_percentile = gr.Slider(80, 100, 99, label="최대 면적 퍼센타일")
                    
                    use_dish_filtering_checkbox = gr.Checkbox(label="배양 접시 필터링 사용", value=True)
                    dish_overlap_threshold = gr.Slider(0.0, 1.0, 0.5, label="배양 접시 겹침 임계값", step=0.01)
            
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
            conf_threshold,
            iou_threshold,
            circularity_threshold,
            better_quality,
            min_area_percentile,
            max_area_percentile,
            use_dish_filtering_checkbox,
            dish_overlap_threshold
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
    
    # 배양 접시 필터링 제목 업데이트
    def update_title(is_enabled):
        if is_enabled:
            return "⚙️ 분석 설정 (배양접시 필터링 적용 중)"
        else:
            return "⚙️ 분석 설정"
    
    # 초기 UI 로딩 시 제목 업데이트 (배양접시 필터링이 기본값으로 True이므로)
    analysis_setting.label = update_title(True)
    
    use_dish_filtering_checkbox.change(
        update_title,
        inputs=[use_dish_filtering_checkbox],
        outputs=[analysis_setting]
    )

if __name__ == "__main__":
    demo.launch()