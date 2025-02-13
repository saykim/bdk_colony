import os
from datetime import datetime
from ultralytics import YOLO
import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import cv2

# AI 기반 분할 모델 로드
model_path = 'weights/AI_Segmentation_Model.pt'
model = YOLO(model_path)

# 디바이스 설정 (CUDA, MPS, CPU 순으로 사용 가능)
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)

class ColonyCounter:
    def __init__(self):
        self.manual_points = []  # 수동으로 추가된 포인트 리스트
        self.auto_points = []    # 자동으로 감지된 CFU 중심점 리스트
        self.current_image = None
        self.auto_detected_count = 0
        self.remove_mode = False
        self.original_image = None
        self.last_method = None
        self.zoom_factor = 1.0  # 확대/축소 비율

    def reset(self):
        """카운터 초기화"""
        self.manual_points = []
        self.auto_points = []  # 자동 포인트 초기화
        self.current_image = None
        self.auto_detected_count = 0
        self.remove_mode = False
        self.original_image = None
        self.last_method = None
        self.zoom_factor = 1.0  # 확대/축소 비율 초기화

    def set_original_image(self, image):
        """
        원본 이미지 설정

        Args:
            image (PIL.Image): 원본 이미지
        """
        self.original_image = np.array(image)

    def toggle_remove_mode(self):
        """편집 모드 전환 (추가/제거 모드)"""
        self.remove_mode = not self.remove_mode
        img_with_points = self.draw_points()
        mode_text = "🔴 제거 모드" if self.remove_mode else "🟢 추가 모드"
        return img_with_points, mode_text

    def add_or_remove_point(self, image, evt: gr.SelectData):
        """
        이미지를 클릭하여 포인트 추가 또는 제거

        Args:
            image (numpy.ndarray): 출력 이미지
            evt (gr.SelectData): 선택 데이터 이벤트

        Returns:
            tuple: 업데이트된 이미지와 카운트 텍스트
        """
        try:
            if self.current_image is None and image is not None:
                self.current_image = np.array(image)

            x, y = evt.index  # 클릭한 좌표

            if self.remove_mode:
                # 제거 모드: 자동 포인트 또는 수동 포인트에서 가장 가까운 포인트 제거
                closest_auto = self.find_closest_point(x, y, self.auto_points)
                closest_manual = self.find_closest_point(x, y, self.manual_points)
                removed = False

                if closest_auto is not None:
                    self.auto_points.pop(closest_auto)
                    self.auto_detected_count = len(self.auto_points)
                    removed = True
                elif closest_manual is not None:
                    self.manual_points.pop(closest_manual)
                    removed = True

                if not removed:
                    print("제거할 포인트가 충분히 가까이 있지 않습니다.")
            else:
                # 추가 모드: 새로운 포인트 추가
                self.manual_points.append((x, y))

            img_with_points = self.draw_points()
            return img_with_points, self.get_count_text()
        except Exception as e:
            print(f"포인트 추가/제거 중 오류 발생: {str(e)}")
            return image, self.get_count_text()

    def find_closest_point(self, x, y, points, threshold=20):
        """
        주어진 좌표와 가장 가까운 포인트를 찾습니다.

        Args:
            x (int): 클릭한 x 좌표
            y (int): 클릭한 y 좌표
            points (list): 포인트 리스트
            threshold (int, optional): 최대 거리 임계값. Defaults to 20.

        Returns:
            int or None: 가장 가까운 포인트의 인덱스 또는 None
        """
        if not points:
            return None

        distances = []
        for idx, (px, py) in enumerate(points):
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            distances.append((dist, idx))

        if not distances:
            return None

        closest_dist, closest_idx = min(distances, key=lambda item: item[0])
        return closest_idx if closest_dist < threshold else None

    def remove_last_point(self, image):
        """
        수동으로 추가된 마지막 포인트 제거

        Args:
            image (numpy.ndarray): 출력 이미지

        Returns:
            tuple: 업데이트된 이미지와 카운트 텍스트
        """
        try:
            if self.manual_points:
                self.manual_points.pop()
                img_with_points = self.draw_points()
                return img_with_points, self.get_count_text()
            return image, self.get_count_text()
        except Exception as e:
            print(f"마지막 포인트 제거 중 오류 발생: {str(e)}")
            return image, self.get_count_text()

    def get_count_text(self):
        """
        카운트 결과 텍스트 생성

        Returns:
            str: 카운트 결과 텍스트
        """
        try:
            method_text = f"방법: {self.last_method}\n" if self.last_method else ""
            return (f"{method_text}전체 CFU 수: {self.auto_detected_count + len(self.manual_points)}\n"
                    f"🤖 자동 감지된 CFU: {self.auto_detected_count}\n"
                    f"👆 수동으로 추가된 CFU: {len(self.manual_points)}")
        except Exception as e:
            print(f"카운트 텍스트 생성 중 오류 발생: {str(e)}")
            return "카운트 계산 오류"

    def draw_points(self):
        """
        현재 이미지에 포인트를 그려서 반환

        Returns:
            numpy.ndarray: 포인트가 그려진 이미지
        """
        try:
            if self.current_image is None:
                return None

            img_with_points = self.current_image.copy()
            overlay = np.zeros_like(img_with_points)
            square_size = 30  # 25에서 30으로 20% 증가

            # 글꼴 설정
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7 #0.7
            font_thickness = 1
            outline_thickness = 3

            # 자동 감지된 CFU 번호 표시 (파란색)
            for idx, (x, y) in enumerate(self.auto_points, 1):
                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

                text_x = int(x - text_width / 2)
                text_y = int(y - 10)

                # 검은색 외곽선
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    cv2.putText(img_with_points, text,
                                (text_x + dx, text_y + dy),
                                font, font_scale, (0, 0, 0), outline_thickness)

                # 파란색 텍스트
                cv2.putText(img_with_points, text,
                            (text_x, text_y),
                            font, font_scale, (255, 0, 0), font_thickness)  # 파란색 텍스트

            # 수동으로 추가된 포인트 사각형 그리기 (빨간색)
            for x, y in self.manual_points:
                pt1 = (int(x - square_size / 2), int(y - square_size / 2))
                pt2 = (int(x + square_size / 2), int(y + square_size / 2))
                cv2.rectangle(overlay, pt1, pt2, (0, 0, 255), -1)  # 빨간색 사각형

            # 오버레이 적용
            cv2.addWeighted(overlay, 0.4, img_with_points, 1.0, 0, img_with_points)

            # 수동 포인트 번호 표시 (파란색)
            for idx, (x, y) in enumerate(self.manual_points, len(self.auto_points) + 1):
                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = int(x - text_width / 2)
                text_y = int(y - 10)

                # 하얀색 외곽선
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    cv2.putText(img_with_points, text,
                                (text_x + dx, text_y + dy),
                                font, font_scale, (255, 255, 255), outline_thickness)

                # 파란색 텍스트
                cv2.putText(img_with_points, text,
                            (text_x, text_y),
                            font, font_scale, (255, 0, 0), font_thickness)  # 파란색 텍스트

            # 제거 모드 표시
            if self.remove_mode:
                overlay_mode = img_with_points.copy()
                cv2.rectangle(overlay_mode, (0, 0), (img_with_points.shape[1], 40), (255, 0, 0), -1)  # 빨간색 상자
                cv2.addWeighted(overlay_mode, 0.3, img_with_points, 0.7, 0, img_with_points)
                cv2.putText(img_with_points, "제거 모드", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 흰색 텍스트

            # Total Count 표시 추가
            total_count = self.auto_detected_count + len(self.manual_points)
            text = f'Total Count: {total_count}'
            
            # 이미지 크기 가져오기
            height, width = img_with_points.shape[:2]
            
            # 폰트 설정
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5 #0.7
            font_thickness = 2
            
            # 텍스트 크기 계산
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, font_thickness
            )
            
            # 텍스트 위치 계산 (우하단)
            #text_x = width - text_width - 20  # 우측에서 20픽셀 여백
            text_x = 20
            text_y = height - 20  # 하단에서 20픽셀 여백
            
            # 텍스트 배경 그리기 (가독성 향상)
            padding = 5
            cv2.rectangle(
                img_with_points,
                (text_x - padding, text_y - text_height - padding),
                (text_x + text_width + padding, text_y + padding),
                (0, 0, 0),  # 검정 배경
                -1
            )
            
            # 텍스트 그리기
            cv2.putText(
                img_with_points,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 255, 255),  # 노란색
                font_thickness
            )

            return img_with_points
        
        except Exception as e:
            print(f"포인트 그리기 중 오류 발생: {str(e)}")
            return self.current_image

# ColonyCounter 인스턴스 생성
counter = ColonyCounter()

def preprocess_image(
    input_image,
    to_grayscale=False,
    binary=False,
    binary_threshold=128,
    edge_detection=False,
    sharpen=False,
    sharpen_amount=1.0
):
    """
    이미지 전처리를 수행합니다.

    Args:
        input_image (PIL.Image): 전처리할 입력 이미지
        to_grayscale (bool, optional): 흑백 변환 여부. Defaults to False.
        binary (bool, optional): 바이너리 변환 여부. Defaults to False.
        binary_threshold (int, optional): 바이너리 변환 임계값. Defaults to 128.
        edge_detection (bool, optional): 에지 검출 여부. Defaults to False.
        sharpen (bool, optional): 샤픈 여부. Defaults to False.
        sharpen_amount (float, optional): 샤픈 강도. Defaults to 1.0.

    Returns:
        PIL.Image: 전처리된 이미지
    """
    try:
        image = input_image.copy()

        # 흑백 변환
        if to_grayscale:
            image = image.convert('L').convert('RGB')  # 흑백 후 RGB로 변환

        # 바이너리 변환
        if binary:
            image_np = np.array(image)
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            _, binary_img = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)
            image = Image.fromarray(binary_img).convert('RGB')

        # 에지 검출
        if edge_detection:
            image_np = np.array(image)
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            image = Image.fromarray(edges).convert('RGB')

        # 샤픈
        if sharpen:
            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))  # 기본 샤픈 설정
            if sharpen_amount != 1.0:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(sharpen_amount)

        return image
    except Exception as e:
        print(f"이미지 전처리 중 오류 발생: {str(e)}")
        return input_image

def process_image(colony_annotations, dish_annotation, image, device, scale, better_quality, mask_random_color, bbox, use_retina, withContours):
    """
    마스크 주석을 기반으로 이미지를 처리하여 CFU의 윤곽선을 그립니다.
    
    Args:
        colony_annotations (list): CFU의 마스크 주석 리스트
        dish_annotation (torch.Tensor): 배양접시의 마스크 주석
        image (PIL.Image): 원본 이미지
        device (torch.device): 디바이스 설정
        scale (float): 이미지 스케일
        better_quality (bool): 향상된 품질 여부
        mask_random_color (bool): 마스크에 랜덤 색상 적용 여부
        bbox (None): 바운딩 박스 (현재 사용되지 않음)
        use_retina (bool): Retina 모드 사용 여부 (현재 사용되지 않음)
        withContours (bool): 윤곽선 표시 여부

    Returns:
        PIL.Image: 처리된 이미지
    """
    image_np = np.array(image).copy()

    # CFU 마스크 처리
    for ann in colony_annotations:
        mask = ann.cpu().numpy()
        if mask.ndim == 2:
            mask = mask > 0
            if mask_random_color:
                color = np.random.randint(100, 200, (3,)).tolist()  # 중간 밝기의 랜덤 색상 생성
                image_np[mask] = color
            else:
                image_np[mask] = (0, 255, 0)  # 기본 녹색

        if withContours:
            # 윤곽선 찾기 및 그리기
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_np, contours, -1, (255, 0, 0), 2)  # 파란색 윤곽선

    # 배양접시 마스크 처리 (윤곽선만 그리기)
    if dish_annotation is not None:
        dish_mask = dish_annotation.cpu().numpy()
        if dish_mask.ndim == 2:
            dish_mask = dish_mask > 0
            if withContours:
                contours, _ = cv2.findContours(dish_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_np, contours, -1, (0, 0, 255), 3)  # 빨간색 윤곽선

    processed_image = Image.fromarray(image_np)
    return processed_image

def segment_and_count_colonies(
    preprocessed_image,
    method='AI Detection',
    input_size=1024,
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    mask_random_color=True,
    min_area_percentile=1,
    max_area_percentile=99,
    circularity_threshold=0.8,
    progress=gr.Progress()
):
    """
    이미지를 분석하고 CFU를 감지하여 카운팅합니다.

    Args:
        preprocessed_image (PIL.Image): 전처리된 이미지
        method (str, optional): 분석 방법. Defaults to 'AI Detection'.
        input_size (int, optional): 입력 이미지 크기. Defaults to 1024.
        iou_threshold (float, optional): IOU 임계값. Defaults to 0.7.
        conf_threshold (float, optional): 신뢰도 임계값. Defaults to 0.25.
        better_quality (bool, optional): 향상된 품질 여부. Defaults to False.
        withContours (bool, optional): 윤곽선 표시 여부. Defaults to True.
        mask_random_color (bool, optional): 마스크에 랜덤 색상 적용 여부. Defaults to True.
        min_area_percentile (int, optional): 최소 면적 백분위수. Defaults to 1.
        max_area_percentile (int, optional): 최대 면적 백분위수. Defaults to 99.
        circularity_threshold (float, optional): 원형도 임계값. Defaults to 0.8.
        progress (gr.Progress, optional): 진행 상황 표시. Defaults to gr.Progress().

    Returns:
        tuple: 분석된 이미지과 카운트 텍스트
    """
    try:
        if preprocessed_image is None:
            return None, "이미지를 선택해주세요."

        progress(0.1, desc="초기화 중...")
        counter.reset()
        counter.set_original_image(preprocessed_image)
        image_to_use = preprocessed_image

        # 이미지 크기 조정 (비율 유지)
        w, h = image_to_use.size
        scale = input_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        input_resized = image_to_use.resize((new_w, new_h))

        progress(0.3, desc="AI 분석 중...")
        # CFU 감지
        results = model.predict(
            input_resized,
            device=device,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=input_size
        )

        if not results[0].masks:
            return np.array(input_resized), "CFU가 감지되지 않았습니다."

        progress(0.6, desc="마스크 리사이즈 중...")
        # 마스크 리사이즈
        annotations = results[0].masks.data
        resized_annotations = []
        for ann in annotations:
            mask_np = ann.cpu().numpy().astype(np.uint8) * 255  # 이진 마스크로 변환
            mask_resized = cv2.resize(mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mask_resized = mask_resized > 0  # 다시 불리언으로 변환
            resized_annotations.append(torch.from_numpy(mask_resized))

        # 영역 계산
        areas = [np.sum(ann.numpy()) for ann in resized_annotations]

        # 가장 큰 마스크를 배양접시로 인식
        dish_idx = np.argmax(areas)
        dish_annotation = resized_annotations[dish_idx]
        colony_annotations = [ann for idx, ann in enumerate(resized_annotations) if idx != dish_idx]

        progress(0.8, desc="결과 처리 중...")
        # 결과 시각화
        fig = process_image(
            colony_annotations=colony_annotations,
            dish_annotation=dish_annotation,
            image=input_resized,
            device=device,
            scale=1.0,
            better_quality=better_quality,
            mask_random_color=mask_random_color,
            bbox=None,
            use_retina=False,
            withContours=withContours
        )

        counter.current_image = np.array(fig)
        counter.auto_detected_count = len(colony_annotations)

        # 자동 감지된 CFU의 중심점 계산
        counter.auto_points = []
        for ann in colony_annotations:
            mask = ann.numpy()
            if mask.ndim == 2:
                mask = mask > 0
                y_indices, x_indices = np.where(mask)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    counter.auto_points.append((center_x, center_y))

        progress(1.0, desc="완료!")
        img_with_points = counter.draw_points()
        return img_with_points, counter.get_count_text()
    except Exception as e:
        error_msg = f"분석 중 오류가 발생했습니다: {str(e)}"
        print(error_msg)
        return np.array(preprocessed_image), error_msg

def save_results(original_image, processed_image):
    """
    원본 이미지와 처리된 이미지를 /outputs 폴더에 저장합니다.

    Args:
        original_image (PIL.Image): 원본 이미지
        processed_image (numpy.ndarray): 처리된 이미지

    Returns:
        str: 저장된 파일 경로
    """
    try:
        # /outputs 폴더가 없으면 생성
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 고유한 식별자 생성
        unique_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"

        # 파일명 설정
        original_filename = f"original_{unique_id}.png"
        processed_filename = f"카운팅완료_{unique_id}.png"

        # 파일 경로 설정
        original_path = os.path.join(output_dir, original_filename)
        processed_path = os.path.join(output_dir, processed_filename)

        # 이미지 저장
        original_image.save(original_path)
        processed_image_pil = Image.fromarray(processed_image)
        processed_image_pil.save(processed_path)

        return f"저장된 파일:\n- 원본: {original_path}\n- 처리된 이미지: {processed_path}"
    except Exception as e:
        error_msg = f"이미지 저장 중 오류가 발생했습니다: {str(e)}"
        print(error_msg)
        return error_msg

# CSS 스타일링 개선
css = """
body {
    background-color: #f0f2f5;
}
.container {
    max-width: 1400px;
    margin: auto;
    padding: 20px;
}
.header {
    text-align: center;
    margin-bottom: 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    color: white;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.result-text {
    font-size: 1.2em;
    font-weight: bold;
    padding: 20px;
    background: #ffffff;
    border-radius: 10px;
    border-left: 5px solid #4a90e2;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.image-display {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    overflow: hidden;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.image-display:hover {
    border-color: #4a90e2;
}
.button-primary {
    background-color: #4a90e2;
    color: white;
}
.button-secondary {
    background-color: #f0f2f5;
    color: #333333;
}
.accordion-header {
    font-weight: bold;
    background-color: #4a90e2;
    color: white;
}
"""

# Gradio 인터페이스 설정
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown(
        """
        <div class="header">
            <h1>🔬 고급 CFU 카운터</h1>
            <h3>FastSAM을 이용한 자동 CFU 감지 및 수동 수정</h3>
        </div>
        """
    )

    with gr.Tab("CFU 카운터"):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                input_image = gr.Image(
                    label="이미지 업로드",
                    type="pil",
                    elem_classes="input-image"
                )

                # 이미지 전처리 설정
                with gr.Accordion("🛠️ 이미지 전처리 설정", open=False, elem_classes="accordion-header"):
                    to_grayscale = gr.Checkbox(
                        label="흑백 변환",
                        value=False,
                        info="이미지를 흑백으로 변환합니다."
                    )
                    binary = gr.Checkbox(
                        label="바이너리 변환",
                        value=False,
                        info="이미지를 바이너리(이진) 이미지로 변환합니다."
                    )
                    binary_threshold = gr.Slider(
                        0, 255, 128,
                        step=1,
                        label="바이너리 임계값",
                        info="바이너리 변환 시 사용할 임계값입니다."
                    )
                    edge_detection = gr.Checkbox(
                        label="에지 검출",
                        value=False,
                        info="이미지의 에지를 검출합니다."
                    )
                    sharpen = gr.Checkbox(
                        label="샤픈",
                        value=False,
                        info="이미지의 선명도를 높입니다."
                    )
                    sharpen_amount = gr.Slider(
                        0.5, 5.0, 1.0,
                        step=0.1,
                        label="샤픈 강도",
                        info="샤픈의 강도를 조절합니다."
                    )
                    preprocess_button = gr.Button(
                        "🔄 전처리 적용",
                        variant="secondary",
                        elem_classes="button-secondary"
                    )

                # Gradio의 State 컴포넌트 추가 (전처리된 이미지 저장)
                preprocessed_image_state = gr.State()

                with gr.Row():
                    method_select = gr.Radio(
                        choices=['AI Detection'],  # FastSAM 기반으로 가정
                        value='AI Detection',
                        label="탐지 방법",
                        info="AI 기반 탐지 방법을 선택하세요",
                        elem_id="method_select"
                    )
                    segment_button = gr.Button(
                        "🔍 이미지 분석",
                        variant="primary",
                        scale=2,
                        elem_classes="button-primary"
                    )

                # mask_random_color 정의를 이미지 전처리 설정 아래로 이동
                mask_random_color = gr.Checkbox(
                    label="마스크에 랜덤 색상 적용",
                    value=True,
                    info="감지된 CFU의 마스크에 랜덤 색상을 적용합니다."
                )

                with gr.Accordion("⚙️ 분석 설정", open=False, elem_classes="accordion-header"):
                    with gr.Tab("일반"):
                        input_size_slider = gr.Slider(
                            512, 2048, 1024,
                            step=64,
                            label="입력 크기",
                            info="크기가 클수록 정확도가 높아지지만 속도는 느려집니다."
                        )
                        better_quality_checkbox = gr.Checkbox(
                            label="향상된 품질",
                            value=True,
                            info="속도를 희생하고 출력 품질을 향상시킵니다."
                        )
                        withContours_checkbox = gr.Checkbox(
                            label="윤곽선 표시",
                            value=True,
                            info="CFU의 경계를 표시합니다."
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

                    with gr.Tab("형태 필터"):
                        circularity_threshold_slider = gr.Slider(
                            0.0, 1.0, 0.8,
                            step=0.01,
                            label="원형도 임계값",
                            info="대략적으로 원형인 CFU만 감지합니다 (1 = 완벽한 원)."
                        )

            with gr.Column(scale=1, min_width=300):
                output_image = gr.Image(
                    label="분석 결과",
                    type="numpy",
                    interactive=True,
                    elem_classes="output-image"
                )
                colony_count_text = gr.Textbox(
                    label="카운트 결과",
                    lines=3,
                    elem_classes="result-text"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        remove_mode_button = gr.Button(
                            "🔄 편집 모드 전환",
                            variant="secondary",
                            elem_classes="button-secondary"
                        )
                        remove_mode_text = gr.Textbox(
                            label="현재 모드",
                            value="🟢 추가 모드",
                            lines=1,
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        remove_point_button = gr.Button(
                            "↩️ 최근 포인트 취소",
                            variant="secondary",
                            elem_classes="button-secondary"
                        )

                # 결과 저장 기능 (옵션)
                save_button = gr.Button("💾 결과 저장", elem_classes="button-primary")
                save_output = gr.Textbox(
                    label="저장 결과",
                    lines=2,
                    interactive=False,
                    elem_classes="result-text"
                )

        with gr.Row(elem_classes="instruction-box"):
            gr.Markdown(
                """
                ### 📝 빠른 가이드
                1. **이미지 업로드**: 분석할 CFU 이미지를 업로드하세요.
                2. **이미지 전처리 설정**: 필요에 따라 흑백 변환, 바이너리 변환, 에지 검출, 샤픈 설정을 조절하세요.
                3. **전처리 적용**: "전처리 적용" 버튼을 클릭하여 설정한 전처리를 이미지에 적용하세요.
                4. **탐지 방법 선택**: 자동 분석을 위해 AI Detection을 선택하세요.
                5. **이미지 분석**: "이미지 분석" 버튼을 클릭하여 이미지를 처리하세요.
                6. **수동 수정**:
                    - 👆 이미지를 클릭하여 누락된 CFU를 추가하거나 자동으로 식별된 CFU를 제거하세요.
                    - 🔄 '편집 모드 전환' 버튼을 사용하여 추가/제거 모드를 전환하세요.
                    - ↩️ '최근 포인트 취소' 버튼을 사용하여 가장 최근에 추가된 포인트를 제거하세요.
                7. **분석 설정 조정**:
                    - **입력 크기**: 입력 이미지 크기를 설정하세요 (크기가 클수록 정확도가 증가하지만 처리 속도는 느려집니다).
                    - **IOU 임계값**: 겹치는 탐지에 대한 민감도를 조정하세요 (값이 높을수록 겹침 기준이 엄격해집니다).
                    - **신뢰도 임계값**: 탐지의 신뢰 수준을 설정하세요 (값이 높을수록 더 신뢰할 수 있는 탐지만 표시됩니다).
                    - **최소/최대 크기 백분위수**: 너무 작거나 큰 CFU를 제외하기 위해 크기 필터를 적용하세요.
                    - **원형도 임계값**: 대략적으로 원형인 CFU만 탐지하기 위해 원형도 필터를 설정하세요.

                **🔵 자동 감지된 CFU**는 파란색으로, **🔴 수동으로 추가된 CFU**는 빨간색으로 표시됩니다. 제거하려면 해당 포인트를 클릭하세요.
                """
            )

        # 전처리 적용 버튼 이벤트
        preprocess_button.click(
            preprocess_image,
            inputs=[
                input_image,
                to_grayscale,
                binary,
                binary_threshold,
                edge_detection,
                sharpen,
                sharpen_amount
            ],
            outputs=preprocessed_image_state
        )

        # 이미지 분석 버튼 이벤트 수정
        segment_button.click(
            fn=lambda orig, pre, method, input_size, iou, conf, bq, wc, mrc, minp, maxp, cir: segment_and_count_colonies(
                pre if pre is not None else orig, 
                method, 
                input_size, 
                iou, 
                conf, 
                bq, 
                wc, 
                mrc, 
                minp, 
                maxp, 
                cir
            ),
            inputs=[
                input_image,                   # original image
                preprocessed_image_state,     # preprocessed image (could be None)
                method_select,
                input_size_slider,
                iou_threshold_slider,
                conf_threshold_slider,
                better_quality_checkbox,
                withContours_checkbox,
                mask_random_color,
                min_area_percentile_slider,
                max_area_percentile_slider,
                circularity_threshold_slider
            ],
            outputs=[output_image, colony_count_text]
        )

        # 수동 및 자동 포인트 추가/제거 이벤트
        output_image.select(
            counter.add_or_remove_point,
            inputs=[output_image],
            outputs=[output_image, colony_count_text]
        )

        remove_point_button.click(
            counter.remove_last_point,
            inputs=[output_image],
            outputs=[output_image, colony_count_text]
        )

        remove_mode_button.click(
            counter.toggle_remove_mode,
            inputs=[],
            outputs=[output_image, remove_mode_text]
        )

        # 결과 저장 버튼 이벤트
        save_button.click(
            save_results,
            inputs=[input_image, output_image],
            outputs=save_output
        )

    # 앱 실행
    demo.launch()