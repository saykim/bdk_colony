import os
from datetime import datetime
from ultralytics import YOLO
import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Tuple

# AI 기반 분할 모델 로드
model_path = 'weights/FastSAM-x.pt'
model = YOLO(model_path)

# 디바이스 설정 (CUDA, MPS, CPU 순으로 사용 가능)
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)

class ImagePreprocessHistory:
    """이미지 전처리 히스토리 관리 클래스"""
    def __init__(self):
        self.original_image = None
        self.current_image = None
        self.history = []
        self.current_index = -1

    def set_original(self, image):
        """원본 이미지 설정"""
        if image is None:
            return None
            
        # PIL Image로 변환
        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except Exception as e:
                print(f"Error converting image: {str(e)}")
                return None
        
        self.original_image = image
        self.current_image = image
        self.history = [image]
        self.current_index = 0
        return image

    def add_state(self, image):
        """새로운 이미지 상태 추가"""
        if image is None:
            return None
            
        # PIL Image로 변환
        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except Exception as e:
                print(f"Error converting image: {str(e)}")
                return None
        
        # 현재 인덱스 이후의 히스토리는 삭제
        self.history = self.history[:self.current_index + 1]
        self.history.append(image)
        self.current_index = len(self.history) - 1
        self.current_image = image
        return image

    def reset(self):
        """원본 이미지로 복원"""
        if self.original_image is not None:
            self.current_image = self.original_image
            self.current_index = 0
            return self.original_image
        return None

    def undo(self):
        """이전 상태로 복원"""
        if self.current_index > 0:
            self.current_index -= 1
            self.current_image = self.history[self.current_index]
            return self.current_image
        return self.current_image

# 이미지 히스토리 객체 생성
image_history = ImagePreprocessHistory()

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
        mode_text = "🔴 REMOVE MODE" if self.remove_mode else "🟢 ADD MODE"
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
            font_scale = 0.7
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
                          font, font_scale, (255, 0, 0), font_thickness)

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
                          font, font_scale, (255, 0, 0), font_thickness)

            # 제거 모드 표시
            if self.remove_mode:
                overlay_mode = img_with_points.copy()
                cv2.rectangle(overlay_mode, (0, 0), (img_with_points.shape[1], 50), (255, 0, 0), -1)
                cv2.addWeighted(overlay_mode, 0.3, img_with_points, 0.7, 0, img_with_points)
                cv2.putText(img_with_points, "REMOVE MODE", (10, 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            else:
                overlay_mode = img_with_points.copy()
                cv2.rectangle(overlay_mode, (0, 0), (img_with_points.shape[1], 50), (0, 255, 0), -1)
                cv2.addWeighted(overlay_mode, 0.3, img_with_points, 0.7, 0, img_with_points)
                cv2.putText(img_with_points, "ADD MODE", (10, 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            # 전체 카운트 표시 (왼쪽 하단)
            total_count = self.auto_detected_count + len(self.manual_points)
            count_text = f"Total Count: {total_count}"
            
            # 배경 상자 그리기
            text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            margin = 20
            cv2.rectangle(img_with_points, 
                         (10, img_with_points.shape[0] - text_size[1] - margin * 2),
                         (text_size[0] + margin * 2, img_with_points.shape[0]),
                         (0, 0, 0), -1)
            
            # 카운트 텍스트 표시
            cv2.putText(img_with_points, count_text,
                      (margin, img_with_points.shape[0] - margin),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            return img_with_points
        except Exception as e:
            print(f"포인트 그리기 중 오류 발생: {str(e)}")
            return self.current_image

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
            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
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
    input_image,
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
        input_image (PIL.Image): 전처리된 이미지
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
        if input_image is None:
            return None, "이미지를 선택해주세요."

        progress(0.1, desc="초기화 중...")
        counter.reset()
        counter.set_original_image(input_image)
        image_to_use = input_image

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
            imgsz=input_size,
            retina_masks=True
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

def handle_batch_upload(
    files: List[str],
    input_size: int,
    iou_threshold: float,
    conf_threshold: float,
    better_quality: bool,
    withContours: bool,
    min_area_percentile: int,
    max_area_percentile: int,
    circularity_threshold: float,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """
    배치 업로드 처리를 담당하는 함수입니다.

    Args:
        files (List[str]): 업로드된 파일 경로 리스트
        ... (다른 매개변수들은 segment_and_count_colonies와 동일)

    Returns:
        Tuple[str, str]: 처리 결과 요약 메시지
    """
    if not files:
        return "처리된 이미지 없음", "파일을 선택해주세요."
    
    # 이미지 파일만 필터링
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    valid_images = [f for f in files if Path(f).suffix.lower() in image_extensions]
    
    if not valid_images:
        return "처리된 이미지 없음", "유효한 이미지 파일이 없습니다."
    
    # 출력 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("batch_outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 배치 처리 실행
    results = []
    summary_data = []
    total_images = len(valid_images)
    
    for idx, img_path in enumerate(valid_images):
        progress((idx + 1) / total_images, desc=f"이미지 처리 중 ({idx + 1}/{total_images}): {Path(img_path).name}")
        
        try:
            # 이미지 로드
            start_time = time.time()
            image = Image.open(img_path)
            
            # 이미지 분석
            processed_image, count_text = segment_and_count_colonies(
                image,
                input_size=input_size,
                iou_threshold=iou_threshold,
                conf_threshold=conf_threshold,
                better_quality=better_quality,
                withContours=withContours,
                mask_random_color=True,
                min_area_percentile=min_area_percentile,
                max_area_percentile=max_area_percentile,
                circularity_threshold=circularity_threshold
            )
            
            # 결과 저장
            filename = Path(img_path).stem
            processed_filename = f"{filename}_processed.png"
            processed_path = os.path.join(output_dir, processed_filename)
            
            # 처리된 이미지 저장
            processed_image_pil = Image.fromarray(processed_image)
            processed_image_pil.save(processed_path)
            
            process_time = time.time() - start_time
            
            # 요약 데이터 수집
            summary_data.append({
                'filename': Path(img_path).name,
                'total_count': counter.auto_detected_count + len(counter.manual_points),
                'auto_count': counter.auto_detected_count,
                'manual_count': len(counter.manual_points),
                'process_time': f"{process_time:.2f}초",
                'output_path': processed_path,
                'status': 'success'
            })
            
        except Exception as e:
            error_msg = f"이미지 처리 중 오류 발생: {str(e)}"
            print(f"Error processing {img_path}: {error_msg}")
            summary_data.append({
                'filename': Path(img_path).name,
                'total_count': 0,
                'auto_count': 0,
                'manual_count': 0,
                'process_time': '0초',
                'status': 'error',
                'error_message': error_msg
            })
    
    # 요약 데이터프레임 생성
    summary_df = pd.DataFrame(summary_data)
    
    # CSV 파일로 저장
    csv_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 처리 결과 요약 생성
    successful = summary_df['status'].value_counts().get('success', 0)
    failed = summary_df['status'].value_counts().get('error', 0)
    
    # 평균 처리 시간 계산 (성공한 케이스만)
    successful_times = [float(t.replace('초', '')) for t in summary_df[summary_df['status'] == 'success']['process_time']]
    avg_time = sum(successful_times) / len(successful_times) if successful_times else 0
    
    summary_message = f"""
    처리 완료:
    - 총 이미지 수: {total_images}
    - 성공: {successful}
    - 실패: {failed}
    - 평균 처리 시간: {avg_time:.2f}초
    
    처리 결과는 다음 경로에 저장되었습니다:
    {output_dir}
    
    상세 결과는 summary.csv 파일을 확인해주세요.
    """
    
    # 처리된 이미지 경로 목록 생성
    processed_images = [row['output_path'] for row in summary_data if row['status'] == 'success']
    
    return processed_images, summary_message

# CSS 스타일링 개선
css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

body {
    background-color: #f8fafc;
    font-family: 'Noto Sans KR', sans-serif;
    color: #334155;
    line-height: 1.4;
}

.container {
    max-width: 1280px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    background: linear-gradient(135deg, #2563EB, #1e40af);
    padding: 30px;
    border-radius: 12px;
    color: #ffffff;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-content {
    text-align: left;
}

.header-date {
    text-align: right;
    color: rgba(255, 255, 255, 0.9);
    font-size: 14px;
}

.header h1 {
    font-size: 32px;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, #ffffff, #e2e8f0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.header h3 {
    font-size: 16px;
    font-weight: 500;
    margin: 10px 0 0;
    color: #e2e8f0;
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #1e40af;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.button-primary {
    background-color: #2563EB !important;
    color: #ffffff !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
}

.button-primary:hover {
    background-color: #1d4ed8 !important;
    box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3) !important;
    transform: translateY(-1px) !important;
}

.button-secondary {
    background-color: #475569 !important;
    color: #ffffff !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 4px rgba(71, 85, 105, 0.2) !important;
}

.button-secondary:hover {
    background-color: #334155 !important;
    box-shadow: 0 4px 6px rgba(71, 85, 105, 0.3) !important;
    transform: translateY(-1px) !important;
}

.accordion-header {
    background-color: #f1f5f9 !important;
    color: #1e40af !important;
    padding: 15px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    border: 1px solid #e2e8f0 !important;
    margin: 10px 0 !important;
    transition: all 0.3s ease !important;
}

.accordion-header:hover {
    border-color: #2563EB !important;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.1) !important;
}

.input-image, .output-image {
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    background: #ffffff !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    margin: 15px 0 !important;
}

.input-image:hover, .output-image:hover {
    border-color: #2563EB !important;
    box-shadow: 0 4px 6px rgba(37, 99, 235, 0.1) !important;
}

.result-text {
    background: #ffffff !important;
    border-left: 5px solid #10b981 !important;
    padding: 15px !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    margin: 15px 0 !important;
    color: #334155 !important;
}

.gradio-container {
    background-color: #ffffff !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    padding: 20px !important;
    margin: 20px auto !important;
}

.gradio-row {
    gap: 20px !important;
}

.gradio-column {
    background: #ffffff !important;
    padding: 20px !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}

.slider-container {
    background: #f1f5f9 !important;
    padding: 15px !important;
    border-radius: 8px !important;
    margin: 10px 0 !important;
}

.slider {
    accent-color: #2563EB !important;
}

.checkbox-container {
    background: #f1f5f9 !important;
    padding: 12px !important;
    border-radius: 8px !important;
    margin: 8px 0 !important;
}

.checkbox {
    accent-color: #10b981 !important;
}

.instruction-box {
    background-color: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin-top: 30px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}

.instruction-box h3 {
    color: #1e40af !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    margin-bottom: 15px !important;
}

.instruction-box p {
    color: #334155 !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    margin-bottom: 10px !important;
}

/* 탭 스타일링 */
.tabs {
    border-bottom: 1px solid #e2e8f0 !important;
    margin-bottom: 20px !important;
}

.tab-nav {
    padding: 12px 24px !important;
    color: #475569 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: all 0.3s ease !important;
}

.tab-nav:hover {
    color: #2563EB !important;
}

.tab-nav.selected {
    color: #2563EB !important;
    border-bottom: 2px solid #2563EB !important;
    background-color: rgba(37, 99, 235, 0.05) !important;
}

/* 중요도 표시 */
.priority-high {
    color: #ef4444 !important;
    font-weight: 500 !important;
}

.priority-medium {
    color: #f59e0b !important;
    font-weight: 500 !important;
}

.priority-low {
    color: #10b981 !important;
    font-weight: 500 !important;
}

/* 카드 컴포넌트 */
.card {
    background: #ffffff !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    padding: 20px !important;
    margin: 15px 0 !important;
    border: 1px solid #e2e8f0 !important;
    transition: all 0.3s ease !important;
}

.card:hover {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    transform: translateY(-2px) !important;
}

/* 스크롤바 커스터마이징 */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}
"""

# 전역 counter 객체
counter = ColonyCounter()

# Gradio 인터페이스 설정
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown(
        """
        <div class="header">
            <div class="header-content">
                <h1>🔬 BDK CFU 카운터</h1>
                <h3>AI 자동 CFU 감지</h3>
            </div>
            <div class="header-date">
                <span>최종 업데이트: 2023년 12월</span>
            </div>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("단일 이미지 처리"):
            with gr.Row():
                with gr.Column(scale=6, min_width=300):
                    gr.Markdown("<div class='section-title'>📁 이미지 업로드</div>")
                    input_image = gr.Image(
                        type="pil",
                        elem_classes=["input-image"],
                        show_label=False
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
                        with gr.Row():
                            preprocess_button = gr.Button(
                                "🔄 전처리 적용",
                                variant="secondary",
                                elem_classes="button-secondary"
                            )
                            reset_button = gr.Button(
                                "↺ 초기화",
                                variant="secondary",
                                elem_classes="button-secondary"
                            )
                            undo_button = gr.Button(
                                "↶ 실행 취소",
                                variant="secondary",
                                elem_classes="button-secondary"
                            )

                    with gr.Row():
                        segment_button = gr.Button(
                            "🔍 이미지 분석",
                            variant="primary",
                            scale=2,
                            elem_classes="button-primary"
                        )

                    # 분석 설정
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

                        circularity_threshold_slider = gr.Slider(
                            0.0, 1.0, 0.8,
                            step=0.01,
                            label="원형도 임계값",
                            info="대략적으로 원형인 CFU만 감지합니다 (1 = 완벽한 원)."
                        )

                with gr.Column(scale=4, min_width=300):
                    gr.Markdown("<div class='section-title'>📊 분석 결과</div>")
                    output_image = gr.Image(
                        type="numpy",
                        interactive=True,
                        elem_classes=["output-image"],
                        show_label=False
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
                                value="🟢 ADD MODE",
                                lines=1,
                                interactive=False
                            )
                        with gr.Column(scale=1):
                            remove_point_button = gr.Button(
                                "↩️ 최근 포인트 취소",
                                variant="secondary",
                                elem_classes="button-secondary"
                            )

                    # 결과 저장 기능
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
                    <h3>📝 빠른 가이드</h3>
                    <p><span class="priority-high">1. 이미지 업로드:</span> 분석할 CFU 이미지를 업로드하세요.</p>
                    <p><span class="priority-medium">2. 이미지 전처리 설정:</span> 필요에 따라 흑백 변환, 바이너리 변환, 에지 검출, 샤픈 설정을 조절하세요.</p>
                    <p><span class="priority-medium">3. 전처리 적용:</span> "전처리 적용" 버튼을 클릭하여 설정한 전처리를 이미지에 적용하세요.</p>
                    <p><span class="priority-high">4. 이미지 분석:</span> "이미지 분석" 버튼을 클릭하여 이미지를 처리하세요.</p>
                    <p><span class="priority-medium">5. 수동 수정:</span></p>
                    <ul>
                        <li>👆 이미지를 클릭하여 누락된 CFU를 추가하거나 자동으로 식별된 CFU를 제거하세요.</li>
                        <li>🔄 '편집 모드 전환' 버튼을 사용하여 추가/제거 모드를 전환하세요.</li>
                        <li>↩️ '최근 포인트 취소' 버튼을 사용하여 가장 최근에 추가된 포인트를 제거하세요.</li>
                    </ul>
                    <p><span class="priority-low">6. 분석 설정 조정:</span></p>
                    <ul>
                        <li>입력 크기: 입력 이미지 크기를 설정하세요 (크기가 클수록 정확도가 증가하지만 처리 속도는 느려집니다).</li>
                        <li>IOU 임계값: 겹치는 탐지에 대한 민감도를 조정하세요 (값이 높을수록 겹침 기준이 엄격해집니다).</li>
                        <li>신뢰도 임계값: 탐지의 신뢰 수준을 설정하세요 (값이 높을수록 더 신뢰할 수 있는 탐지만 표시됩니다).</li>
                        <li>최소/최대 크기 백분위수: 너무 작거나 큰 CFU를 제외하기 위해 크기 필터를 적용하세요.</li>
                        <li>원형도 임계값: 대략적으로 원형인 CFU만 탐지하기 위해 원형도 필터를 설정하세요.</li>
                    </ul>
                    """
                )

            # 이벤트 핸들러 연결
            def handle_image_upload(image):
                """이미지 업로드 처리"""
                if image is None:
                    return None, None
                
                try:
                    # PIL Image로 변환
                    if not isinstance(image, Image.Image):
                        image = Image.fromarray(image)
                    
                    # 이미지 크기 제한
                    max_size = 1024
                    w, h = image.size
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        image = image.resize((new_w, new_h), Image.LANCZOS)
                    
                    # 이미지 히스토리에 저장
                    processed_image = image_history.set_original(image)
                    return processed_image, "이미지가 업로드되었습니다."
                except Exception as e:
                    print(f"Error in handle_image_upload: {str(e)}")
                    return None, f"이미지 업로드 중 오류 발생: {str(e)}"

            def apply_preprocessing(image, to_grayscale, binary, binary_threshold, edge_detection, sharpen, sharpen_amount):
                """이미지 전처리 적용"""
                if image is None:
                    return None
                
                try:
                    processed = preprocess_image(
                        image,
                        to_grayscale,
                        binary,
                        binary_threshold,
                        edge_detection,
                        sharpen,
                        sharpen_amount
                    )
                    if processed is not None:
                        return image_history.add_state(processed)
                    return image
                except Exception as e:
                    print(f"Error in apply_preprocessing: {str(e)}")
                    return image

            def reset_image():
                """이미지 초기화"""
                try:
                    return image_history.reset()
                except Exception as e:
                    print(f"Error in reset_image: {str(e)}")
                    return None

            def undo_image():
                """이전 상태로 복원"""
                try:
                    return image_history.undo()
                except Exception as e:
                    print(f"Error in undo_image: {str(e)}")
                    return None

            # 이벤트 핸들러 연결
            input_image.upload(
                handle_image_upload,
                inputs=[input_image],
                outputs=[input_image, colony_count_text]
            )

            preprocess_button.click(
                apply_preprocessing,
                inputs=[
                    input_image,
                    to_grayscale,
                    binary,
                    binary_threshold,
                    edge_detection,
                    sharpen,
                    sharpen_amount
                ],
                outputs=[input_image]
            )

            reset_button.click(
                reset_image,
                inputs=[],
                outputs=[input_image]
            )

            undo_button.click(
                undo_image,
                inputs=[],
                outputs=[input_image]
            )

            segment_button.click(
                segment_and_count_colonies,
                inputs=[
                    input_image,
                    input_size_slider,
                    iou_threshold_slider,
                    conf_threshold_slider,
                    better_quality_checkbox,
                    withContours_checkbox,
                    min_area_percentile_slider,
                    max_area_percentile_slider,
                    circularity_threshold_slider
                ],
                outputs=[output_image, colony_count_text]
            )

            # 수동 포인트 추가/제거 이벤트
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

            # 결과 저장 이벤트
            save_button.click(
                save_results,
                inputs=[input_image, output_image],
                outputs=[save_output]
            )

        with gr.Tab("배치 처리"):
            with gr.Row():
                with gr.Column(scale=6, min_width=300):
                    # 파일 업로드 컴포넌트
                    gr.Markdown("<div class='section-title'>📁 배치 이미지 업로드</div>")
                    batch_files = gr.File(
                        label="이미지 파일 선택 (여러 개 선택 가능)",
                        file_count="multiple",
                        file_types=["image"],
                        elem_classes=["card"]
                    )

                    # 배치 처리 설정
                    with gr.Accordion("⚙️ 배치 처리 설정", open=True, elem_classes="accordion-header"):
                        with gr.Row():
                            batch_input_size = gr.Slider(
                                512, 2048, 1024,
                                step=64,
                                label="입력 크기",
                                info="크기가 클수록 정확도가 높아지지만 속도는 느려집니다."
                            )
                            batch_better_quality = gr.Checkbox(
                                label="향상된 품질",
                                value=True,
                                info="속도를 희생하고 출력 품질을 향상시킵니다."
                            )

                        with gr.Row():
                            batch_iou_threshold = gr.Slider(
                                0.1, 0.9, 0.7,
                                step=0.1,
                                label="IOU 임계값",
                                info="높을수록 겹침 기준이 엄격해집니다."
                            )
                            batch_conf_threshold = gr.Slider(
                                0.1, 0.9, 0.25,
                                step=0.05,
                                label="신뢰도 임계값",
                                info="높을수록 더 신뢰할 수 있는 탐지만 표시됩니다."
                            )

                        with gr.Row():
                            batch_min_area = gr.Slider(
                                0, 10, 1,
                                step=1,
                                label="최소 크기 백분위수",
                                info="더 작은 CFU를 필터링합니다."
                            )
                            batch_max_area = gr.Slider(
                                90, 100, 99,
                                step=1,
                                label="최대 크기 백분위수",
                                info="더 큰 객체를 필터링합니다."
                            )

                        batch_circularity = gr.Slider(
                            0.0, 1.0, 0.8,
                            step=0.01,
                            label="원형도 임계값",
                            info="대략적으로 원형인 CFU만 감지합니다."
                        )

                    # 배치 처리 시작 버튼
                    with gr.Row():
                        batch_process_button = gr.Button(
                            "🔍 배치 처리 시작",
                            variant="primary",
                            scale=2,
                            elem_classes="button-primary"
                        )

                with gr.Column(scale=4, min_width=300):
                    # 처리 결과 요약
                    gr.Markdown("<div class='section-title'>📊 처리 결과</div>")
                    batch_summary = gr.Textbox(
                        label="처리 결과 요약",
                        lines=10,
                        elem_classes="result-text"
                    )
                    
                    # 결과 폴더 열기 버튼
                    open_output_button = gr.Button(
                        "📂 결과 폴더 열기",
                        variant="secondary",
                        elem_classes="button-secondary"
                    )
            
            with gr.Row(elem_classes="instruction-box"):
                gr.Markdown(
                    """
                    <h3>📝 배치 처리 가이드</h3>
                    <p><span class="priority-high">1. 이미지 업로드:</span> 분석할 여러 CFU 이미지 파일을 선택하세요.</p>
                    <p><span class="priority-medium">2. 배치 처리 설정:</span> 필요에 따라 분석 매개변수를 조정하세요.</p>
                    <p><span class="priority-high">3. 배치 처리 시작:</span> "배치 처리 시작" 버튼을 클릭하여 모든 이미지를 처리하세요.</p>
                    <p><span class="priority-medium">4. 결과 확인:</span> 처리가 완료되면 결과 요약을 확인하고 "결과 폴더 열기" 버튼을 클릭하여 저장된 파일을 확인하세요.</p>
                    <p><span class="priority-low">5. 참고 사항:</span></p>
                    <ul>
                        <li>처리된 이미지는 batch_outputs 폴더에 저장됩니다.</li>
                        <li>각 배치 처리는 타임스탬프가 있는 하위 폴더에 저장됩니다.</li>
                        <li>summary.csv 파일에서 모든 이미지의 처리 결과를 확인할 수 있습니다.</li>
                    </ul>
                    """
                )

            # 배치 처리 이벤트 핸들러
            def open_output_folder():
                import webbrowser
                output_dir = os.path.abspath("batch_outputs")
                webbrowser.open(f"file://{output_dir}")
                return "결과 폴더를 열었습니다."

            batch_process_button.click(
                handle_batch_upload,
                inputs=[
                    batch_files,
                    batch_input_size,
                    batch_iou_threshold,
                    batch_conf_threshold,
                    batch_better_quality,
                    gr.Checkbox(value=True, visible=False),  # withContours
                    batch_min_area,
                    batch_max_area,
                    batch_circularity
                ],
                outputs=[batch_summary]
            )

            open_output_button.click(
                open_output_folder,
                inputs=[],
                outputs=[batch_summary]
            )

if __name__ == "__main__":
    demo.launch()
