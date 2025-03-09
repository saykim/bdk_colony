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
import glob
import json

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
        self.removed_points = []  # 삭제된 포인트를 저장할 리스트 추가
        self.scale_factor = 1.0   # 이미지 리사이징 스케일 팩터 추가
        self.base_image = None    # 포인트 표시 없는 처리된 기본 이미지
        self.auto_annotations = []  # 자동 감지된 주석 데이터 저장

    def reset(self):
        self.manual_points = []
        self.auto_points = []
        self.current_image = None
        self.auto_detected_count = 0
        self.remove_mode = False
        self.original_image = None
        self.removed_points = []  # 삭제된 포인트 히스토리도 초기화
        self.scale_factor = 1.0
        self.base_image = None
        self.auto_annotations = []

    def set_original_image(self, image):
        self.original_image = np.array(image)

    def toggle_remove_mode(self):
        self.remove_mode = not self.remove_mode
        img_with_points = self.draw_points()
        mode_text = "🔴 REMOVE MODE" if self.remove_mode else "🟢 ADD MODE"
        return img_with_points, mode_text

    def add_or_remove_point(self, image, evt: gr.SelectData):
        try:
            # 현재 이미지가 없는 경우 입력된 이미지 사용
            if self.current_image is None and image is not None:
                self.current_image = np.array(image)
                
            # 이벤트 좌표 가져오기
            x, y = evt.index
            
            # 이미지가 없으면 처리 불가
            if self.current_image is None and self.base_image is None:
                print("Warning: No image available to add/remove points")
                return image, self.get_count_text()
                
            if self.remove_mode:
                # 제거 모드: 가장 가까운 점 찾기
                closest_auto = self.find_closest_point(x, y, self.auto_points)
                closest_manual = self.find_closest_point(x, y, self.manual_points)
                
                if closest_auto is not None:
                    # 삭제 전에 포인트와 타입 저장
                    removed_point = self.auto_points[closest_auto]
                    self.removed_points.append(('auto', removed_point))
                    self.auto_points.pop(closest_auto)
                    self.auto_detected_count = len(self.auto_points)
                elif closest_manual is not None:
                    # 삭제 전에 포인트와 타입 저장
                    removed_point = self.manual_points[closest_manual]
                    self.removed_points.append(('manual', removed_point))
                    self.manual_points.pop(closest_manual)
            else:
                # 추가 모드: 수동 포인트 추가
                self.manual_points.append((x, y))
                
            # 결과 이미지 다시 그리기
            img_with_points = self.draw_points()
            
            # 이미지가 None인 경우 기본 처리
            if img_with_points is None:
                if image is not None:
                    print("Warning: draw_points returned None, using input image")
                    return image, self.get_count_text()
                else:
                    # 입력 이미지도 없는 경우
                    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                    return dummy_img, self.get_count_text()
                    
            return img_with_points, self.get_count_text()
            
        except Exception as e:
            print(f"Error in add_or_remove_point: {str(e)}")
            # 오류 발생 시 입력 이미지 그대로 반환
            if image is not None:
                return image, self.get_count_text()
            # 모든 이미지가 None인 경우
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            return dummy_img, self.get_count_text()

    def find_closest_point(self, x, y, points, threshold=20):
        if not points:
            return None
        distances = [(np.sqrt((x - px) ** 2 + (y - py) ** 2), idx) for idx, (px, py) in enumerate(points)]
        closest_dist, closest_idx = min(distances, key=lambda item: item[0])
        return closest_idx if closest_dist < threshold else None

    def remove_last_point(self, image):
        if self.manual_points:
            removed_point = self.manual_points.pop()
            self.removed_points.append(('manual', removed_point))
        img_with_points = self.draw_points()
        return img_with_points, self.get_count_text()

    def undo_remove(self, image):
        """삭제된 포인트 복구 메서드"""
        if not self.removed_points:
            # 복구할 포인트가 없으면 현재 이미지와 카운트 텍스트만 반환
            return self.current_image, self.get_count_text()
            
        # 가장 최근에 삭제된 포인트 복구
        point_type, point = self.removed_points.pop()
        
        if point_type == 'auto':
            self.auto_points.append(point)
            self.auto_detected_count = len(self.auto_points)
        elif point_type == 'manual':
            self.manual_points.append(point)
            
        # 포인트가 그려진 이미지 업데이트
        img_with_points = self.draw_points()
        return img_with_points, self.get_count_text()

    def get_count_text(self):
        total_count = self.auto_detected_count + len(self.manual_points)
        return (f"전체 CFU 수: {total_count}\n"
                f"🤖 자동 감지된 CFU: {self.auto_detected_count}\n"
                f"👆 수동 추가된 CFU: {len(self.manual_points)}")

    def draw_points(self):
        """
        이미지에 감지된 포인트와 카운트를 그립니다
        
        Return:
            np.ndarray: 포인트와 카운트가 표시된 이미지
        """
        # base_image나 current_image가 없으면 기본 이미지 반환
        if self.base_image is None and self.current_image is None:
            print("Warning: No image available for drawing points")
            return None
            
        try:
            # 기반 이미지(세그멘테이션 결과)를 사용하여 매번 새롭게 그리기
            if self.base_image is not None:
                img_with_points = self.base_image.copy()
            elif self.current_image is not None:
                img_with_points = self.current_image.copy()
            else:
                print("Warning: Both base_image and current_image are None")
                return None
                
            # 이미지가 None인지 확인
            if img_with_points is None:
                print("Warning: Failed to create a copy of the image")
                return None
                
            overlay = np.zeros_like(img_with_points)
            
            # [사용자 설정 가능한 변수들]
            # 1. 마커 크기 및 폰트 설정
            square_size = 25  # 수동 포인트의 사각형 크기
            font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 종류
            font_scale = 0.77  # 텍스트 크기
            font_thickness = 1  # 내부 텍스트 두께
            outline_thickness = 4  # 외곽선 두께

            # 2. 색상 설정 (B, G, R 형식)
            AUTO_TEXT_COLOR = (255, 255, 255)  # 자동 감지 텍스트 색상 (흰색)
            AUTO_OUTLINE_COLOR = (0, 0, 0)     # 자동 감지 외곽선 색상 (검은색)
            MANUAL_RECT_COLOR = (0, 0, 255)    # 수동 추가 사각형 색상 (빨간색)
            MANUAL_BORDER_COLOR = (0, 0, 0)    # 수동 추가 테두리 색상 (검은색)
            MANUAL_TEXT_COLOR = (255, 136, 0)  # 수동 추가 텍스트 색상 (주황색)

            # 3. 투명도 설정
            OVERLAY_OPACITY = 0.4  # 오버레이 투명도
            
            # 자동 감지된 포인트 표시 (녹색 원 + 번호)
            for idx, (x, y) in enumerate(self.auto_points, 1):
                # 좌표는 이미 원본 이미지 좌표계이므로 현재 화면 크기에 맞게 직접 표시
                display_x = x 
                display_y = y
                
                # 텍스트 크기 계산하여 중앙 정렬
                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = int(display_x - text_width / 2)
                text_y = int(display_y - 10)
                
                # [중요] 8방향 검은색 외곽선으로 텍스트 가시성 향상
                for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    cv2.putText(img_with_points, text, 
                              (text_x + dx, text_y + dy), 
                              font, font_scale, AUTO_OUTLINE_COLOR, outline_thickness)
                
                # 흰색 텍스트
                cv2.putText(img_with_points, text, (text_x, text_y), 
                          font, font_scale, AUTO_TEXT_COLOR, font_thickness)
                
                # 녹색 원
                cv2.circle(img_with_points, (display_x, display_y), 5, (0, 255, 0), -1)

            # 수동 추가된 포인트 표시 (빨간 사각형)
            for idx, (x, y) in enumerate(self.manual_points, len(self.auto_points) + 1):
                # 좌표는 이미 원본 이미지 좌표계이므로 현재 화면 크기에 맞게 직접 표시
                display_x = x
                display_y = y
                
                # 사각형 좌표 계산
                pt1 = (int(display_x - square_size / 2), int(display_y - square_size / 2))
                pt2 = (int(display_x + square_size / 2), int(display_y + square_size / 2))
                
                # 반투명 사각형 그리기
                cv2.rectangle(overlay, pt1, pt2, MANUAL_RECT_COLOR, -1)  # 색상 채우기
                cv2.rectangle(overlay, pt1, pt2, MANUAL_BORDER_COLOR, 2)  # 테두리

                # 번호 텍스트 추가
                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = int(display_x - text_width / 2)
                text_y = int(display_y - 10)

                # 8방향 검은색 외곽선
                for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    cv2.putText(img_with_points, text,
                              (text_x + dx, text_y + dy),
                              font, font_scale, AUTO_OUTLINE_COLOR, outline_thickness)

                # 주황색 텍스트
                cv2.putText(img_with_points, text,
                          (text_x, text_y),
                          font, font_scale, MANUAL_TEXT_COLOR, font_thickness)

            # 오버레이 이미지를 원본과 블렌딩
            cv2.addWeighted(overlay, OVERLAY_OPACITY, img_with_points, 1.0, 0, img_with_points)

            # 제거 모드 표시 (상단에 빨간색 배너)
            if self.remove_mode:
                # 상단에 빨간색 배너 추가
                overlay = img_with_points.copy()
                cv2.rectangle(overlay, (0, 0), (img_with_points.shape[1], 40), MANUAL_RECT_COLOR, -1)
                cv2.addWeighted(overlay, 0.3, img_with_points, 0.7, 0, img_with_points)
                # "REMOVE MODE" 텍스트 표시
                cv2.putText(img_with_points, "REMOVE MODE", (10, 30),
                          font, 1, AUTO_TEXT_COLOR, 2)

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
        except Exception as e:
            print(f"Error in draw_points: {str(e)}")
            # 오류 발생 시 기존 이미지를 최대한 안전하게 반환
            if self.base_image is not None:
                return self.base_image
            if self.current_image is not None:
                return self.current_image
            # 모든 이미지가 None인 경우 빈 이미지 생성 (100x100 검은색)
            print("Creating fallback empty image")
            return np.zeros((100, 100, 3), dtype=np.uint8)

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
    conf_threshold=0.25,           # AI 예측 신뢰도 임계값 (높을수록 엄격)
    iou_threshold=0.7,             # 마스크 중복 임계값 (IoU 기준)
    circularity_threshold=0.8,     # 윤곽선의 원형도 필터링 임계값
    draw_contours=True,            # 윤곽선 시각화 여부
    mask_random_color=True,        # 랜덤 컬러 마스크 여부
    input_size=1024,               # AI 입력 이미지 크기
    better_quality=False,          # 품질 향상 모드 활성화 여부
    min_area_percentile=1,         # 최소 면적 필터링 임계치
    max_area_percentile=99,        # 최대 면적 필터링 기준
    use_dish_filtering=False,      # 페트리 접시 필터링 사용 여부
    dish_overlap_threshold=0.5,    # 배양 접시 중첩 비율 기준
    device=device,                 # 장치 명시화 (GPU/CPU)
    progress=gr.Progress()
):
    try:
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
        
        # 스케일 비율 저장 (좌표 변환에 사용)
        counter.scale_factor = scale
        
        try:
            input_resized = input_image.resize((new_w, new_h))
        except Exception as e:
            return np.array(input_image), f"이미지 리사이즈 오류: {str(e)}"
            
        image_np = np.array(input_resized)

        progress(0.3, desc="AI 분석 중...")
        try:
            results = model.predict(
                source=image_np,
                device=device,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=input_size,
                retina_masks=True
            )
        except Exception as e:
            return image_np, f"AI 모델 분석 오류: {str(e)}"

        # getattr()를 사용해 마스크 데이터 안전하게 접근
        annotations = getattr(results[0].masks, 'data', None)
        if annotations is None or len(annotations) == 0:
            return image_np, "CFU가 감지되지 않았습니다."

        progress(0.6, desc="결과 처리 중...")
        processed_image = image_np.copy()
        counter.auto_points = []

        # 배양 접시 필터링 처리
        dish_mask = None
        dish_idx = -1
        if use_dish_filtering:
            try:
                progress(0.65, desc="배양 접시 감지 중...")
                # 가장 큰 마스크를 배양 접시로 식별
                if len(annotations) > 0:
                    # 면적 계산
                    areas = [np.sum(ann.cpu().numpy()) for ann in annotations]
                    dish_idx = np.argmax(areas)
                    dish_annotation = annotations[dish_idx].cpu().numpy()
                    if dish_annotation.ndim == 3 and dish_annotation.shape[0] == 1:
                        dish_annotation = dish_annotation[0]  # (1, H, W) -> (H, W)
                    dish_mask = dish_annotation > 0
                    
                    # 배양 접시 윤곽선 시각화 (파란색)
                    dish_contours, _ = cv2.findContours(dish_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(processed_image, dish_contours, -1, (0, 0, 255), 2)
            except Exception as e:
                print(f"배양 접시 필터링 오류: {str(e)}")
                # 오류 발생 시에도 진행 - 접시 필터링 없이 계속 진행

        # 마스크 면적 계산 및 필터링
        all_masks = []
        all_areas = []
        
        try:
            # 모든 콜로니 마스크 처리 (배양 접시 제외)
            for idx, mask in enumerate(annotations):
                if idx == dish_idx:  # 배양 접시인 경우 건너뛰기
                    continue
                    
                mask_np = mask.cpu().numpy()
                if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                    mask_np = mask_np[0]  # (1, H, W) -> (H, W) 변환
                mask_binary = mask_np > 0
                
                # 배양 접시 필터링 적용
                if use_dish_filtering and dish_mask is not None:
                    # 배양 접시 내부 여부 확인 (마스크 교집합)
                    overlap_ratio = np.sum(mask_binary * dish_mask) / np.sum(mask_binary)
                    if overlap_ratio < dish_overlap_threshold:
                        continue  # 배양 접시와 충분히 겹치지 않는 경우 건너뛰기
                
                contours, _ = cv2.findContours(mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    all_areas.append(area)
                    all_masks.append((mask_binary, contour, area))
                    
            # 면적 기준 필터링 준비
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
                color_min = 120
                color_max = 255
            else:
                contour_thickness = 2
                color_min = 100
                color_max = 200
            
            # 필터링된 마스크 처리 (면적 및 원형도 기반)
            for mask_np, contour, area in all_masks:
                if min_area <= area <= max_area:
                    # 원형도 계산 (4π × area / perimeter²)
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
                    
                    # 원형도 임계값 이상의 윤곽선만 필터링
                    if perimeter > 0 and circularity >= circularity_threshold:
                        # 중심점 계산
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            counter.auto_points.append((cX, cY))
                            
                            # 콜로니 윤곽선 시각화
                            if draw_contours:
                                color = tuple(np.random.randint(color_min, color_max, 3).tolist()) if mask_random_color else (0, 255, 0)
                                cv2.drawContours(processed_image, [contour], -1, color, contour_thickness)
                                
                                # 마스크 내부 채우기 (선택적)
                                if mask_random_color:
                                    # 주석: 이 부분은 마스크 내부를 채우는 코드지만
                                    # 현재는 윤곽선만 그리므로 주석 처리
                                    # 원하는 경우 활성화하여 내부 채우기 가능
                                    # color_alpha = tuple(list(color) + [64])  # RGB + Alpha
                                    # processed_image[mask_np] = color[:3]
                                    pass
        except Exception as e:
            print(f"콜로니 처리 오류: {str(e)}")
            # 이미 찾은 auto_points는 유지

        # 결과 이미지 생성
        progress(0.9, desc="결과 이미지 생성 중...")
        try:
            # 카운터 정보 업데이트
            counter.auto_detected_count = len(counter.auto_points)
            
            # 안전 검사: processed_image가 None이면 원본 이미지를 사용
            if processed_image is None:
                print("Warning: processed_image is None, using original image instead")
                processed_image = image_np.copy()
                
            # 기본 이미지 저장
            counter.base_image = processed_image  # 포인트 없는 기본 처리 이미지 저장
            
            # 포인트 그리기
            counter.current_image = counter.draw_points()
            if counter.current_image is None:
                print("Warning: draw_points returned None, using processed_image instead")
                counter.current_image = processed_image
            
            # 원본 이미지 크기로 결과 이미지 조정
            try:
                img_with_points_pil = Image.fromarray(counter.current_image)
                try:
                    # PIL 9.0.0 이상
                    resampling_filter = Image.Resampling.LANCZOS
                except AttributeError:
                    # PIL 9.0.0 미만
                    resampling_filter = Image.LANCZOS
                img_with_points_resized = img_with_points_pil.resize((original_width, original_height), resampling_filter)
                return np.array(img_with_points_resized), counter.get_count_text()
            except Exception as e:
                print(f"Error during image resizing: {str(e)}")
                # 크기 조정 실패 시 원본 크기 그대로 반환
                return counter.current_image, counter.get_count_text()
                
        except Exception as e:
            print(f"결과 이미지 생성 오류: {str(e)}")
            # 오류 발생 시 원본 이미지 반환
            if input_image is not None:
                try:
                    # 원본 이미지를 NumPy 배열로 안전하게 변환
                    original_np = np.array(input_image)
                    return original_np, f"결과 이미지 생성 오류: {str(e)}"
                except:
                    # 모든 변환 실패 시 빈 이미지 생성
                    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                    return dummy_img, f"심각한 이미지 처리 오류: {str(e)}"
            # 입력 이미지 자체가 None인 경우
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            return dummy_img, f"분석 중 오류 발생: {str(e)}"
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        return np.array(input_image) if input_image else None, f"분석 중 오류 발생: {str(e)}"

# 결과 저장 함수 (원본과 결과 이미지 저장)
def save_results(original_image, processed_image):
    try:
        # 기본 출력 디렉터리 설정
        output_dir = "outputs"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            return f"결과 폴더 생성 오류: {str(e)}"
            
        # 타임스탬프로 고유 ID 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(output_dir, f"result_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # 파일 경로 설정
        original_path = os.path.join(result_dir, f"original.png")
        processed_path = os.path.join(result_dir, f"카운팅완료.png")
        
        # 원본 이미지 저장
        try:
            if original_image is not None:
                original_image.save(original_path)
            else:
                return "원본 이미지가 없습니다."
        except Exception as e:
            return f"원본 이미지 저장 오류: {str(e)}"
            
        # 처리된 이미지 저장
        try:
            if processed_image is not None:
                Image.fromarray(processed_image).save(processed_path)
            else:
                return "처리된 이미지가 없습니다."
        except Exception as e:
            if os.path.exists(original_path):
                # 적어도 원본은 저장했다면 알려줌
                return f"원본은 저장되었으나 결과 이미지 저장 중 오류 발생: {str(e)}\n- 원본: {original_path}"
            return f"결과 이미지 저장 오류: {str(e)}"
        
        # 결과 정보 저장 (JSON 형식)
        result_info = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'time': datetime.now().strftime("%H:%M:%S"),
            'auto_count': counter.auto_detected_count,
            'manual_count': len(counter.manual_points),
            'total_count': counter.auto_detected_count + len(counter.manual_points),
            'original_image': original_path,
            'processed_image': processed_path
        }
        
        # JSON 파일 저장
        json_path = os.path.join(result_dir, "result.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_info, f, indent=4)
        except Exception as e:
            print(f"JSON 결과 저장 오류: {str(e)}")
            
        return f"저장 완료:\n- 결과 폴더: {result_dir}\n- 원본: {original_path}\n- 결과: {processed_path}"
    
    except Exception as e:
        return f"저장 과정 중 오류 발생: {str(e)}"

# 배치 처리 함수 추가
def batch_process_images(
    folder_path,
    conf_threshold=0.25,
    iou_threshold=0.7,
    circularity_threshold=0.8,
    draw_contours=True,
    mask_random_color=True,
    input_size=1024,
    better_quality=False,
    min_area_percentile=1,
    max_area_percentile=99,
    use_dish_filtering=False,
    dish_overlap_threshold=0.5,
    progress=gr.Progress()
):
    try:
        # 폴더 경로 유효성 검사
        if not folder_path or not os.path.exists(folder_path):
            return "폴더 경로가 유효하지 않습니다."
        
        # 지원하는 이미지 확장자
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # 이미지 파일 목록 가져오기
        image_files = []
        for ext in valid_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
        
        if not image_files:
            return "폴더에 처리할 이미지 파일이 없습니다."
        
        # 결과 저장 폴더 생성
        try:
            batch_output_dir = os.path.join("outputs", f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(batch_output_dir, exist_ok=True)
        except Exception as e:
            return f"결과 폴더 생성 오류: {str(e)}"
        
        # 결과 CSV 파일 생성
        csv_path = os.path.join(batch_output_dir, "results.csv")
        results_data = []
        
        total_files = len(image_files)
        processed_count = 0
        error_count = 0
        
        progress(0, desc=f"총 {total_files}개 파일 처리 준비 중...")
        
        for idx, img_path in enumerate(image_files):
            try:
                file_name = os.path.basename(img_path)
                progress(idx/total_files, desc=f"처리 중: {file_name} ({idx+1}/{total_files})")
                
                # 이미지 로드
                try:
                    img = Image.open(img_path)
                except Exception as e:
                    error_count += 1
                    print(f"이미지 로드 오류 ({file_name}): {str(e)}")
                    results_data.append({
                        'FileName': file_name,
                        'FilePath': img_path,
                        'TotalCount': 'ERROR',
                        'AutoCount': 'ERROR',
                        'ManualCount': 'ERROR',
                        'ResultPath': 'ERROR',
                        'Error': f"이미지 로드 오류: {str(e)}"
                    })
                    continue
                
                # 이미지 분석
                try:
                    result_img, count_text = segment_and_count_colonies(
                        img,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        circularity_threshold=circularity_threshold,
                        draw_contours=draw_contours,
                        mask_random_color=mask_random_color,
                        input_size=input_size,
                        better_quality=better_quality,
                        min_area_percentile=min_area_percentile,
                        max_area_percentile=max_area_percentile,
                        use_dish_filtering=use_dish_filtering,
                        dish_overlap_threshold=dish_overlap_threshold
                    )
                except Exception as e:
                    error_count += 1
                    print(f"이미지 분석 오류 ({file_name}): {str(e)}")
                    results_data.append({
                        'FileName': file_name,
                        'FilePath': img_path,
                        'TotalCount': 'ERROR',
                        'AutoCount': 'ERROR',
                        'ManualCount': 'ERROR',
                        'ResultPath': 'ERROR',
                        'Error': f"이미지 분석 오류: {str(e)}"
                    })
                    continue
                
                # 결과 텍스트에서 숫자 추출
                try:
                    count_lines = count_text.split('\n')
                    total_count = int(count_lines[0].split(': ')[1]) if len(count_lines) > 0 else 0
                    auto_count = int(count_lines[1].split(': ')[1]) if len(count_lines) > 1 else 0
                    manual_count = int(count_lines[2].split(': ')[1]) if len(count_lines) > 2 else 0
                except Exception as e:
                    print(f"카운트 텍스트 파싱 오류 ({file_name}): {str(e)}")
                    total_count = -1
                    auto_count = -1
                    manual_count = -1
                
                # 결과 저장
                try:
                    result_file_path = os.path.join(batch_output_dir, f"result_{file_name}")
                    Image.fromarray(result_img).save(result_file_path)
                except Exception as e:
                    print(f"결과 저장 오류 ({file_name}): {str(e)}")
                    result_file_path = "저장 실패"
                
                # CSV 데이터 추가
                results_data.append({
                    'FileName': file_name,
                    'FilePath': img_path,
                    'TotalCount': total_count,
                    'AutoCount': auto_count,
                    'ManualCount': manual_count,
                    'ResultPath': result_file_path
                })
                
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"파일 처리 오류 ({file_name}): {str(e)}")
                results_data.append({
                    'FileName': file_name,
                    'FilePath': img_path,
                    'TotalCount': 'ERROR',
                    'AutoCount': 'ERROR',
                    'ManualCount': 'ERROR',
                    'ResultPath': 'ERROR',
                    'Error': str(e)
                })
        
        # CSV 파일로 결과 저장
        try:
            df = pd.DataFrame(results_data)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"CSV 저장 오류: {str(e)}")
            return f"처리 완료 ({processed_count}/{total_files} 성공) 하였으나 CSV 저장 중 오류 발생: {str(e)}"
        
        progress(1.0, desc="배치 처리 완료!")
        
        summary = (
            f"배치 처리 완료!\n"
            f"- 총 파일 수: {total_files}\n"
            f"- 성공: {processed_count}\n"
            f"- 실패: {error_count}\n"
            f"- 결과 폴더: {batch_output_dir}\n"
            f"- 결과 CSV: {csv_path}"
        )
        
        return summary
    
    except Exception as e:
        print(f"배치 처리 전체 오류: {str(e)}")
        return f"배치 처리 중 오류 발생: {str(e)}"

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
.footer {
    text-align: center;
    margin-top: 20px;
    padding: 10px;
    color: #666;
    font-size: 0.9em;
    border-top: 1px solid #eee;
}
.manual-edit-row {
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-top: 10px;
    margin-bottom: 10px;
}
.manual-edit-btn {
    background-color: #f1f1f1;
    color: #333;
    border: 1px solid #ddd;
    padding: 8px 16px;
    border-radius: 4px;
    transition: all 0.3s ease;
}
.manual-edit-btn:hover {
    background-color: #e0e0e0;
    transform: translateY(-1px);
}
.manual-edit-status {
    font-weight: bold;
    text-align: center;
    border: none;
    background-color: #fff;
    color: #2E7D32;
    padding: 8px 12px;
}
"""

counter = ColonyCounter()

with gr.Blocks(theme=gr.themes.Default(), css=css) as demo:
    gr.Markdown("""
    # 🔬 Colony Counter
    ## **AI 기반 CFU 자동 감지 및 수동 편집**
    
    이 애플리케이션은 AI 모델을 사용하여 배양접시의 CFU(Colony Forming Units)를 자동으로 감지하고 카운팅합니다.
    사용자는 자동 감지 결과를 수동으로 편집할 수도 있습니다.    
    
    """)
    
    with gr.Tabs() as tabs:
        with gr.TabItem("🔍 단일 이미지 분석"):
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
                    with gr.Accordion("⚙️ 분석 설정 [배양 접시 필터링 활성화됨]", open=False, elem_classes="accordion-header") as analysis_accordion:
                        analysis_setting_title = gr.Markdown("FastSAM AI 모델의 파라미터를 조정하여 CFU 감지 결과를 최적화할 수 있습니다.")
                        
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
                    
                    with gr.Row(elem_classes="manual-edit-row"):
                        remove_mode_btn = gr.Button("🔄 모드 전환", variant="secondary", elem_classes="manual-edit-btn")
                        mode_text = gr.Textbox(value="🟢 ADD MODE", interactive=False, elem_classes="manual-edit-status", show_label=False)
                        remove_last_btn = gr.Button("↩️ 최근 포인트 삭제", variant="secondary", elem_classes="manual-edit-btn")
                        undo_remove_btn = gr.Button("🔄 삭제 취소", variant="secondary", elem_classes="manual-edit-btn")
                    
                    save_btn = gr.Button("💾 결과 저장", variant="primary", elem_classes="button-primary")
                    save_output = gr.Textbox(label="저장 결과", interactive=False, elem_classes="result-text")
        
        with gr.TabItem("📊 배치 처리"):
            gr.Markdown("""
            <div class='section-title'>📊 배치 처리</div>
            <p>여러 이미지를 한 번에 처리하여 시간을 절약하세요.</p>
            """)
            
            with gr.Tabs():
                with gr.TabItem("📂 폴더 선택"):
                    folder_path_input = gr.Textbox(label="이미지 폴더 경로", placeholder="예: /path/to/images")
                    gr.Markdown("폴더 내의 모든 이미지 파일(.jpg, .jpeg, .png, .bmp, .tiff, .webp)을 처리합니다.")
                
                with gr.TabItem("🖼️ 파일 선택"):
                    batch_files = gr.File(file_count="multiple", label="이미지 파일 선택", file_types=["image"])
                    gr.Markdown("여러 이미지 파일을 직접 선택하여 처리합니다.")
            
            with gr.Accordion("⚙️ 배치 처리 설정", open=False):
                with gr.Row():
                    batch_input_size = gr.Slider(
                        512, 2048, 1024,
                        step=64,
                        label="입력 크기",
                        info="크기가 클수록 정확도가 높아지지만 속도는 느려집니다."
                    )
                    batch_better_quality = gr.Checkbox(
                        label="향상된 품질",
                        value=False,
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
                    batch_min_area_percentile = gr.Slider(
                        0, 10, 1,
                        step=1,
                        label="최소 크기 백분위수",
                        info="더 작은 CFU를 필터링합니다."
                    )
                    batch_max_area_percentile = gr.Slider(
                        90, 100, 99,
                        step=1,
                        label="최대 크기 백분위수",
                        info="더 큰 객체를 필터링합니다."
                    )
                
                batch_circularity_threshold = gr.Slider(
                    0.0, 1.0, 0.8,
                    step=0.01,
                    label="원형도 임계값",
                    info="대략적으로 원형인 CFU만 감지합니다."
                )
                
                batch_use_dish_filtering = gr.Checkbox(
                    label="배양 접시 필터링 사용",
                    value=True,
                    info="가장 큰 영역을 배양 접시로 인식하고 내부 CFU만 감지합니다."
                )
                
                batch_dish_overlap_threshold = gr.Slider(
                    0.1, 1.0, 0.5,
                    step=0.1,
                    label="배양 접시 겹침 임계값",
                    info="CFU가 배양 접시와 최소한 이 비율만큼 겹쳐야 합니다."
                )
            
            with gr.Row():
                batch_process_folder_btn = gr.Button("📂 폴더 처리 시작", variant="primary", elem_classes="button-primary")
                batch_process_files_btn = gr.Button("🖼️ 파일 처리 시작", variant="primary", elem_classes="button-primary")
                open_output_btn = gr.Button("📂 결과 폴더 열기", variant="secondary", elem_classes="button-secondary")
            
            batch_summary_output = gr.Textbox(label="배치 처리 결과", interactive=False, elem_classes="result-text", lines=10)
            
            # 결과 폴더 열기 함수
            def open_output_folder():
                import webbrowser
                output_dir = os.path.abspath("outputs")
                webbrowser.open(f"file://{output_dir}")
                return "결과 폴더를 열었습니다."

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

    undo_remove_btn.click(
        counter.undo_remove,
        inputs=[output_image],
        outputs=[output_image, count_text]
    )

    save_btn.click(
        save_results,
        inputs=[input_image, output_image],
        outputs=save_output
    )
    
    # 배양 접시 필터링 상태에 따라 제목 업데이트
    def update_title(is_enabled):
        try:
            if is_enabled:
                return gr.Accordion.update(label="⚙️ 분석 설정 [배양 접시 필터링 활성화됨]")
            else:
                return gr.Accordion.update(label="⚙️ 분석 설정")
        except Exception as e:
            print(f"update_title 오류: {str(e)}")
            return gr.Accordion.update(label="⚙️ 분석 설정")
    
    use_dish_filtering_checkbox.change(
        update_title,
        inputs=[use_dish_filtering_checkbox],
        outputs=[analysis_accordion]
    )
    
    # 파일 기반 배치 처리 함수
    def batch_process_files(
        files,
        conf_threshold,
        iou_threshold,
        circularity_threshold,
        draw_contours,
        mask_random_color,
        input_size,
        better_quality,
        min_area_percentile,
        max_area_percentile,
        use_dish_filtering,
        dish_overlap_threshold,
        progress=gr.Progress()
    ):
        try:
            if not files:
                return "처리할 파일이 없습니다. 파일을 선택해주세요."
            
            # 결과 저장 폴더 생성
            try:
                batch_output_dir = os.path.join("outputs", f"batch_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(batch_output_dir, exist_ok=True)
            except Exception as e:
                return f"결과 폴더 생성 오류: {str(e)}"
            
            # 결과 저장을 위한 데이터 준비
            results_data = []
            
            total_files = len(files)
            processed_count = 0
            error_count = 0
            
            progress(0, desc=f"총 {total_files}개 파일 처리 준비 중...")
            
            for idx, file in enumerate(files):
                progress_pct = (idx / total_files) * 100
                progress(progress_pct / 100, desc=f"파일 {idx+1}/{total_files} 처리 중...")
                
                # 결과 정보 초기화
                result_info = {
                    'filename': os.path.basename(file.name),
                    'original_path': file.name,
                    'processed_path': '',
                    'error': None,
                    'auto_count': 0,
                    'manual_count': 0,
                    'total_count': 0,
                    'count_text': '',
                    'process_time': 0
                }
                
                try:
                    start_time = time.time()
                    
                    # 이미지 파일 로드
                    try:
                        img = Image.open(file.name).convert("RGB")
                    except Exception as e:
                        result_info['error'] = f"이미지 로드 오류: {str(e)}"
                        results_data.append(result_info)
                        error_count += 1
                        continue
                        
                    # 콜로니 분석 수행
                    try:
                        # 로컬 카운터 생성 (현재 전역 카운터에 영향 없도록)
                        local_counter = ColonyCounter()
                        local_counter.reset()
                        
                        processed_img, count_text = segment_and_count_colonies(
                            input_image=img,
                            conf_threshold=conf_threshold,
                            iou_threshold=iou_threshold,
                            circularity_threshold=circularity_threshold,
                            draw_contours=draw_contours,
                            mask_random_color=mask_random_color,
                            input_size=input_size,
                            better_quality=better_quality,
                            min_area_percentile=min_area_percentile,
                            max_area_percentile=max_area_percentile,
                            use_dish_filtering=use_dish_filtering,
                            dish_overlap_threshold=dish_overlap_threshold
                        )
                        
                        # 오류 확인
                        if "오류" in count_text:
                            result_info['error'] = count_text
                            results_data.append(result_info)
                            error_count += 1
                            continue
                            
                        # 결과 이미지 저장
                        filename_only = os.path.splitext(os.path.basename(file.name))[0]
                        processed_path = os.path.join(batch_output_dir, f"{filename_only}_result.png")
                        Image.fromarray(processed_img).save(processed_path)
                        
                        # 처리 시간 측정
                        process_time = time.time() - start_time
                        
                        # 결과 정보 업데이트
                        result_info['processed_path'] = processed_path
                        result_info['auto_count'] = counter.auto_detected_count
                        result_info['manual_count'] = len(counter.manual_points)
                        result_info['total_count'] = counter.auto_detected_count + len(counter.manual_points)
                        result_info['count_text'] = count_text
                        result_info['process_time'] = round(process_time, 2)
                        
                        processed_count += 1
                        
                    except Exception as e:
                        result_info['error'] = f"분석 처리 오류: {str(e)}"
                    
                except Exception as e:
                    result_info['error'] = f"파일 처리 오류: {str(e)}"
                
                results_data.append(result_info)
                
            # 요약 정보 생성
            progress(0.95, desc="결과 요약 생성 중...")
            
            try:
                # 요약 JSON 생성
                summary = {
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'total_images': total_files,
                    'successful': processed_count,
                    'failed': error_count,
                    'results': results_data
                }
                
                # JSON 파일로 저장
                summary_json_path = os.path.join(batch_output_dir, "summary.json")
                with open(summary_json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=4)
                
                # CSV 파일로도 저장
                df = pd.DataFrame(results_data)
                csv_path = os.path.join(batch_output_dir, "results.csv")
                df.to_csv(csv_path, index=False)
                
            except Exception as e:
                return f"결과 요약 생성 오류: {str(e)}\n처리된 파일 수: {processed_count}, 오류 발생 파일 수: {error_count}"
            
            # 최종 요약 메시지 반환
            result_msg = f"""
            처리 완료:
            - 총 파일: {total_files}개
            - 처리 성공: {processed_count}개
            - 오류 발생: {error_count}개
            - 결과 폴더: {batch_output_dir}\n
            """
            
            return result_msg
            
        except Exception as e:
            return f"배치 처리 중 오류 발생: {str(e)}"
    
    # 이벤트 핸들러 연결
    batch_process_folder_btn.click(
        batch_process_images,
        inputs=[
            folder_path_input,
            batch_conf_threshold,
            batch_iou_threshold,
            batch_circularity_threshold,
            gr.Checkbox(value=True, visible=False),  # draw_contours
            gr.Checkbox(value=True, visible=False),  # mask_random_color
            batch_input_size,
            batch_better_quality,
            batch_min_area_percentile,
            batch_max_area_percentile,
            batch_use_dish_filtering,
            batch_dish_overlap_threshold
        ],
        outputs=batch_summary_output
    )
    
    batch_process_files_btn.click(
        batch_process_files,
        inputs=[
            batch_files,
            batch_conf_threshold,
            batch_iou_threshold,
            batch_circularity_threshold,
            gr.Checkbox(value=True, visible=False),  # draw_contours
            gr.Checkbox(value=True, visible=False),  # mask_random_color
            batch_input_size,
            batch_better_quality,
            batch_min_area_percentile,
            batch_max_area_percentile,
            batch_use_dish_filtering,
            batch_dish_overlap_threshold
        ],
        outputs=batch_summary_output
    )
    
    open_output_btn.click(
        open_output_folder,
        inputs=[],
        outputs=batch_summary_output
    )

    # 푸터 추가
    gr.Markdown("<div class='footer'>Produced by BDK®</div>")

if __name__ == "__main__":
    demo.launch()