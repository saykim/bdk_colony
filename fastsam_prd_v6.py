import os
import sys # sys 모듈 추가 (프로그램 종료용)
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
import webbrowser # 결과 폴더 열기용

# --- 수정: 모델 경로 및 존재 확인 ---
model_path = 'weights/FastSAM-x.pt'
if not os.path.exists(model_path):
    print(f"오류: 모델 파일 '{model_path}'를 찾을 수 없습니다.")
    print("프로그램을 종료합니다.")
    sys.exit(1) # 모델 파일 없으면 종료
# ---------------------------------

# AI 기반 분할 모델 로드
model = YOLO(model_path)

# 디바이스 설정 (CUDA, MPS, CPU 순으로 사용 가능)
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"사용 중인 디바이스: {device}")

# --- 수정: 결과 저장 기본 경로 ---
OUTPUT_BASE_DIR = "outputs"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
# ---------------------------------

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
        # 더 이상 되돌릴 수 없으면 현재 이미지 반환
        return self.current_image

# 이미지 히스토리 객체 생성
image_history = ImagePreprocessHistory()

class ColonyCounter:
    """콜로니 카운팅 및 포인트 관리 클래스"""
    def __init__(self):
        """초기화"""
        self.auto_points = []  # 자동 감지된 포인트 (중심점 좌표)
        self.manual_points = []  # 수동으로 추가된 포인트
        self.auto_detected_count = 0  # 자동 감지된 콜로니 수
        self.original_image = None  # 원본 이미지 (리사이징 후)
        self.current_image = None  # 현재 이미지 (포인트가 그려진 상태)
        self.colony_masks = []  # 콜로니 마스크 리스트 (Tensor 리스트)
        self.dish_mask = None  # 배양접시 마스크 (Tensor)
        self.base_image = None  # 처리된 베이스 이미지 (Numpy array)
        self.remove_mode = False  # 제거 모드 여부
        self.scale_factor = 1.0  # 이미지 스케일 팩터
        self.last_method = None  # 마지막으로 사용한 분석 방법

    def reset(self):
        """카운터 초기화"""
        self.auto_points = []
        self.manual_points = []
        self.auto_detected_count = 0
        self.original_image = None
        self.current_image = None
        self.colony_masks = []
        self.dish_mask = None
        self.base_image = None
        self.remove_mode = False
        self.scale_factor = 1.0
        self.last_method = None

    def set_original_image(self, image):
        """원본 이미지 설정 (리사이징 후)"""
        if image is not None:
            if isinstance(image, Image.Image):
                self.original_image = np.array(image)
            else:
                # 이미 Numpy 배열인 경우
                self.original_image = image.copy()
            # base_image도 초기에는 원본과 동일하게 설정
            self.base_image = self.original_image.copy()

    def toggle_remove_mode(self):
        """편집 모드 전환 (추가/제거 모드)"""
        self.remove_mode = not self.remove_mode
        # 모드 변경 시 현재 상태로 이미지 다시 그림
        img_with_points = self.draw_points()
        # mode_text = "🔴 REMOVE MODE" if self.remove_mode else "🟢 ADD MODE"
        # return img_with_points, mode_text

        # HTML 형식으로 모드 표시 문자열 반환
        if self.remove_mode:
            mode_indicator_html = "<span style='color: red; font-weight: bold; padding: 10px; display: inline-block;'>🔴 REMOVE MODE</span>"
        else:
            mode_indicator_html = "<span style='color: green; font-weight: bold; padding: 10px; display: inline-block;'>🟢 ADD MODE</span>"
        return img_with_points, mode_indicator_html

    def set_segmentation_data(self, colony_masks, dish_mask, processed_image_np):
        """세그멘테이션 데이터 설정 (콜로니 마스크, 접시 마스크, 처리된 이미지)"""
        self.colony_masks = colony_masks if colony_masks is not None else []
        self.dish_mask = dish_mask
        # base_image는 항상 Numpy 배열로 저장
        self.base_image = processed_image_np.copy() if processed_image_np is not None else None
        # current_image도 초기에는 base_image와 동일하게 설정
        self.current_image = self.base_image.copy() if self.base_image is not None else None


    def _get_mask_centroid(self, mask_tensor):
        """마스크 텐서의 중심점을 계산하는 내부 함수"""
        if mask_tensor is None:
            return None
        mask_np = mask_tensor.cpu().numpy()
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        mask_bool = mask_np > 0
        if np.any(mask_bool):
            y_indices, x_indices = np.where(mask_bool)
            center_x = int(np.mean(x_indices))
            center_y = int(np.mean(y_indices))
            return (center_x, center_y)
        return None

    def add_or_remove_point(self, image, evt: gr.SelectData):
        """이미지를 클릭하여 포인트 추가 또는 제거"""
        try:
            # current_image가 None이면 초기화 시도 (오류 방지)
            if self.current_image is None and self.base_image is not None:
                 self.current_image = self.base_image.copy()
            elif self.current_image is None and image is not None:
                 self.current_image = np.array(image) # Fallback

            if self.current_image is None:
                 print("오류: 현재 이미지가 설정되지 않아 포인트 추가/제거 불가")
                 return image, self.get_count_text() # 원본 이미지 반환

            x, y = evt.index  # Gradio에서 클릭된 좌표 (UI 기준)

            if self.remove_mode:
                # 제거 모드: 가장 가까운 포인트 찾기 (자동/수동 통합)
                closest_point_idx = -1
                min_dist_sq = float('inf')
                is_auto = False

                # 자동 포인트에서 가장 가까운 점 찾기
                for idx, (px, py) in enumerate(self.auto_points):
                    dist_sq = (x - px)**2 + (y - py)**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_point_idx = idx
                        is_auto = True

                # 수동 포인트에서 더 가까운 점이 있는지 확인
                for idx, (px, py) in enumerate(self.manual_points):
                    dist_sq = (x - px)**2 + (y - py)**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_point_idx = idx
                        is_auto = False

                threshold_sq = 20**2 # 거리 임계값 (제곱으로 비교하여 sqrt 연산 줄임)
                removed = False

                if min_dist_sq < threshold_sq:
                    if is_auto:
                        # 자동 포인트 및 해당 마스크 제거 로직
                        removed_point = self.auto_points.pop(closest_point_idx)
                        # 인덱스 대신 좌표 기반으로 마스크 찾기
                        mask_to_remove_idx = -1
                        min_centroid_dist_sq = float('inf')
                        for mask_idx, mask_tensor in enumerate(self.colony_masks):
                            centroid = self._get_mask_centroid(mask_tensor)
                            if centroid:
                                centroid_dist_sq = (removed_point[0] - centroid[0])**2 + (removed_point[1] - centroid[1])**2
                                # 매우 가까운 마스크를 찾음 (동일 좌표로 가정)
                                if centroid_dist_sq < 1.0: # 중심점이 거의 일치하는 마스크
                                     mask_to_remove_idx = mask_idx
                                     break # 찾으면 중단
                                # elif centroid_dist_sq < min_centroid_dist_sq: # 혹시 모르니 가장 가까운 것도 기록
                                #      min_centroid_dist_sq = centroid_dist_sq

                        if mask_to_remove_idx != -1:
                             self.colony_masks.pop(mask_to_remove_idx)
                             print(f"자동 포인트 {closest_point_idx}와 연관된 마스크 {mask_to_remove_idx} 제거됨.")
                        else:
                             print(f"경고: 자동 포인트 {closest_point_idx} ({removed_point}) 에 해당하는 마스크를 찾지 못했습니다.")

                        self.auto_detected_count = len(self.auto_points)
                        removed = True
                    else:
                        # 수동 포인트 제거 (인덱스 주의: is_auto가 False일 때 closest_point_idx는 manual_points 내의 인덱스)
                        actual_manual_idx = closest_point_idx # closest_point_idx가 manual_points 내 인덱스임
                        if actual_manual_idx < len(self.manual_points):
                            self.manual_points.pop(actual_manual_idx)
                            removed = True
                        else:
                            print(f"오류: 잘못된 수동 포인트 인덱스 {actual_manual_idx}")


                if removed:
                    # 마스크가 변경되었거나 수동 포인트가 제거되었으므로 base_image를 다시 생성하고 current_image 업데이트
                    self.redraw_segmentation_and_base() # 자동 제거 시 마스크 기반 이미지 재생성
                else:
                    print("제거할 포인트가 충분히 가까이 있지 않습니다.")

            else:
                # 추가 모드: 새로운 수동 포인트 추가
                self.manual_points.append((x, y))
                # 추가 시에는 세그멘테이션 변경 없으므로 redraw 필요 없음

            # 최종적으로 포인트가 그려진 이미지 업데이트
            img_with_points = self.draw_points()
            self.current_image = img_with_points # current_image 업데이트
            return img_with_points, self.get_count_text()

        except Exception as e:
            print(f"포인트 추가/제거 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc() # 상세 오류 출력
            # 오류 발생 시 안전하게 현재 이미지 또는 원본 이미지 반환
            return self.current_image if self.current_image is not None else image, self.get_count_text()

    def redraw_segmentation_and_base(self):
        """
        세그멘테이션 마스크를 기반으로 base_image를 다시 그리고,
        current_image도 업데이트합니다. (자동 포인트 제거 시 호출)
        """
        if self.original_image is None:
            print("오류: 원본 이미지가 없어 세그멘테이션을 다시 그릴 수 없습니다.")
            return

        # 원본 이미지 복사 (리사이징된 이미지 기준)
        image_np = self.original_image.copy()

        # 현재 남아있는 콜로니 마스크 처리
        for mask_tensor in self.colony_masks:
            mask_np = mask_tensor.cpu().numpy()
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np[0]
            mask_bool = mask_np > 0

            if mask_bool.ndim == 2 and np.any(mask_bool):
                # 랜덤 색상 적용
                color = np.random.randint(100, 200, (3,)).tolist()
                image_np[mask_bool] = color

                # 윤곽선 그리기
                contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_np, contours, -1, (255, 0, 0), 2) # 파란색 윤곽선

        # 배양접시 마스크 처리
        if self.dish_mask is not None:
            dish_mask_np = self.dish_mask.cpu().numpy()
            if dish_mask_np.ndim == 3 and dish_mask_np.shape[0] == 1:
                dish_mask_np = dish_mask_np[0]
            dish_mask_bool = dish_mask_np > 0

            if dish_mask_bool.ndim == 2 and np.any(dish_mask_bool):
                contours, _ = cv2.findContours(dish_mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_np, contours, -1, (0, 0, 255), 3) # 빨간색 윤곽선

        # 업데이트된 이미지를 base_image로 설정
        self.base_image = image_np
        # current_image도 즉시 업데이트 (포인트는 draw_points에서 그려짐)
        self.current_image = self.base_image.copy()


    def remove_last_point(self, image):
        """마지막으로 추가된 *수동* 포인트 제거"""
        try:
            if len(self.manual_points) > 0:
                self.manual_points.pop()
                # 포인트 제거 후 이미지 다시 그림
                updated_image = self.draw_points()
                self.current_image = updated_image # current_image 업데이트
                return updated_image, self.get_count_text()
            else:
                # 제거할 수동 포인트가 없으면 변화 없음
                return image, self.get_count_text()
        except Exception as e:
            print(f"마지막 수동 포인트 제거 중 오류 발생: {str(e)}")
            return image, self.get_count_text()

    def get_count_text(self):
        """현재 카운트 텍스트 가져오기"""
        auto_count = self.auto_detected_count
        manual_count = len(self.manual_points)
        total_count = auto_count + manual_count

        # 방법 정보 제거 (FastSAM 표시 안함)
        text = f"총 콜로니 수: {total_count}\n(자동: {auto_count}, 수동: {manual_count})"
        return text

    def draw_points(self):
        """현재 이미지에 포인트를 그려서 반환"""
        try:
            # base_image를 기반으로 그림
            if self.base_image is None:
                print("경고: base_image가 없어 포인트를 그릴 수 없습니다.")
                # current_image라도 있으면 반환, 아니면 None
                return self.current_image if self.current_image is not None else None
            img_with_points = self.base_image.copy()
            overlay = np.zeros_like(img_with_points, dtype=np.uint8) # 오버레이 타입 명시

            # 사용자 설정 가능한 변수들
            square_size = 40
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 1
            outline_thickness = 3
            AUTO_TEXT_COLOR = (255, 255, 255)
            AUTO_OUTLINE_COLOR = (0, 0, 0)
            MANUAL_RECT_COLOR = (0, 0, 255)
            MANUAL_BORDER_COLOR = (0, 0, 0)
            MANUAL_TEXT_COLOR = (255, 255, 0)
            OVERLAY_OPACITY = 0.6
            REMOVE_MODE_OPACITY = 0.3

            # 1. 자동 감지된 CFU 번호 표시
            for idx, (x, y) in enumerate(self.auto_points, 1):
                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = int(x - text_width / 2)
                text_y = int(y + text_height / 2) # Y 좌표 중앙 정렬 개선

                # 외곽선
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                             cv2.putText(img_with_points, text, (text_x + dx, text_y + dy),
                                         font, font_scale, AUTO_OUTLINE_COLOR, outline_thickness)
                # 텍스트
                cv2.putText(img_with_points, text, (text_x, text_y),
                          font, font_scale, AUTO_TEXT_COLOR, font_thickness)

            # 2. 수동으로 추가된 포인트 표시
            manual_font_scale = 1.0
            manual_font_thickness = 2
            for idx, (x, y) in enumerate(self.manual_points, len(self.auto_points) + 1):
                pt1 = (int(x - square_size / 2), int(y - square_size / 2))
                pt2 = (int(x + square_size / 2), int(y + square_size / 2))

                # 오버레이에 사각형 그리기
                cv2.rectangle(overlay, pt1, pt2, MANUAL_RECT_COLOR, -1)
                cv2.rectangle(overlay, pt1, pt2, MANUAL_BORDER_COLOR, 2)

                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, manual_font_scale, manual_font_thickness)
                text_x = int(x - text_width / 2)
                text_y = int(y + text_height / 2) # Y 좌표 중앙 정렬 개선

                # 외곽선 (오버레이에 그림)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                         if dx != 0 or dy != 0:
                              cv2.putText(overlay, text, (text_x + dx, text_y + dy),
                                          font, manual_font_scale, AUTO_OUTLINE_COLOR, outline_thickness + 1) # 외곽선 더 두껍게
                # 텍스트 (오버레이에 그림)
                cv2.putText(overlay, text, (text_x, text_y),
                          font, manual_font_scale, MANUAL_TEXT_COLOR, manual_font_thickness)

            # 오버레이 이미지를 원본과 블렌딩
            cv2.addWeighted(overlay, OVERLAY_OPACITY, img_with_points, 1.0, 0, img_with_points)

            # 3. 제거 모드 표시
            mode_overlay = img_with_points.copy() # 모드 표시용 복사본
            if self.remove_mode:
                cv2.rectangle(mode_overlay, (0, 0), (img_with_points.shape[1], 50), (0, 0, 255), -1) # 빨간색 배경
                cv2.addWeighted(mode_overlay, REMOVE_MODE_OPACITY, img_with_points, 1.0 - REMOVE_MODE_OPACITY, 0, img_with_points)
                cv2.putText(img_with_points, "REMOVE MODE", (10, 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            else:
                cv2.rectangle(mode_overlay, (0, 0), (img_with_points.shape[1], 50), (0, 255, 0), -1) # 초록색 배경
                cv2.addWeighted(mode_overlay, REMOVE_MODE_OPACITY, img_with_points, 1.0 - REMOVE_MODE_OPACITY, 0, img_with_points)
                cv2.putText(img_with_points, "ADD MODE", (10, 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            # 4. 전체 카운트 표시 (왼쪽 하단)
            # count_text_str = self.get_count_text().split('\\n')[0] # 첫 줄만 표시 (기존 코드 주석 처리)
            total_count = len(self.auto_points) + len(self.manual_points)
            count_text_str = f"Total count : {total_count}" # 새로운 텍스트 형식
            count_font_scale = 1.0
            count_font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(count_text_str, font, count_font_scale, count_font_thickness)
            margin = 15
            # 텍스트 배경 사각형 그리기
            cv2.rectangle(img_with_points,
                          (margin, img_with_points.shape[0] - margin - text_height - baseline),
                          (margin + text_width + margin, img_with_points.shape[0] - margin + baseline),
                          (0, 0, 0), -1) # 검은색 배경
            # 카운트 텍스트 표시
            cv2.putText(img_with_points, count_text_str, # 변경된 텍스트 사용
                      (margin + margin // 2, img_with_points.shape[0] - margin - baseline // 2),
                      font, count_font_scale, (255, 255, 255), count_font_thickness) # 흰색 텍스트

            return img_with_points
        except Exception as e:
            print(f"이미지에 포인트를 그리는 중 오류가 발생했습니다: {str(e)}")
            # 오류 시 안전하게 현재 이미지 또는 base 이미지 반환
            return self.current_image if self.current_image is not None else self.base_image

def preprocess_image(
    input_image,
    to_grayscale=False,
    binary=False,
    binary_threshold=128,
    edge_detection=False,
    sharpen=False,
    sharpen_amount=1.0
):
    """이미지 전처리를 수행합니다."""
    try:
        if not isinstance(input_image, Image.Image):
             print("전처리 입력은 PIL Image여야 합니다.")
             return input_image # 원본 반환

        image = input_image.copy()

        if to_grayscale:
            image = image.convert('L').convert('RGB')
        if binary:
            image_np = np.array(image.convert('L')) # 흑백으로 변환 후 처리
            _, binary_img = cv2.threshold(image_np, binary_threshold, 255, cv2.THRESH_BINARY)
            image = Image.fromarray(binary_img).convert('RGB')
        if edge_detection:
            image_np = np.array(image.convert('L'))
            edges = cv2.Canny(image_np, 100, 200)
            image = Image.fromarray(edges).convert('RGB')
        if sharpen:
            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            if sharpen_amount != 1.0:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(sharpen_amount)
        return image
    except Exception as e:
        print(f"이미지 전처리 중 오류 발생: {str(e)}")
        return input_image # 오류 시 원본 반환

def process_image(colony_annotations, dish_annotation, image, device, scale, better_quality, mask_random_color, bbox, use_retina, withContours):
    """마스크 주석을 기반으로 이미지를 처리하여 CFU의 윤곽선을 그립니다."""
    try:
        image_np = np.array(image).copy()

        # CFU 마스크 처리
        for ann in colony_annotations:
            ann_cpu = ann.cpu().numpy()
            if ann_cpu.ndim == 3 and ann_cpu.shape[0] == 1:
                ann_cpu = ann_cpu[0]

            mask = ann_cpu > 0
            if mask.ndim == 2 and np.any(mask):
                if mask_random_color:
                    color = np.random.randint(100, 200, (3,)).tolist()
                    image_np[mask] = color
                else:
                    image_np[mask] = (0, 255, 0) # 기본 녹색

                if withContours:
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image_np, contours, -1, (255, 0, 0), 2) # 파란색 윤곽선

        # 배양접시 마스크 처리 (윤곽선만 그리기)
        if dish_annotation is not None:
            dish_mask = dish_annotation.cpu().numpy()
            if dish_mask.ndim == 3 and dish_mask.shape[0] == 1:
                dish_mask = dish_mask[0]
            dish_mask = dish_mask > 0
            if dish_mask.ndim == 2 and np.any(dish_mask):
                if withContours:
                    contours, _ = cv2.findContours(dish_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image_np, contours, -1, (0, 0, 255), 3) # 빨간색 윤곽선

        processed_image = Image.fromarray(image_np)
        return processed_image
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return image # 오류 시 원본 PIL 이미지 반환

def segment_and_count_colonies(
    input_image, # PIL Image 입력
    input_size=1024,
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    mask_random_color=True, # 기본값 True 사용
    # min_area_percentile=1, # 기존 파라미터 제거
    # max_area_percentile=99, # 기존 파라미터 제거
    circularity_threshold=0.8,
    # --- 면적 필터링 파라미터 추가 ---
    enable_area_filter=False,
    min_area_percentile=1,
    max_area_percentile=99,
    # -------------------------------
    progress=gr.Progress()
):
    """이미지를 분석하고 CFU를 감지하여 카운팅합니다."""
    global counter # 전역 counter 객체 사용

    try:
        if input_image is None:
            return None, "이미지를 업로드해주세요."

        # 분석 시작 시 전역 counter 리셋
        counter.reset()

        progress(0.1, desc="이미지 준비 중...")
        image_to_use = input_image
        original_width, original_height = image_to_use.size
        w, h = image_to_use.size
        scale = input_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        # PIL 9.0.0 이상/미만 호환성
        try: resampling_filter = Image.Resampling.LANCZOS
        except AttributeError: resampling_filter = Image.LANCZOS
        input_resized_pil = image_to_use.resize((new_w, new_h), resampling_filter)
        input_resized_np = np.array(input_resized_pil) # 모델 입력용 Numpy 배열

        # counter에 원본(리사이즈된) 이미지와 스케일 저장
        counter.set_original_image(input_resized_np) # Numpy 배열로 저장
        counter.scale_factor = scale
        counter.last_method = "FastSAM" # 분석 방법 기록

        progress(0.3, desc="AI 분석 중...")
        results = model.predict(
            input_resized_np, # Numpy 배열 입력
            device=device,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=input_size,
            retina_masks=True
        )

        annotations = getattr(results[0].masks, 'data', None)
        if annotations is None or len(annotations) == 0:
            counter.current_image = input_resized_np # 분석 실패 시 리사이즈된 이미지 표시
            return input_resized_np, "CFU가 감지되지 않았습니다."

        progress(0.6, desc="마스크 처리 및 필터링 중...")
        areas = [np.sum(ann.cpu().numpy() > 0) for ann in annotations]
        if not areas: # 모든 마스크가 비어있는 경우 방지
             counter.current_image = input_resized_np
             return input_resized_np, "유효한 마스크 영역이 없습니다."

        dish_idx = np.argmax(areas)
        dish_annotation = annotations[dish_idx]
        colony_annotations_all = [ann for idx, ann in enumerate(annotations) if idx != dish_idx]

        # 면적 및 원형도 필터링
        valid_colony_annotations = []
        valid_auto_points = []
        colony_areas = [areas[i] for i in range(len(areas)) if i != dish_idx]

        if colony_areas: # 콜로니 후보가 있을 경우에만 필터링
             # 백분위수 대신 실제 면적 임계값 사용 고려 (옵션)
             # min_area = np.percentile(colony_areas, min_area_percentile)
             # max_area = np.percentile(colony_areas, max_area_percentile)
             # min_area = 3 # 최소 면적 임계값을 3으로 변경 # 하드코딩된 값 삭제
             # max_area = input_size * input_size * 0.1 # 하드코딩된 값 삭제

             # --- 면적 필터링 로직 수정 ---
             if enable_area_filter:
                 min_area = np.percentile(colony_areas, min_area_percentile)
                 max_area = np.percentile(colony_areas, max_area_percentile)
                 print(f"면적 필터링 활성화: 최소 {min_area:.2f}, 최대 {max_area:.2f} (백분위 {min_area_percentile}-{max_area_percentile} %)")
             # ---------------------------

             for ann, area in zip(colony_annotations_all, colony_areas):
                 # 면적 필터링 (필터링 활성화 시 적용)
                 if enable_area_filter and (area < min_area or area > max_area):
                      continue

                 ann_cpu = ann.cpu().numpy()
                 if ann_cpu.ndim == 3 and ann_cpu.shape[0] == 1: ann_cpu = ann_cpu[0]
                 mask = ann_cpu > 0

                 # 원형도 계산
                 contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                 if contours:
                     perimeter = cv2.arcLength(contours[0], True)
                     circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
                     if circularity >= circularity_threshold:
                         valid_colony_annotations.append(ann)
                         # 중심점 계산 및 추가
                         centroid = counter._get_mask_centroid(ann)
                         if centroid:
                             valid_auto_points.append(centroid)

        progress(0.8, desc="결과 시각화 중...")
        processed_image_pil = process_image(
            colony_annotations=valid_colony_annotations,
            dish_annotation=dish_annotation,
            image=input_resized_pil, # PIL 이미지 전달
            device=device,
            scale=1.0, # process_image 내부 스케일은 1.0
            better_quality=better_quality,
            mask_random_color=mask_random_color, # 함수 인자로 전달된 값 사용
            bbox=None,
            use_retina=False,
            withContours=withContours
        )
        processed_image_np = np.array(processed_image_pil) # Numpy 배열로 변환

        # counter 객체 업데이트
        counter.set_segmentation_data(valid_colony_annotations, dish_annotation, processed_image_np)
        counter.auto_points = valid_auto_points
        counter.auto_detected_count = len(valid_auto_points)

        progress(1.0, desc="완료!")
        img_with_points_np = counter.draw_points() # 최종 포인트 포함 이미지 (Numpy)

        # UI에는 리사이즈된 결과 이미지 반환
        return img_with_points_np, counter.get_count_text()

    except Exception as e:
        error_msg = f"분석 중 오류가 발생했습니다: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc() # 상세 오류 출력
        # 오류 발생 시 원본 이미지 반환 (리사이즈된)
        if 'input_resized_np' in locals():
             return input_resized_np, error_msg
        elif input_image is not None:
             return np.array(input_image), error_msg
        return None, error_msg

def save_results(original_input_pil, processed_output_np):
    """
    원본 이미지(PIL)와 처리된 이미지(Numpy)를 저장합니다.
    """
    global counter # 전역 counter 사용
    try:
        if processed_output_np is None:
             return "저장할 처리된 이미지가 없습니다."

        # 고유한 식별자 생성 (타임스탬프)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 저장 경로 표준화
        save_dir = os.path.join(OUTPUT_BASE_DIR, f"result_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        # 원본 이미지 저장 (업로드된 원본 PIL 이미지 사용)
        original_path = os.path.join(save_dir, "original.png")
        if original_input_pil is not None and isinstance(original_input_pil, Image.Image):
             original_input_pil.save(original_path)
        else:
             print("경고: 원본 PIL 이미지가 없어 저장하지 못했습니다.")

        # 처리된 이미지 저장 (Numpy 배열 -> PIL -> 저장)
        result_path = os.path.join(save_dir, "result.png")
        processed_image_pil = Image.fromarray(processed_output_np)
        processed_image_pil.save(result_path)

        # 결과 텍스트 저장
        count_text = counter.get_count_text()
        result_txt_path = os.path.join(save_dir, "count_results.txt")
        with open(result_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(count_text)
            # 추가 정보 (옵션)
            f.write(f"\n\n--- 상세 정보 ---\n")
            f.write(f"사용된 분석 방법: {counter.last_method}\n")
            f.write(f"자동 감지된 콜로니: {counter.auto_detected_count}\n")
            f.write(f"수동 추가된 콜로니: {len(counter.manual_points)}\n")

        # 요약 JSON 저장
        summary = {
            'datetime': datetime.now().isoformat(),
            'auto_count': counter.auto_detected_count,
            'manual_count': len(counter.manual_points),
            'total_count': counter.auto_detected_count + len(counter.manual_points),
            'analysis_method': counter.last_method
        }
        summary_json_path = os.path.join(save_dir, "summary.json")
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        return f"결과 저장 완료:\n{save_dir}"
    except Exception as e:
        error_msg = f"이미지 저장 중 오류가 발생했습니다: {str(e)}"
        print(error_msg)
        return error_msg

def handle_batch_upload(
    files: List[Any], # Gradio File 컴포넌트는 파일 객체 리스트 반환
    input_size: int,
    iou_threshold: float,
    conf_threshold: float,
    better_quality: bool,
    withContours: bool,
    # min_area_percentile: int,
    # max_area_percentile: int,
    # ---
    enable_area_filter: bool,
    min_area_percentile: int,
    max_area_percentile: int,
    # ---
    circularity_threshold: float,
    progress=gr.Progress()
) -> Tuple[str, List[Tuple[np.ndarray, str]]]: # 결과 메시지와 갤러리 데이터 반환
    """배치 모드에서 여러 이미지를 처리합니다."""
    global counter # 전역 counter 사용

    try:
        if not files:
            return "처리할 파일이 없습니다.", []

        # 출력 디렉토리 표준화
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_BASE_DIR, f"batch_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        results_summary = [] # CSV/JSON 요약용
        gallery_data = [] # Gradio 갤러리 표시용

        total_files = len(files)
        progress(0, desc="배치 처리 시작...")

        for idx, file_obj in enumerate(files):
            current_progress = (idx + 1) / total_files
            # file_obj.name이 None일 수 있는 경우 방지
            img_filename = os.path.basename(file_obj.name) if file_obj and file_obj.name else f"unknown_file_{idx+1}"
            progress(current_progress, desc=f"이미지 처리 중 ({idx + 1}/{total_files}): {img_filename}")

            # 배치 처리 시 상태 초기화
            counter.reset() # 각 이미지 처리 전 counter 상태 초기화

            try:
                # 파일 객체 유효성 검사
                if not file_obj or not hasattr(file_obj, 'name') or not file_obj.name:
                     raise ValueError("잘못된 파일 객체입니다.")

                # 이미지 로드 (파일 객체에서 직접 로드)
                image_pil = Image.open(file_obj.name).convert("RGB")

                # 이미지 분석 (segment_and_count_colonies 함수 재사용)
                processed_image_np, count_text = segment_and_count_colonies(
                    image_pil, # 원본 PIL 이미지 전달
                    input_size=input_size,
                    iou_threshold=iou_threshold,
                    conf_threshold=conf_threshold,
                    better_quality=better_quality,
                    withContours=withContours,
                    mask_random_color=True, # 배치에서는 랜덤 색상 고정
                    # min_area_percentile=min_area_percentile,
                    # max_area_percentile=max_area_percentile,
                    # ---
                    enable_area_filter=enable_area_filter,
                    min_area_percentile=min_area_percentile,
                    max_area_percentile=max_area_percentile,
                    # ---
                    circularity_threshold=circularity_threshold
                )

                # 개별 이미지 결과 저장
                img_name_base = Path(img_filename).stem
                img_save_dir = os.path.join(output_dir, f"{img_name_base}_result")
                os.makedirs(img_save_dir, exist_ok=True)

                # 원본 저장
                original_path = os.path.join(img_save_dir, "original.png")
                image_pil.save(original_path)
                # 결과 저장
                result_path = os.path.join(img_save_dir, "result.png")
                if processed_image_np is not None:
                    Image.fromarray(processed_image_np).save(result_path)
                else:
                    print(f"경고: {img_filename}의 처리된 이미지가 없어 저장할 수 없습니다.")
                # 텍스트 저장
                result_txt_path = os.path.join(img_save_dir, "count_results.txt")
                with open(result_txt_path, 'w', encoding='utf-8') as f:
                     f.write(f"파일: {img_filename}\n")
                     f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                     f.write(count_text)

                # 요약 정보 추가
                results_summary.append({
                    'filename': img_filename,
                    'total_count': counter.auto_detected_count + len(counter.manual_points),
                    'auto_count': counter.auto_detected_count,
                    'manual_count': len(counter.manual_points),
                    'status': 'success',
                    'output_path': img_save_dir
                })
                # 갤러리 데이터 추가
                if processed_image_np is not None:
                    gallery_data.append((processed_image_np, f"{img_filename}\n{count_text}"))
                else: # 처리 실패 시 원본 이미지와 함께 표시
                    gallery_data.append((np.array(image_pil), f"{img_filename}\n분석 실패"))


            except Exception as e:
                error_msg = f"오류 처리 중 {img_filename}: {str(e)}"
                print(error_msg)
                results_summary.append({
                    'filename': img_filename,
                    'total_count': 0, 'auto_count': 0, 'manual_count': 0,
                    'status': 'error', 'error_message': error_msg,
                    'output_path': None
                })
                # 오류 발생 시 갤러리에 원본 이미지와 에러 메시지 표시 (옵션)
                try:
                     error_img_np = np.array(Image.open(file_obj.name).convert("RGB"))
                     cv2.putText(error_img_np, "ERROR", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)
                     gallery_data.append((error_img_np, f"{img_filename}\n처리 오류"))
                except:
                     gallery_data.append((None, f"{img_filename}\n처리 오류")) # 이미지 로드도 실패 시

        # 전체 배치 요약 CSV/JSON 저장
        summary_df = pd.DataFrame(results_summary)
        csv_path = os.path.join(output_dir, "batch_summary.csv")
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        summary_json_path = os.path.join(output_dir, "batch_summary.json")
        summary_final = {
             'batch_start_time': timestamp,
             'total_images': total_files,
             'success_count': sum(1 for r in results_summary if r['status'] == 'success'),
             'error_count': sum(1 for r in results_summary if r['status'] == 'error'),
             'results': results_summary
        }
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_final, f, indent=4, ensure_ascii=False)

        # 처리 완료 메시지 생성
        success_count = summary_final['success_count']
        fail_count = summary_final['error_count']
        result_msg = f"배치 처리 완료: 총 {total_files}개 중 {success_count}개 성공, {fail_count}개 실패.\n"
        result_msg += f"결과 저장 폴더: {output_dir}\n"
        result_msg += f"요약 파일: batch_summary.csv, batch_summary.json"

        return result_msg, gallery_data

    except Exception as e:
        error_msg = f"배치 처리 중 심각한 오류 발생: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc() # 상세 오류 출력
        return error_msg, []


# --- UI 코드 ---

# CSS 스타일링 개선
css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

body {
    background-color: #f8fafc;
    font-family: 'Noto Sans KR', sans-serif;
    color: #334155;
    line-height: 1.4;
}
.container { max-width: 1280px; margin: 0 auto; padding: 20px; }
.header { background: linear-gradient(135deg, #2563EB, #1e40af); padding: 30px; border-radius: 12px; color: #ffffff; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2); display: flex; justify-content: space-between; align-items: center; }
.header-content { text-align: left; }
.header-date { text-align: right; color: rgba(255, 255, 255, 0.9); font-size: 14px; }
.header h1 { font-size: 32px; font-weight: 700; margin: 0; background: linear-gradient(90deg, #ffffff, #e2e8f0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); }
.header h3 { font-size: 16px; font-weight: 500; margin: 10px 0 0; color: #e2e8f0; }
.section-title { font-size: 18px; font-weight: 600; color: #1e40af; margin-bottom: 15px; display: flex; align-items: center; gap: 8px; }
.button-primary { background-color: #2563EB !important; color: #ffffff !important; border: none !important; padding: 12px 24px !important; border-radius: 8px !important; font-size: 14px !important; font-weight: 500 !important; cursor: pointer !important; transition: all 0.3s ease !important; box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important; }
.button-primary:hover { background-color: #1d4ed8 !important; box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3) !important; transform: translateY(-1px) !important; }
.button-secondary { background-color: #475569 !important; color: #ffffff !important; border: none !important; padding: 12px 24px !important; border-radius: 8px !important; font-size: 14px !important; font-weight: 500 !important; cursor: pointer !important; transition: all 0.3s ease !important; box-shadow: 0 2px 4px rgba(71, 85, 105, 0.2) !important; }
.button-secondary:hover { background-color: #334155 !important; box-shadow: 0 4px 6px rgba(71, 85, 105, 0.3) !important; transform: translateY(-1px) !important; }
.accordion-header { background-color: #f1f5f9 !important; color: #1e40af !important; padding: 15px !important; border-radius: 8px !important; font-weight: 500 !important; border: 1px solid #e2e8f0 !important; margin: 10px 0 !important; transition: all 0.3s ease !important; }
.accordion-header:hover { border-color: #2563EB !important; box-shadow: 0 2px 4px rgba(37, 99, 235, 0.1) !important; }
.input-image, .output-image { border: 2px solid #e2e8f0 !important; border-radius: 12px !important; padding: 10px !important; background: #ffffff !important; transition: all 0.3s ease !important; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important; margin: 10px 0 !important; width: 100% !important; height: 500px !important; object-fit: contain !important; display: flex !important; justify-content: center !important; align-items: center !important; } /* 패딩/마진 조정, 고정 높이 */
.input-image img, .output-image img { max-width: 100% !important; max-height: 100% !important; object-fit: contain !important; }
.input-image:hover, .output-image:hover { border-color: #2563EB !important; box-shadow: 0 4px 6px rgba(37, 99, 235, 0.1) !important; }
.result-text { background: #ffffff !important; border-left: 5px solid #10b981 !important; padding: 15px !important; border-radius: 8px !important; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important; font-size: 14px !important; line-height: 1.6 !important; margin: 15px 0 !important; color: #334155 !important; }
.gradio-container { background-color: #ffffff !important; border-radius: 12px !important; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important; padding: 20px !important; margin: 20px auto !important; }
.gradio-row { gap: 20px !important; }
.gradio-column { background: #ffffff !important; padding: 20px !important; border-radius: 12px !important; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important; }
.slider-container { background: #f1f5f9 !important; padding: 15px !important; border-radius: 8px !important; margin: 10px 0 !important; }
.slider { accent-color: #2563EB !important; }
.checkbox-container { background: #f1f5f9 !important; padding: 12px !important; border-radius: 8px !important; margin: 8px 0 !important; }
.checkbox { accent-color: #10b981 !important; }
.instruction-box { background-color: #f1f5f9 !important; border: 1px solid #e2e8f0 !important; border-radius: 12px !important; padding: 20px !important; margin-top: 30px !important; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important; }
.instruction-box h3 { color: #1e40af !important; font-size: 18px !important; font-weight: 600 !important; margin-bottom: 15px !important; }
.instruction-box p { color: #334155 !important; font-size: 14px !important; line-height: 1.6 !important; margin-bottom: 10px !important; }
.tabs { border-bottom: 1px solid #e2e8f0 !important; margin-bottom: 20px !important; }
.tab-nav { padding: 12px 24px !important; color: #475569 !important; font-weight: 500 !important; font-size: 14px !important; transition: all 0.3s ease !important; }
.tab-nav:hover { color: #2563EB !important; }
.tab-nav.selected { color: #2563EB !important; border-bottom: 2px solid #2563EB !important; background-color: rgba(37, 99, 235, 0.05) !important; }
.priority-high { color: #ef4444 !important; font-weight: 500 !important; }
.priority-medium { color: #f59e0b !important; font-weight: 500 !important; }
.priority-low { color: #10b981 !important; font-weight: 500 !important; }
.card { background: #ffffff !important; border-radius: 12px !important; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important; padding: 20px !important; margin: 15px 0 !important; border: 1px solid #e2e8f0 !important; transition: all 0.3s ease !important; }
.card:hover { box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important; transform: translateY(-2px) !important; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 4px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
.batch-gallery { min-height: 300px; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; background-color: #f8fafc;} /* 배치 갤러리 스타일 */
"""

# 전역 counter 객체
counter = ColonyCounter()

# Gradio 인터페이스 설정
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    # 원본 이미지를 저장하기 위한 숨겨진 상태 변수
    original_input_image_state = gr.State(None)

    gr.Markdown(
        """
        <div class="header">
            <div class="header-content">
                <h1>🔬 BDK CFU 카운터 (v5 수정)</h1>
                <h3>AI 자동 CFU 감지 및 수동 보정</h3>
            </div>
            <div class="header-date">
                <span>최종 업데이트: 2025년 4월</span>
            </div>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("단일 이미지 처리"):
            with gr.Row():
                # --- 입력 컬럼 ---
                with gr.Column(scale=1): # scale 조정 (1:1 비율)
                    gr.Markdown("<div class='section-title'>📁 이미지 업로드 & 전처리</div>")
                    input_image = gr.Image(
                        type="pil",
                        label="입력 이미지",
                        elem_classes=["input-image"],
                        show_label=False,
                        height=500, # 고정 높이
                    )

                    with gr.Accordion("🛠️ 이미지 전처리 설정", open=False, elem_classes="accordion-header"):
                        to_grayscale = gr.Checkbox(label="흑백 변환", value=False)
                        binary = gr.Checkbox(label="바이너리 변환", value=False)
                        binary_threshold = gr.Slider(0, 255, 128, step=1, label="바이너리 임계값")
                        edge_detection = gr.Checkbox(label="에지 검출", value=False)
                        sharpen = gr.Checkbox(label="샤픈", value=False)
                        sharpen_amount = gr.Slider(0.5, 5.0, 1.0, step=0.1, label="샤픈 강도")
                        with gr.Row():
                            preprocess_button = gr.Button("🔄 전처리 적용", elem_classes="button-secondary")
                            reset_button = gr.Button("↺ 원본 복원", elem_classes="button-secondary")
                            undo_button = gr.Button("↶ 실행 취소", elem_classes="button-secondary")

                    gr.Markdown("<div class='section-title' style='margin-top: 20px;'>⚙️ 분석 설정</div>")
                    with gr.Accordion("분석 파라미터", open=True, elem_classes="accordion-header"): # 기본 열림
                        input_size_slider = gr.Slider(512, 2048, 1024, step=64, label="입력 크기", info="클수록 정확, 느림")
                        iou_threshold_slider = gr.Slider(0.1, 0.9, 0.7, step=0.1, label="IOU 임계값", info="높을수록 엄격")
                        conf_threshold_slider = gr.Slider(0.1, 0.9, 0.25, step=0.05, label="신뢰도 임계값", info="높을수록 확실")
                        circularity_threshold_slider = gr.Slider(0.0, 1.0, 0.8, step=0.01, label="원형도 임계값", info="1에 가까울수록 원형")
                        withContours_checkbox = gr.Checkbox(label="윤곽선 표시", value=True, info="콜로니 경계 표시")
                        better_quality_checkbox = gr.Checkbox(label="향상된 품질 (미사용)", value=True, visible=False)
                        # --- 면적 필터링 UI 추가 ---
                        enable_area_filter_checkbox = gr.Checkbox(label="면적 필터링 사용", value=False, info="체크 시 아래 백분위 기준으로 콜로니 필터링")
                        min_area_percentile_slider = gr.Slider(0, 10, 1, step=1, label="최소 면적 (백분위 %)", visible=True) # visible=True로 변경
                        max_area_percentile_slider = gr.Slider(90, 100, 99, step=1, label="최대 면적 (백분위 %)", visible=True) # visible=True로 변경

                    segment_button = gr.Button(
                        "🔍 이미지 분석 실행",
                        variant="primary",
                        elem_classes="button-primary",
                    )

                # --- 출력 및 수정 컬럼 ---
                with gr.Column(scale=1): # scale 조정 (1:1 비율)
                    gr.Markdown("<div class='section-title'>📊 분석 결과 & 수동 수정</div>")
                    output_image = gr.Image(
                        type="numpy",
                        label="결과 이미지",
                        interactive=True, # 클릭 이벤트 활성화
                        elem_classes=["output-image"],
                        show_label=False,
                        height=500, # 고정 높이
                    )
                    colony_count_text = gr.Textbox(
                        label="카운트 결과",
                        lines=3, # 3줄로 늘림
                        elem_classes="result-text",
                        interactive=False # 직접 수정 불가
                    )

                    # --- 편집 모드 UI 개선 ---
                    # with gr.Row(): # 기존 코드 주석 처리
                    #     remove_mode_button = gr.Button("🔄 편집 모드 (추가/제거)", elem_classes="button-secondary")
                    #     remove_mode_text = gr.Textbox(label="현재 모드", value="🟢 ADD MODE", lines=1, interactive=False)
                    with gr.Row(variant="panel"): # 버튼과 모드 표시를 한 줄에 배치
                        remove_mode_button = gr.Button("🔄 편집 모드 전환", elem_classes="button-secondary") # scale 제거
                        remove_mode_indicator = gr.Markdown(value="<span style='color: green; font-weight: bold; padding: 10px; display: inline-block;'>🟢 ADD MODE</span>", elem_classes="mode-indicator") # scale 제거

                    with gr.Row():
                        remove_point_button = gr.Button("↩️ 마지막 수동 포인트 취소", elem_classes="button-secondary")

                    save_button = gr.Button("💾 결과 저장", elem_classes="button-primary")
                    save_output = gr.Textbox(label="저장 결과", lines=2, interactive=False, elem_classes="result-text")

            # --- 단일 처리 가이드 ---
            with gr.Row(elem_classes="instruction-box"):
                 gr.Markdown( # 가이드 내용 업데이트
                      """
                      <h3>📝 빠른 사용 가이드 (단일 이미지)</h3>
                      <p><span class="priority-high">1. 이미지 업로드:</span> 분석할 콜로니 이미지를 업로드하세요.</p>
                      <p><span class="priority-medium">2. (선택) 전처리:</span> 필요시 '이미지 전처리 설정'에서 옵션을 선택하고 '전처리 적용'을 누르세요. '원본 복원' 또는 '실행 취소'로 되돌릴 수 있습니다.</p>
                      <p><span class="priority-medium">3. (선택) 분석 설정:</span> '분석 파라미터'에서 AI 탐지 관련 설정을 조정하세요.</p>
                      <p><span class="priority-high">4. 이미지 분석 실행:</span> '이미지 분석 실행' 버튼을 클릭하세요.</p>
                      <p><span class="priority-medium">5. 수동 수정:</span></p>
                      <ul>
                          <li>결과 이미지 위에서 클릭하여 포인트를 추가하거나 제거할 수 있습니다.</li>
                          <li>'편집 모드' 버튼으로 추가(🟢 ADD)/제거(🔴 REMOVE) 모드를 전환하세요.</li>
                          <li>'마지막 수동 포인트 취소' 버튼으로 가장 최근에 **수동으로 추가**한 포인트를 제거합니다.</li>
                      </ul>
                       <p><span class="priority-low">6. 결과 저장:</span> '결과 저장' 버튼을 눌러 원본, 결과 이미지, 카운트 정보를 저장하세요.</p>
                      """
                 )

            # --- 이벤트 핸들러 연결 ---

            # 이미지 업로드 시 원본 이미지 상태 저장
            input_image.upload(
                 lambda img: (img, img), # 업로드된 이미지를 input_image와 상태 변수 모두에 설정
                 inputs=[input_image],
                 outputs=[input_image, original_input_image_state]
            ).then(
                 lambda: counter.reset(), # 새 이미지 업로드 시 counter 리셋
                 inputs=None,
                 outputs=None
            ).then(
                 lambda img: image_history.set_original(img), # 이미지 히스토리에도 설정
                 inputs=[input_image],
                 outputs=[input_image] # 히스토리 설정 후 input_image 업데이트 (크기 조정 등 반영)
            ).then(
                 lambda: "이미지가 업로드되었습니다. 분석을 진행하세요.", # 초기 메시지
                 outputs=[colony_count_text]
            )

            # 전처리 버튼
            preprocess_button.click(
                lambda img, gs, b, bt, edge, sh, sa: image_history.add_state(preprocess_image(img, to_grayscale=gs, binary=b, binary_threshold=bt, edge_detection=edge, sharpen=sh, sharpen_amount=sa)),
                inputs=[input_image, to_grayscale, binary, binary_threshold, edge_detection, sharpen, sharpen_amount],
                outputs=[input_image] # 전처리 후 input_image 업데이트
            )
            reset_button.click(
                lambda: image_history.reset(), # 원본으로 리셋
                inputs=[],
                outputs=[input_image]
            )
            undo_button.click(
                lambda: image_history.undo(), # 이전 단계로
                inputs=[],
                outputs=[input_image]
            )

            # 분석 버튼
            segment_button.click(
                segment_and_count_colonies,
                inputs=[
                    input_image, # 현재 표시된 (전처리된) 이미지 사용
                    input_size_slider,
                    iou_threshold_slider,
                    conf_threshold_slider,
                    better_quality_checkbox,
                    withContours_checkbox,
                    # --- SyntaxError 수정: 해당 라인 제거 ---
                    # mask_random_color=gr.Checkbox(value=True, visible=False),
                    # --------------------------------------
                    # min_area_percentile_slider, # 기존 슬라이더 제거
                    # max_area_percentile_slider, # 기존 슬라이더 제거
                    circularity_threshold_slider,
                    # --- 면적 필터링 UI 입력 추가 ---
                    enable_area_filter_checkbox,
                    min_area_percentile_slider,
                    max_area_percentile_slider,
                    # -------------------------------
                ],
                outputs=[output_image, colony_count_text]
            )

            # 수동 포인트 추가/제거 이벤트
            output_image.select(
                counter.add_or_remove_point,
                inputs=[output_image], # 현재 출력 이미지 전달 (좌표계 기준)
                outputs=[output_image, colony_count_text] # 업데이트된 이미지와 텍스트 반환
            )

            # 마지막 수동 포인트 제거 버튼
            remove_point_button.click(
                counter.remove_last_point,
                inputs=[output_image],
                outputs=[output_image, colony_count_text]
            )

            # 편집 모드 전환 버튼
            remove_mode_button.click(
                counter.toggle_remove_mode,
                inputs=[], # 입력 필요 없음
                outputs=[output_image, remove_mode_indicator] # 이미지와 모드 텍스트 업데이트
            )

            # 결과 저장 버튼
            save_button.click(
                save_results,
                inputs=[original_input_image_state, output_image], # 원본(State)과 결과(Image) 전달
                outputs=[save_output]
            )

        # --- 배치 처리 탭 ---
        with gr.Tab("배치 처리"):
            with gr.Row():
                # --- 배치 설정 컬럼 ---
                with gr.Column(scale=1):
                    gr.Markdown("<div class='section-title'>📁 배치 이미지 업로드 & 설정</div>")
                    batch_files = gr.File(
                        label="이미지 파일 선택 (여러 개 선택 가능)",
                        file_count="multiple",
                        file_types=["image"],
                        elem_classes=["card"]
                    )

                    with gr.Accordion("⚙️ 배치 처리 설정", open=True, elem_classes="accordion-header"):
                        batch_input_size = gr.Slider(512, 2048, 1024, step=64, label="입력 크기")
                        batch_iou_threshold = gr.Slider(0.1, 0.9, 0.7, step=0.1, label="IOU 임계값")
                        batch_conf_threshold = gr.Slider(0.1, 0.9, 0.25, step=0.05, label="신뢰도 임계값")
                        batch_circularity = gr.Slider(0.0, 1.0, 0.8, step=0.01, label="원형도 임계값")
                        batch_withContours = gr.Checkbox(label="윤곽선 표시", value=True)
                        batch_better_quality = gr.Checkbox(label="향상된 품질 (미사용)", value=True, visible=False)
                        # --- 배치 면적 필터링 UI 추가 ---
                        batch_enable_area_filter = gr.Checkbox(label="면적 필터링 사용", value=False)
                        batch_min_area = gr.Slider(0, 10, 1, step=1, label="최소 면적 (백분위 %)", visible=True) # visible=True로 변경
                        batch_max_area = gr.Slider(90, 100, 99, step=1, label="최대 면적 (백분위 %)", visible=True) # visible=True로 변경

                    batch_process_button = gr.Button(
                        "🚀 배치 처리 시작",
                        variant="primary",
                        elem_classes="button-primary"
                    )

                # --- 배치 결과 컬럼 ---
                with gr.Column(scale=1):
                    gr.Markdown("<div class='section-title'>📊 처리 결과 요약 & 갤러리</div>")
                    batch_summary = gr.Textbox(
                        label="처리 결과 요약",
                        lines=8, # 줄 수 조정
                        elem_classes="result-text",
                        interactive=False
                    )
                    gr.Textbox(label="결과 저장 기본 폴더", value=OUTPUT_BASE_DIR, interactive=False)
                    open_output_button = gr.Button("📂 기본 결과 폴더 열기", elem_classes="button-secondary")
                    batch_gallery = gr.Gallery(
                         label="처리 결과 이미지", show_label=False, elem_classes=["batch-gallery"], columns=4, height=400
                    )

            # --- 배치 처리 가이드 ---
            with gr.Row(elem_classes="instruction-box"):
                 gr.Markdown( # 가이드 내용 업데이트
                      """
                      <h3>📝 배치 처리 가이드</h3>
                      <p><span class="priority-high">1. 이미지 파일 선택:</span> 분석할 여러 이미지 파일을 한 번에 선택하세요.</p>
                      <p><span class="priority-medium">2. (선택) 배치 처리 설정:</span> 필요에 따라 분석 파라미터를 조정하세요.</p>
                      <p><span class="priority-high">3. 배치 처리 시작:</span> '배치 처리 시작' 버튼을 클릭하세요.</p>
                      <p><span class="priority-medium">4. 결과 확인:</span></p>
                      <ul>
                          <li>처리 진행 상황과 최종 요약 메시지가 '처리 결과 요약' 창에 표시됩니다.</li>
                          <li>처리된 이미지 미리보기가 아래 갤러리에 나타납니다.</li>
                          <li>모든 결과(개별 이미지 폴더, 요약 CSV/JSON)는 '결과 저장 기본 폴더' 아래의 `batch_[타임스탬프]` 폴더에 저장됩니다.</li>
                          <li>'기본 결과 폴더 열기' 버튼으로 해당 폴더를 바로 열 수 있습니다.</li>
                      </ul>
                      """
                 )

            # --- 배치 처리 이벤트 핸들러 ---
            batch_process_button.click(
                handle_batch_upload,
                inputs=[
                    batch_files,
                    batch_input_size,
                    batch_iou_threshold,
                    batch_conf_threshold,
                    batch_better_quality,
                    batch_withContours,
                    # batch_min_area, # 기존 슬라이더 제거
                    # batch_max_area, # 기존 슬라이더 제거
                    batch_circularity,
                    # --- 배치 면적 필터링 UI 입력 추가 ---
                    batch_enable_area_filter,
                    batch_min_area,
                    batch_max_area,
                    # -------------------------------
                ],
                outputs=[batch_summary, batch_gallery] # 요약 텍스트와 갤러리 데이터 반환
            )

            # 결과 폴더 열기 로직
            def open_folder(folder_path):
                try:
                    abs_folder_path = os.path.abspath(folder_path)
                    if not os.path.isdir(abs_folder_path):
                         return f"오류: 폴더를 찾을 수 없습니다 - {abs_folder_path}"

                    if sys.platform == "win32":
                        os.startfile(abs_folder_path)
                    elif sys.platform == "darwin": # macOS
                        webbrowser.open(f"file://{abs_folder_path}")
                    else: # Linux 등
                        webbrowser.open(f"file://{abs_folder_path}")
                    return f"폴더 열기 시도: {abs_folder_path}"
                except Exception as e:
                    return f"폴더 열기 오류: {str(e)}"

            open_output_button.click(
                 lambda: open_folder(OUTPUT_BASE_DIR), # 기본 결과 폴더 열기
                 inputs=[],
                 outputs=[batch_summary] # 상태 메시지를 요약 창에 표시
            )

if __name__ == "__main__":
    # 공유 옵션 등 추가 가능
    demo.launch(share=False)

