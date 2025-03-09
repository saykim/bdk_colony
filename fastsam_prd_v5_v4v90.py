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
    """콜로니 카운팅 및 포인트 관리 클래스"""
    def __init__(self):
        """초기화"""
        self.auto_points = []  # 자동 감지된 포인트 (중심점 좌표)
        self.manual_points = []  # 수동으로 추가된 포인트
        self.removed_points = []  # 제거된 포인트 (실행 취소용)
        self.auto_detected_count = 0  # 자동 감지된 콜로니 수
        self.original_image = None  # 원본 이미지
        self.current_image = None  # 현재 이미지 (포인트가 그려진 상태)
        self.colony_masks = None  # 콜로니 마스크 리스트
        self.dish_mask = None  # 배양접시 마스크
        self.base_image = None  # 처리된 베이스 이미지
        self.remove_mode = False  # 제거 모드 여부 (True: 클릭 시 포인트 제거, False: 클릭 시 포인트 추가)
        self.scale_factor = 1.0  # 이미지 스케일 팩터
        self.last_method = None  # 마지막으로 사용한 분석 방법

    def reset(self):
        """
        카운터 초기화
        """
        self.current_image = None
        self.auto_points = []
        self.manual_points = []
        self.auto_detected_count = 0
        self.remove_mode = False
        self.original_image = None
        self.scale_factor = 1.0  # 확대/축소 비율 초기화
        self.colony_masks = []  # 콜로니 마스크 초기화
        self.dish_mask = None  # 배양접시 마스크 초기화
        self.base_image = None  # 처리된 베이스 이미지 초기화

    def set_original_image(self, image):
        """
        원본 이미지 설정

        Args:
            image (PIL.Image): 원본 이미지
        """
        if image is not None:
            self.original_image = image
            if isinstance(image, Image.Image):
                self.original_image = np.array(image)
            else:
                self.original_image = image.copy()

    def toggle_remove_mode(self):
        """편집 모드 전환 (추가/제거 모드)"""
        self.remove_mode = not self.remove_mode
        img_with_points = self.draw_points()
        mode_text = "🔴 REMOVE MODE" if self.remove_mode else "🟢 ADD MODE"
        return img_with_points, mode_text

    def set_segmentation_data(self, colony_masks, dish_mask, processed_image):
        """
        세그멘테이션 데이터 설정 (콜로니 마스크, 접시 마스크, 처리된 이미지)

        Args:
            colony_masks: 콜로니 마스크 리스트
            dish_mask: 배양접시 마스크
            processed_image: 처리된 이미지
        """
        self.colony_masks = colony_masks
        self.dish_mask = dish_mask
        self.base_image = processed_image.copy() if processed_image is not None else None

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
                    # 자동 포인트와 해당 마스크 제거
                    self.auto_points.pop(closest_auto)
                    if closest_auto < len(self.colony_masks):
                        self.colony_masks.pop(closest_auto)
                    self.auto_detected_count = len(self.auto_points)
                    removed = True
                elif closest_manual is not None:
                    # 수동 포인트 제거
                    self.manual_points.pop(closest_manual)
                    removed = True

                if removed:
                    # 세그멘테이션 이미지 다시 그리기
                    self.redraw_segmentation()

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

    def redraw_segmentation(self):
        """
        세그멘테이션 이미지 다시 그리기
        """
        if self.original_image is None or self.base_image is None:
            return
            
        # 원본 이미지 복사
        image_np = np.array(self.original_image).copy()
        
        # 콜로니 마스크 처리
        for mask in self.colony_masks:
            mask_np = mask.cpu().numpy()
            if mask_np.ndim == 2:
                mask_np = mask_np > 0
                # 랜덤 색상 적용
                color = np.random.randint(100, 200, (3,)).tolist()
                image_np[mask_np] = color
                
                # 윤곽선 그리기
                contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_np, contours, -1, (255, 0, 0), 2)
        
        # 배양접시 마스크 처리
        if self.dish_mask is not None:
            dish_mask_np = self.dish_mask.cpu().numpy()
            if dish_mask_np.ndim == 2:
                dish_mask_np = dish_mask_np > 0
                contours, _ = cv2.findContours(dish_mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_np, contours, -1, (0, 0, 255), 3)
        
        # 현재 이미지 업데이트
        self.current_image = image_np

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
        마지막으로 추가된 수동 포인트 제거

        Args:
            image (numpy.ndarray): 현재 이미지

        Returns:
            tuple: 업데이트된 이미지와 카운트 텍스트
        """
        try:
            # 수동 포인트만 제거 (자동 감지된 포인트는 제거하지 않음)
            if len(self.manual_points) > 0:
                removed_point = self.manual_points.pop()
                self.removed_points.append(removed_point)  # 제거된 포인트 저장
                
                # 변경 내용 반영하여 이미지 다시 그리기
                updated_image = self.draw_points()
                
                return updated_image, self.get_count_text()
            else:
                # 수동 포인트가 없으면 아무 변화 없음
                return image, self.get_count_text()
        except Exception as e:
            print(f"포인트 제거 중 오류 발생: {str(e)}")
            return image, self.get_count_text()

    def undo_last_removal(self, image):
        """
        마지막으로 제거된 포인트 복원
        
        Args:
            image (numpy.ndarray): 현재 이미지
            
        Returns:
            tuple: 업데이트된 이미지와 카운트 텍스트
        """
        try:
            if len(self.removed_points) > 0:
                # 마지막으로 제거된 포인트 가져오기
                restored_point = self.removed_points.pop()
                
                # 원래의 카테고리(자동/수동)에 따라 복원
                # 수동 포인트는 계산 방식으로 구분 (자동 포인트 수보다 큰 번호는 수동 포인트)
                if len(self.auto_points) >= self.auto_detected_count:
                    # 자동 감지 포인트로 복원
                    self.auto_points.append(restored_point)
                    self.auto_detected_count += 1
                else:
                    # 수동 추가 포인트로 복원
                    self.manual_points.append(restored_point)
                
                # 변경 내용 반영하여 이미지 다시 그리기
                updated_image = self.draw_points()
                
                return updated_image, self.get_count_text()
            else:
                return image, self.get_count_text()
        except Exception as e:
            print(f"포인트 복원 중 오류 발생: {str(e)}")
            return image, self.get_count_text()

    def get_count_text(self):
        """
        현재 카운트 텍스트 가져오기

        Returns:
            str: 카운트 텍스트
        """
        auto_count = self.auto_detected_count
        manual_count = len(self.manual_points)
        total_count = auto_count + manual_count
        
        # 방법 정보 제거 (FastSAM 표시 안함)
        text = f"총 콜로니 수: {total_count}\n(자동: {auto_count}, 수동: {manual_count})"
        return text

    def draw_points(self):
        """
        현재 이미지에 포인트를 그려서 반환

        Returns:
            numpy.ndarray: 포인트가 그려진 이미지
        """
        try:
            if self.current_image is None:
                return None

            # 이미지 복사 및 오버레이 초기화
            img_with_points = self.current_image.copy()
            overlay = np.zeros_like(img_with_points)
            
            # 사용자 설정 가능한 변수들
            # 1. 마커 크기 및 폰트 설정
            square_size = 40  # 수동 포인트의 사각형 크기
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 1
            outline_thickness = 3
            
            # 2. 색상 설정 (B, G, R 형식)
            AUTO_TEXT_COLOR = (255, 255, 255)  # 자동 감지 텍스트 색상 (흰색)
            AUTO_OUTLINE_COLOR = (0, 0, 0)     # 자동 감지 외곽선 색상 (검은색)
            MANUAL_RECT_COLOR = (0, 0, 255)    # 수동 추가 사각형 색상 (빨간색)
            MANUAL_BORDER_COLOR = (0, 0, 0)    # 수동 추가 테두리 색상 (검은색)
            MANUAL_TEXT_COLOR = (255, 255, 0)  # 수동 추가 텍스트 색상 (노란색)
            
            # 3. 투명도 설정
            OVERLAY_OPACITY = 0.6  # 오버레이 투명도
            REMOVE_MODE_OPACITY = 0.3  # 제거 모드 배너 투명도

            # 1. 자동 감지된 CFU 번호 표시
            for idx, (x, y) in enumerate(self.auto_points, 1):
                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

                text_x = int(x - text_width / 2)
                text_y = int(y - 10)

                # 8방향 검은색 외곽선으로 텍스트 가시성 향상
                for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    cv2.putText(img_with_points, text,
                              (text_x + dx, text_y + dy),
                              font, font_scale, AUTO_OUTLINE_COLOR, outline_thickness)

                # 흰색 텍스트를 외곽선 위에 그리기
                cv2.putText(img_with_points, text,
                          (text_x, text_y),
                          font, font_scale, AUTO_TEXT_COLOR, font_thickness)

            # 2. 수동으로 추가된 포인트 표시
            manual_font_scale = 1.0  # 폰트 크기 증가
            manual_font_thickness = 2  # 폰트 두께 증가
            
            for idx, (x, y) in enumerate(self.manual_points, len(self.auto_points) + 1):
                # 사각형 좌표 계산
                pt1 = (int(x - square_size / 2), int(y - square_size / 2))
                pt2 = (int(x + square_size / 2), int(y + square_size / 2))
                
                # 반투명 사각형 그리기
                cv2.rectangle(overlay, pt1, pt2, MANUAL_RECT_COLOR, -1)  # 색상 채우기
                cv2.rectangle(overlay, pt1, pt2, MANUAL_BORDER_COLOR, 2)  # 테두리
                
                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, manual_font_scale, manual_font_thickness)
                text_x = int(x - text_width / 2)
                text_y = int(y - 15)  # 텍스트 위치 약간 위로 조정

                # 8방향 검은색 외곽선
                for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    cv2.putText(img_with_points, text,
                              (text_x + dx, text_y + dy),
                              font, manual_font_scale, AUTO_OUTLINE_COLOR, outline_thickness)

                # 노란색 텍스트 (더 눈에 띄게)
                cv2.putText(img_with_points, text,
                          (text_x, text_y),
                          font, manual_font_scale, MANUAL_TEXT_COLOR, manual_font_thickness)

            # 오버레이 이미지를 원본과 블렌딩
            cv2.addWeighted(overlay, OVERLAY_OPACITY, img_with_points, 1.0, 0, img_with_points)

            # 3. 제거 모드 표시
            if self.remove_mode:
                # 상단에 빨간색 배너 추가
                overlay_mode = img_with_points.copy()
                cv2.rectangle(overlay_mode, (0, 0), (img_with_points.shape[1], 50), (255, 0, 0), -1)
                cv2.addWeighted(overlay_mode, REMOVE_MODE_OPACITY, img_with_points, 0.7, 0, img_with_points)
                cv2.putText(img_with_points, "REMOVE MODE", (10, 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            else:
                overlay_mode = img_with_points.copy()
                cv2.rectangle(overlay_mode, (0, 0), (img_with_points.shape[1], 50), (0, 255, 0), -1)
                cv2.addWeighted(overlay_mode, REMOVE_MODE_OPACITY, img_with_points, 0.7, 0, img_with_points)
                cv2.putText(img_with_points, "ADD MODE", (10, 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            # 전체 카운트 표시 (왼쪽 하단)
            count_text = self.get_count_text()
            
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
            print(f"이미지에 포인트를 그리는 중 오류가 발생했습니다: {str(e)}")
            if self.current_image is not None:
                return self.current_image
            return None

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
    try:
        image_np = np.array(image).copy()

        # CFU 마스크 처리
        for ann in colony_annotations:
            # 차원 확인 및 처리
            ann_cpu = ann.cpu().numpy()
            if ann_cpu.ndim == 3 and ann_cpu.shape[0] == 1:
                ann_cpu = ann_cpu[0]  # (1, H, W) -> (H, W)
            
            mask = ann_cpu > 0
            if mask.ndim == 2:
                if mask_random_color:
                    color = np.random.randint(100, 200, (3,)).tolist()  # 중간 밝기의 랜덤 색상 생성
                    image_np[mask] = color
                else:
                    image_np[mask] = (0, 255, 0)  # 기본 녹색

            if withContours:
                # 윤곽선 찾기 및 그리기
                contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                cv2.drawContours(image_np, contours, -1, (255, 0, 0), 2)  # 파란색 윤곽선

        # 배양접시 마스크 처리 (윤곽선만 그리기)
        if dish_annotation is not None:
            dish_mask = dish_annotation.cpu().numpy()
            # 차원 확인 및 처리
            if dish_mask.ndim == 3 and dish_mask.shape[0] == 1:
                dish_mask = dish_mask[0]
            if dish_mask.ndim == 2:
                dish_mask = dish_mask > 0
                if withContours:
                    contours = cv2.findContours(dish_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]
                    cv2.drawContours(image_np, contours, -1, (0, 0, 255), 3)  # 빨간색 윤곽선

        processed_image = Image.fromarray(image_np)
        return processed_image
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return image

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
        input_size (int, optional): 입력 크기. Defaults to 1024.
        iou_threshold (float, optional): IOU 임계값. Defaults to 0.7.
        conf_threshold (float, optional): 신뢰도 임계값. Defaults to 0.25.
        better_quality (bool, optional): 향상된 품질 여부. Defaults to False.
        withContours (bool, optional): 윤곽선 표시 여부. Defaults to True.
        mask_random_color (bool, optional): 마스크에 랜덤 색상 적용 여부. Defaults to True.
        min_area_percentile (int, optional): 최소 면적 백분위수. Defaults to 1.
        max_area_percentile (int, optional): 최대 면적 백분위수. Defaults to 99.
        circularity_threshold (float, optional): 원형도 임계값. Defaults to 0.8.
        progress (gr.Progress, optional): 진행 상태 표시기. Defaults to gr.Progress().

    Returns:
        tuple: 처리된 이미지와 카운트 텍스트
    """
    try:
        if input_image is None:
            return None, "이미지를 업로드해주세요."

        progress(0.1, desc="이미지 준비 중...")
        # 이미지 크기 조정
        image_to_use = input_image
        original_width, original_height = image_to_use.size
        w, h = image_to_use.size
        scale = input_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        input_resized = image_to_use.resize((new_w, new_h))
        
        # 스케일 팩터 저장
        counter.scale_factor = scale
        # 사용된 분석 방법 저장
        counter.last_method = "FastSAM"

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

        # 안전한 마스크 데이터 접근
        annotations = getattr(results[0].masks, 'data', None)
        if annotations is None or len(annotations) == 0:
            return np.array(input_resized), "CFU가 감지되지 않았습니다."

        progress(0.6, desc="마스크 리사이즈 중...")
        # 각 마스크 면적 계산
        areas = [np.sum(ann.cpu().numpy() > 0) for ann in annotations]

        # 가장 큰 마스크를 배양접시로 인식
        dish_idx = np.argmax(areas)
        dish_annotation = annotations[dish_idx]
        colony_annotations = [ann for idx, ann in enumerate(annotations) if idx != dish_idx]

        # 면적과 원형도 기반 필터링
        valid_colony_annotations = []
        for ann in colony_annotations:
            ann_cpu = ann.cpu().numpy()
            if ann_cpu.ndim == 3 and ann_cpu.shape[0] == 1:
                ann_cpu = ann_cpu[0]
            mask = ann_cpu > 0
            area = np.sum(mask)

            # 원형도 계산
            contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            if contours and len(contours) > 0:
                perimeter = cv2.arcLength(contours[0], True)
                circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
                if circularity >= circularity_threshold:
                    valid_colony_annotations.append(ann)

        progress(0.8, desc="결과 처리 중...")
        # 결과 시각화
        fig = process_image(
            colony_annotations=valid_colony_annotations if valid_colony_annotations else colony_annotations,
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

        # 원본 이미지 저장
        counter.set_original_image(input_resized)
        
        # 세그멘테이션 데이터 저장
        counter.set_segmentation_data(valid_colony_annotations if valid_colony_annotations else colony_annotations, dish_annotation, np.array(fig))
        
        counter.current_image = np.array(fig)
        counter.auto_detected_count = len(valid_colony_annotations) if valid_colony_annotations else len(colony_annotations)

        # 자동 감지된 CFU의 중심점 계산
        counter.auto_points = []
        for ann in (valid_colony_annotations if valid_colony_annotations else colony_annotations):
            ann_cpu = ann.cpu().numpy()
            if ann_cpu.ndim == 3 and ann_cpu.shape[0] == 1:
                ann_cpu = ann_cpu[0]
            mask = ann_cpu > 0
            
            y_indices, x_indices = np.where(mask)
            if len(x_indices) > 0 and len(y_indices) > 0:
                center_x = int(np.mean(x_indices))
                center_y = int(np.mean(y_indices))
                counter.auto_points.append((center_x, center_y))

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
    except Exception as e:
        error_msg = f"분석 중 오류가 발생했습니다: {str(e)}"
        print(error_msg)
        # 오류 발생 시 원본 이미지 반환
        if input_image is not None:
            return np.array(input_image), error_msg
        return None, error_msg

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{timestamp}_{os.getpid()}"
        
        # 결과 저장 폴더 생성
        save_dir = os.path.join(output_dir, f"colony_count_{unique_id}")
        os.makedirs(save_dir, exist_ok=True)

        # 파일 저장
        original_path = os.path.join(save_dir, "original.png")
        result_path = os.path.join(save_dir, "result.png")
        
        # 원본 이미지 저장
        if isinstance(original_image, np.ndarray):
            cv2.imwrite(original_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        else:
            original_image.save(original_path)
        
        # 처리된 이미지 저장
        if isinstance(processed_image, np.ndarray):
            cv2.imwrite(result_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        else:
            processed_image_pil = Image.fromarray(processed_image)
            processed_image_pil.save(result_path)
        
        # 결과 텍스트 저장
        count_text = counter.get_count_text()
        result_txt = os.path.join(save_dir, "count_results.txt")
        with open(result_txt, 'w') as f:
            f.write(count_text)
            f.write(f"\n\n분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            f.write(f"\n자동 감지된 콜로니: {counter.auto_detected_count}")
            f.write(f"\n수동 추가된 콜로니: {len(counter.manual_points)}")
        
        # 요약 JSON 저장
        summary = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'time': datetime.now().strftime("%H:%M:%S"),
            'auto_count': counter.auto_detected_count,
            'manual_count': len(counter.manual_points),
            'total_count': counter.auto_detected_count + len(counter.manual_points)
            # 분석 방법 정보 제거
        }
        
        summary_json_path = os.path.join(save_dir, "summary.json")
        with open(summary_json_path, 'w') as f:
            json.dump(summary, f, indent=4)

        return f"저장된 결과:\n- 결과 폴더: {save_dir}\n- 원본 이미지: {original_path}\n- 처리된 이미지: {result_path}\n- 결과 텍스트: {result_txt}\n- 요약 JSON: {summary_json_path}"
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
    배치 모드에서 여러 이미지를 처리합니다.

    Args:
        files (List[str]): 이미지 파일 경로 리스트
        input_size (int): 입력 크기
        iou_threshold (float): IOU 임계값
        conf_threshold (float): 신뢰도 임계값
        better_quality (bool): 향상된 품질 여부
        withContours (bool): 윤곽선 표시 여부
        min_area_percentile (int): 최소 면적 백분위수
        max_area_percentile (int): 최대 면적 백분위수
        circularity_threshold (float): 원형도 임계값
        progress (gr.Progress, optional): 진행 상태 표시기. Defaults to gr.Progress().

    Returns:
        Tuple[str, str]: 처리 결과 메시지와 갤러리 경로
    """
    try:
        if not files:
            return "처리할 파일이 없습니다.", None

        # 출력 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("outputs", f"batch_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # 결과 저장 리스트
        results = []
        gallery_paths = []
        gallery_captions = []

        total_files = len(files)
        for idx, file_path in enumerate(files, 1):
            progress_value = (idx - 1) / total_files
            progress(progress_value, desc=f"이미지 {idx}/{total_files} 처리 중...")
            
            try:
                # 이미지 로드
                img_filename = os.path.basename(file_path)
                image = Image.open(file_path).convert("RGB")
                
                # 이미지 분석
                input_size_value = int(input_size)
                w, h = image.size
                scale = input_size_value / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                input_resized = image.resize((new_w, new_h))
                
                # FastSAM 모델 예측
                results_predict = model.predict(
                    input_resized,
                    device=device,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=input_size_value,
                    retina_masks=True
                )
                
                # 안전한 마스크 데이터 접근
                annotations = getattr(results_predict[0].masks, 'data', None)
                if annotations is None or len(annotations) == 0:
                    results.append({
                        'filename': img_filename,
                        'count_text': "CFU가 감지되지 않았습니다.",
                        'auto_count': 0,
                        'manual_count': 0,
                        'total_count': 0,
                        'error': "마스크 데이터 없음"
                    })
                    continue
                
                # 각 마스크 면적 계산
                areas = [np.sum(ann.cpu().numpy() > 0) for ann in annotations]
                
                # 가장 큰 마스크를 배양접시로 인식
                dish_idx = np.argmax(areas)
                dish_annotation = annotations[dish_idx]
                colony_annotations = [ann for idx, ann in enumerate(annotations) if idx != dish_idx]
                
                # 면적과 원형도 기반 필터링
                valid_colony_annotations = []
                for ann in colony_annotations:
                    ann_cpu = ann.cpu().numpy()
                    if ann_cpu.ndim == 3 and ann_cpu.shape[0] == 1:
                        ann_cpu = ann_cpu[0]
                    mask = ann_cpu > 0
                    area = np.sum(mask)
                    
                    # 원형도 계산
                    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]
                    if contours and len(contours) > 0:
                        perimeter = cv2.arcLength(contours[0], True)
                        circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
                        if circularity >= circularity_threshold:
                            valid_colony_annotations.append(ann)
                
                # 결과 시각화
                fig = process_image(
                    colony_annotations=valid_colony_annotations if valid_colony_annotations else colony_annotations,
                    dish_annotation=dish_annotation,
                    image=input_resized,
                    device=device,
                    scale=1.0,
                    better_quality=better_quality,
                    mask_random_color=True,  # 항상 랜덤 색상 사용
                    bbox=None,
                    use_retina=False,
                    withContours=withContours
                )
                
                # 이미지 저장을 위한 폴더 생성
                img_name = os.path.splitext(img_filename)[0]
                img_save_dir = os.path.join(output_dir, img_name)
                os.makedirs(img_save_dir, exist_ok=True)
                
                # 원본 및 결과 이미지 저장
                original_path = os.path.join(img_save_dir, "original.png")
                result_path = os.path.join(img_save_dir, "result.png")
                image.save(original_path)
                fig.save(result_path)
                
                # 결과 텍스트 생성 및 저장
                auto_count = len(valid_colony_annotations) if valid_colony_annotations else len(colony_annotations)
                count_text = f"총 콜로니 수: {auto_count}\n(자동: {auto_count}, 수동: 0)"
                
                result_txt = os.path.join(img_save_dir, "count_results.txt")
                with open(result_txt, 'w') as f:
                    f.write(count_text)
                    f.write(f"\n\n분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    f.write(f"\n자동 감지된 콜로니: {auto_count}")
                    f.write(f"\n수동 추가된 콜로니: 0")
                
                # 결과 정보 저장
                result_info = {
                    'filename': img_filename,
                    'count_text': count_text,
                    'auto_count': auto_count,
                    'manual_count': 0,
                    'total_count': auto_count,
                    'path': result_path,
                    'original_path': original_path,
                    'result_dir': img_save_dir
                }
                
                results.append(result_info)
                gallery_paths.append(result_path)
                gallery_captions.append(f"{img_filename}: {auto_count}개 CFU")
                
            except Exception as e:
                error_msg = str(e)
                print(f"이미지 처리 중 오류 발생: {img_filename} - {error_msg}")
                results.append({
                    'filename': img_filename,
                    'count_text': f"Error: {error_msg}",
                    'auto_count': 0,
                    'manual_count': 0,
                    'total_count': 0,
                    'error': error_msg
                })
        
        # 요약 데이터 생성
        summary = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'time': datetime.now().strftime("%H:%M:%S"),
            'total_images': len(files),
            'successful': sum(1 for r in results if 'error' not in r),
            'failed': sum(1 for r in results if 'error' in r),
            'results': results
        }
        
        # 요약 JSON 저장
        summary_json_path = os.path.join(output_dir, "summary.json")
        with open(summary_json_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # 요약 CSV 저장
        df = pd.DataFrame([
            {
                'filename': r['filename'], 
                'total_count': r.get('total_count', 0),
                'auto_count': r.get('auto_count', 0),
                'manual_count': r.get('manual_count', 0),
                'error': r.get('error', '')
            } for r in results
        ])
        csv_path = os.path.join(output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        
        # 처리 완료 메시지
        success_count = summary['successful']
        fail_count = summary['failed']
        total_count = summary['total_images']
        
        result_msg = f"처리 완료 ({success_count}/{total_count} 성공, {fail_count}/{total_count} 실패)\n"
        result_msg += f"출력 폴더: {output_dir}\n"
        result_msg += f"요약 파일:\n- {summary_json_path}\n- {csv_path}"
        
        # 갤러리 생성
        progress(1.0, desc="완료!")
        return result_msg, gallery_paths
        
    except Exception as e:
        error_msg = f"배치 처리 중 오류가 발생했습니다: {str(e)}"
        print(error_msg)
        return error_msg, None

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
    width: 100% !important;
    height: auto !important;
    aspect-ratio: 1/1 !important;
    object-fit: contain !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

.input-image img, .output-image img {
    max-width: 100% !important;
    max-height: 100% !important;
    object-fit: contain !important;
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
                with gr.Column(scale=5, min_width=300):
                    gr.Markdown("<div class='section-title'>📁 이미지 업로드</div>")
                    input_image = gr.Image(
                        type="pil",
                        elem_classes=["input-image"],
                        show_label=False,
                        height=500,  # 고정 높이 설정
                        width=500    # 고정 너비 설정
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

                with gr.Column(scale=5, min_width=300):
                    gr.Markdown("<div class='section-title'>📊 분석 결과</div>")
                    output_image = gr.Image(
                        type="numpy",
                        interactive=True,
                        elem_classes=["output-image"],
                        show_label=False,
                        height=500,  # 고정 높이 설정
                        width=500    # 고정 너비 설정
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
                        try:
                            # PIL 9.0.0 이상
                            resampling_filter = Image.Resampling.LANCZOS
                        except AttributeError:
                            # PIL 9.0.0 미만
                            resampling_filter = Image.LANCZOS
                        image = image.resize((new_w, new_h), resampling_filter)
                    
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
