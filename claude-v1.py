import os
import sys
import warnings
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
import webbrowser
import platform
import subprocess
import gc

# 경고 무시
warnings.filterwarnings("ignore", category=FutureWarning)

# 기본 출력 디렉토리 설정
DEFAULT_OUTPUT_DIR = os.path.join(str(Path.home()), 'colony_counter_results')
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# 모델 경로 및 존재 확인 (v6 안정성)
model_path = 'weights/FastSAM-x.pt'
if not os.path.exists(model_path):
    print(f"오류: 모델 파일 '{model_path}'를 찾을 수 없습니다.")
    print("FastSAM-x.pt 파일을 weights 폴더에 넣어주세요.")
    sys.exit(1)

# AI 기반 분할 모델 로드
try:
    model = YOLO(model_path)
    print(f"모델 로드 성공: {model_path}")
except Exception as e:
    print(f"모델 로드 실패: {str(e)}")
    sys.exit(1)

# 디바이스 설정
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"사용 중인 디바이스: {device}")


class MemoryManager:
    """메모리 관리 유틸리티"""
    @staticmethod
    def clear_gpu_memory():
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    @staticmethod
    def limit_image_size(image, max_dimension=4096):
        """이미지 크기 제한"""
        if hasattr(image, 'size'):
            w, h = image.size
            if max(w, h) > max_dimension:
                scale = max_dimension / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                return image.resize((new_w, new_h), Image.LANCZOS)
        return image


class GPUMemoryMonitor:
    """GPU 메모리 모니터링"""
    def __init__(self, threshold_gb=1.0):
        self.threshold_gb = threshold_gb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_free_memory(self):
        """여유 GPU 메모리 확인 (GB)"""
        if not torch.cuda.is_available():
            return float('inf')
        return torch.cuda.mem_get_info()[0] / 1024**3
    
    def check_memory(self):
        """메모리 체크 및 정리"""
        if self.get_free_memory() < self.threshold_gb:
            self.cleanup()
    
    def cleanup(self):
        """강제 메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


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
        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except Exception as e:
                print(f"Error converting image: {str(e)}")
                return None
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


def calculate_circularity_robust(mask):
    """향상된 원형도 계산"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return 0.0
    
    # 모든 외부 contour 병합 (구멍 있는 콜로니 대응)
    if len(contours) > 1:
        # 가장 큰 contour 찾기
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
    else:
        area = cv2.contourArea(contours[0])
        perimeter = cv2.arcLength(contours[0], True)
    
    if perimeter == 0:
        return 0.0
    
    # 기본 원형도
    circularity = 4 * np.pi * area / (perimeter ** 2)
    
    # Convex Hull을 이용한 보정
    if contours:
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 원형도와 solidity를 조합
        adjusted_circularity = circularity * 0.7 + solidity * 0.3
        return min(adjusted_circularity, 1.0)
    
    return circularity


def identify_petri_dish(annotations, image_shape):
    """향상된 배양접시 식별 알고리즘"""
    if len(annotations) < 2:
        return None, list(range(len(annotations)))
    
    h, w = image_shape[:2]
    image_area = h * w
    candidates = []
    
    for idx, ann in enumerate(annotations):
        mask = ann.cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        mask = mask > 0
        
        # 기본 속성 계산
        area = np.sum(mask)
        area_ratio = area / image_area
        
        # 중심점과 이미지 중심과의 거리
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0:
            continue
            
        center_x = np.mean(x_indices)
        center_y = np.mean(y_indices)
        img_center_x, img_center_y = w / 2, h / 2
        center_distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
        center_distance_ratio = center_distance / np.sqrt(w**2 + h**2)
        
        # 원형도 계산
        circularity = calculate_circularity_robust(mask)
        
        # 최소 외접원 계산
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fill_ratio = 0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            enclosing_circle_area = np.pi * radius ** 2
            fill_ratio = area / enclosing_circle_area if enclosing_circle_area > 0 else 0
        
        # 배양접시 점수 계산
        score = 0
        
        # 1. 면적 비율 (전체 이미지의 20-80%)
        if 0.2 < area_ratio < 0.8:
            score += 30
        
        # 2. 중심 근접도 (이미지 중심에 가까울수록 높은 점수)
        if center_distance_ratio < 0.2:
            score += 25
        elif center_distance_ratio < 0.3:
            score += 15
        
        # 3. 원형도 (0.7 이상)
        if circularity > 0.7:
            score += 25
        elif circularity > 0.5:
            score += 15
        
        # 4. 채움 비율 (원형 대비 실제 면적)
        if fill_ratio > 0.8:
            score += 20
        
        candidates.append({
            'idx': idx,
            'score': score,
            'area': area,
            'circularity': circularity,
            'center_distance_ratio': center_distance_ratio
        })
    
    # 점수 기준 정렬
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # 최고 점수가 임계값 이상인 경우만 배양접시로 인정
    if candidates and candidates[0]['score'] >= 60:
        dish_idx = candidates[0]['idx']
        colony_indices = [c['idx'] for c in candidates if c['idx'] != dish_idx]
        return dish_idx, colony_indices
    
    # 배양접시를 찾지 못한 경우
    return None, list(range(len(annotations)))


class ColonyCounter:
    """콜로니 카운팅 및 포인트 관리 클래스 (통합 버전)"""
    def __init__(self):
        # 포인트 관리
        self.manual_points = []  # 수동으로 추가된 포인트
        self.auto_points = []    # 자동으로 감지된 포인트
        self.removed_history = []  # 삭제된 포인트 기록 (복원용)
        
        # 이미지 관리
        self.current_image = None  # 현재 표시 중인 이미지
        self.original_image = None  # 원본 이미지
        self.base_image = None     # 세그멘테이션 결과 이미지
        
        # 마스크 및 주석 데이터
        self.auto_annotations = []  # 자동 감지된 콜로니 애노테이션
        self.dish_annotation = None  # 페트리 접시 애노테이션
        self.colony_masks = []      # 콜로니 마스크 리스트
        self.dish_mask = None       # 배양 접시 마스크
        
        # 상태 관리
        self.auto_detected_count = 0  # 자동 감지된 콜로니 수
        self.remove_mode = False      # 제거 모드 활성화 여부
        self.scale_factor = 1.0       # 이미지 리사이징 비율
        self.last_method = "FastSAM"  # 마지막 사용 방법

    def reset(self):
        """모든 상태 초기화"""
        self.manual_points = []
        self.auto_points = []
        self.removed_history = []
        self.auto_annotations = []
        self.colony_masks = []
        self.dish_annotation = None
        self.dish_mask = None
        self.current_image = None
        self.base_image = None
        self.original_image = None
        self.auto_detected_count = 0
        self.remove_mode = False
        self.scale_factor = 1.0
        self.last_method = "FastSAM"

    def set_original_image(self, image):
        """원본 이미지 설정"""
        if isinstance(image, Image.Image):
            self.original_image = np.array(image)
            self.base_image = self.original_image.copy()
        else:
            self.original_image = image
            self.base_image = image.copy() if image is not None else None

    def toggle_remove_mode(self):
        """편집 모드 전환 (추가/제거 모드)"""
        self.remove_mode = not self.remove_mode
        current_img = self.draw_points()
        mode_html = self._get_mode_html()
        return current_img, mode_html

    def _get_mode_html(self):
        """현재 모드를 HTML 형식으로 반환"""
        if self.remove_mode:
            return "<span style='color: red; font-weight: bold;'>🔴 REMOVE MODE</span>"
        else:
            return "<span style='color: green; font-weight: bold;'>🟢 ADD MODE</span>"

    def add_or_remove_point(self, image, evt: gr.SelectData):
        """안전한 인덱스 처리로 포인트 추가/제거"""
        try:
            if self.base_image is None and image is not None:
                self.base_image = np.array(image)
                self.current_image = self.base_image.copy()

            x, y = evt.index

            if self.remove_mode:
                closest_idx, is_auto = self.find_closest_point(x, y)
                
                if closest_idx is not None:
                    if is_auto:
                        # 인덱스 범위 검증
                        if 0 <= closest_idx < len(self.auto_points):
                            removed_point = self.auto_points[closest_idx]
                            removed_annotation = None
                            
                            # 각 리스트의 길이 독립적으로 확인
                            if closest_idx < len(self.auto_annotations):
                                removed_annotation = self.auto_annotations[closest_idx]
                                del self.auto_annotations[closest_idx]
                            
                            if closest_idx < len(self.colony_masks):
                                del self.colony_masks[closest_idx]
                            
                            self.removed_history.append(("auto", closest_idx, removed_point, removed_annotation))
                            del self.auto_points[closest_idx]
                            self.auto_detected_count = max(0, self.auto_detected_count - 1)
                    else:
                        # 수동 포인트 인덱스 검증
                        manual_idx = closest_idx - len(self.auto_points)
                        if 0 <= manual_idx < len(self.manual_points):
                            removed_point = self.manual_points[manual_idx]
                            self.removed_history.append(("manual", manual_idx, removed_point, None))
                            del self.manual_points[manual_idx]
            else:
                self.manual_points.append((x, y))

            img_with_points = self.draw_points()
            return img_with_points, self.get_count_text()
        except Exception as e:
            print(f"Error in add_or_remove_point: {str(e)}")
            import traceback
            traceback.print_exc()
            return image, self.get_count_text()

    def find_closest_point(self, x, y, threshold=30):
        """가장 가까운 포인트 찾기"""
        all_points = self.auto_points + self.manual_points
        if not all_points:
            return None, None
        
        distances = []
        for idx, (px, py) in enumerate(all_points):
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            distances.append((dist, idx))

        if not distances:
            return None, None

        closest_dist, closest_idx = min(distances, key=lambda item: item[0])
        if closest_dist < threshold:
            is_auto = (closest_idx < len(self.auto_points))
            return closest_idx, is_auto
        return None, None

    def remove_last_point(self, image):
        """마지막으로 추가된 수동 포인트 제거"""
        try:
            if self.manual_points:
                removed_point = self.manual_points.pop()
                self.removed_history.append(('manual', len(self.manual_points), removed_point, None))
                img_with_points = self.draw_points()
                return img_with_points, self.get_count_text()
            return image, self.get_count_text()
        except Exception as e:
            print(f"Error in remove_last_point: {str(e)}")
            return image, self.get_count_text()

    def undo_last_removal(self, image):
        """마지막으로 삭제된 포인트 복원 (v90 기능)"""
        try:
            if not self.removed_history:
                return image, self.get_count_text() + "\n복원할 포인트가 없습니다."
            
            last_removal = self.removed_history.pop()
            removal_type, index, point, annotation = last_removal
            
            if removal_type == "auto":
                # 안전한 인덱스 삽입
                if 0 <= index <= len(self.auto_points):
                    self.auto_points.insert(index, point)
                else:
                    self.auto_points.append(point)
                
                if annotation is not None:
                    if 0 <= index <= len(self.auto_annotations):
                        self.auto_annotations.insert(index, annotation)
                    else:
                        self.auto_annotations.append(annotation)
                self.auto_detected_count += 1
            else:
                # 수동 포인트 복원
                if 0 <= index <= len(self.manual_points):
                    self.manual_points.insert(index, point)
                else:
                    self.manual_points.append(point)
            
            img_with_points = self.draw_points()
            return img_with_points, self.get_count_text() + "\n포인트가 복원되었습니다."
        except Exception as e:
            print(f"Error in undo_last_removal: {str(e)}")
            return image, self.get_count_text() + f"\n복원 중 오류: {str(e)}"

    def get_count_text(self):
        """카운트 결과 텍스트 생성"""
        total = self.auto_detected_count + len(self.manual_points)
        return (f"총 콜로니 수: {total}\n"
                f"🤖 자동 감지: {self.auto_detected_count}\n"
                f"👆 수동 추가: {len(self.manual_points)}")

    def draw_points(self):
        """포인트와 번호를 이미지에 그리기"""
        try:
            if self.base_image is None:
                return None

            img_with_points = self.base_image.copy()
            overlay = np.zeros_like(img_with_points)

            # 사용자 설정 가능한 변수들
            square_size = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            outline_thickness = 4

            # 색상 설정 (B, G, R)
            AUTO_TEXT_COLOR = (255, 255, 255)
            AUTO_OUTLINE_COLOR = (0, 0, 0)
            MANUAL_RECT_COLOR = (0, 0, 255)
            MANUAL_BORDER_COLOR = (255, 255, 0)
            MANUAL_TEXT_COLOR = (255, 0, 0)
            OVERLAY_OPACITY = 0.5

            # 자동 감지된 포인트 표시
            for idx, (x, y) in enumerate(self.auto_points, 1):
                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = int(x - text_width / 2)
                text_y = int(y - 10)

                # 외곽선
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            cv2.putText(img_with_points, text,
                                      (text_x + dx, text_y + dy),
                                      font, font_scale, AUTO_OUTLINE_COLOR, outline_thickness)

                # 텍스트
                cv2.putText(img_with_points, text,
                          (text_x, text_y),
                          font, font_scale, AUTO_TEXT_COLOR, font_thickness)

            # 수동으로 추가된 포인트 표시
            for idx, (x, y) in enumerate(self.manual_points, len(self.auto_points) + 1):
                pt1 = (int(x - square_size / 2), int(y - square_size / 2))
                pt2 = (int(x + square_size / 2), int(y + square_size / 2))
                
                cv2.rectangle(overlay, pt1, pt2, MANUAL_RECT_COLOR, -1)
                cv2.rectangle(overlay, pt1, pt2, MANUAL_BORDER_COLOR, 2)

                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = int(x - text_width / 2)
                text_y = int(y - 10)

                # 외곽선
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            cv2.putText(overlay, text,
                                      (text_x + dx, text_y + dy),
                                      font, font_scale, AUTO_OUTLINE_COLOR, outline_thickness)

                cv2.putText(img_with_points, text,
                          (text_x, text_y),
                          font, font_scale, MANUAL_TEXT_COLOR, font_thickness)

            cv2.addWeighted(overlay, OVERLAY_OPACITY, img_with_points, 1.0, 0, img_with_points)

            # 제거 모드 표시
            if self.remove_mode:
                mode_overlay = img_with_points.copy()
                h, w = mode_overlay.shape[:2]
                gradient = np.zeros((50, w, 3), dtype=np.uint8)
                for i in range(50):
                    alpha = 1.0 - (i / 50.0) * 0.3
                    gradient[i, :] = (0, 0, 255 * alpha)
                
                mode_overlay[:50, :] = cv2.addWeighted(mode_overlay[:50, :], 0.6, gradient, 0.8, 0)
                cv2.addWeighted(mode_overlay, 0.4, img_with_points, 0.6, 0, img_with_points)
                cv2.putText(img_with_points, "REMOVE MODE", (10, 35),
                          font, 1.2, (255, 255, 255), 2)

            # 전체 카운트 표시
            total_count = self.auto_detected_count + len(self.manual_points)
            count_text = f"Total Count: {total_count}"
            (text_width, text_height), baseline = cv2.getTextSize(count_text, font, 1.0, 2)
            margin = 15
            cv2.rectangle(img_with_points,
                         (margin, img_with_points.shape[0] - margin - text_height - baseline),
                         (margin + text_width + margin, img_with_points.shape[0] - margin + baseline),
                         (0, 0, 0), -1)
            cv2.putText(img_with_points, count_text,
                      (margin + margin // 2, img_with_points.shape[0] - margin - baseline // 2),
                      font, 1.0, (255, 255, 255), 2)

            return img_with_points
        except Exception as e:
            print(f"Error in draw_points: {str(e)}")
            return self.base_image


# 전역 메모리 모니터
gpu_monitor = GPUMemoryMonitor(threshold_gb=1.5)

# 전역 객체 생성
image_history = ImagePreprocessHistory()
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
    """이미지 전처리를 수행합니다."""
    try:
        if not isinstance(input_image, Image.Image):
            return input_image
        
        # 메모리 안전을 위한 이미지 크기 제한
        image = MemoryManager.limit_image_size(input_image.copy())

        if to_grayscale:
            image = image.convert('L').convert('RGB')
        
        if binary:
            image_np = np.array(image.convert('L'))
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
        return input_image


def fast_process(colony_annotations, dish_annotation, image, mask_random_color, withContours, original_size=None):
    """마스크 주석을 기반으로 이미지를 처리합니다."""
    try:
        image_np = np.array(image).copy()

        # 콜로니 마스크 처리
        for ann in colony_annotations:
            ann_cpu = ann.cpu().numpy()
            if ann_cpu.ndim == 3 and ann_cpu.shape[0] == 1:
                ann_cpu = ann_cpu[0]
            
            mask = ann_cpu > 0
            if mask.ndim == 2:
                if mask_random_color:
                    color = np.random.randint(100, 200, (3,)).tolist()
                    image_np[mask] = color
                else:
                    image_np[mask] = (0, 255, 0)

                if withContours:
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image_np, contours, -1, (255, 0, 0), 2)

        # 배양 접시 마스크 처리
        if dish_annotation is not None and withContours:
            dish_mask = dish_annotation.cpu().numpy()
            if dish_mask.ndim == 3 and dish_mask.shape[0] == 1:
                dish_mask = dish_mask[0]
            if dish_mask.ndim == 2:
                dish_mask = dish_mask > 0
                contours, _ = cv2.findContours(dish_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_np, contours, -1, (0, 0, 255), 3)

        processed_image = Image.fromarray(image_np)
        
        if original_size is not None:
            processed_image = processed_image.resize(original_size, Image.LANCZOS)
            
        return processed_image
    except Exception as e:
        print(f"Error in fast_process: {str(e)}")
        return image


def segment_and_count_colonies(
    input_image,
    input_size=1024,
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    mask_random_color=True,
    circularity_threshold=0.8,
    # 면적 필터링 (v6 기능)
    enable_area_filter=False,
    min_area_percentile=1,
    max_area_percentile=99,
    # 배양 접시 필터링 (grok_v6 기능)
    use_dish_filtering=True,
    dish_overlap_threshold=0.5,
    progress=gr.Progress()
):
    """메모리 안전 이미지 분석"""
    global counter
    
    try:
        # 메모리 정리
        MemoryManager.clear_gpu_memory()
        
        if input_image is None:
            return None, "이미지를 업로드해주세요."
        
        # 이미지 크기 제한
        input_image = MemoryManager.limit_image_size(input_image)
        
        counter.reset()
        
        progress(0.1, desc="이미지 준비 중...")
        counter.set_original_image(input_image)
        
        # 원본 크기 저장
        original_size = input_image.size
        
        # 이미지 리사이징
        w, h = input_image.size
        scale = input_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        input_resized = input_image.resize((new_w, new_h))
        counter.scale_factor = scale

        progress(0.3, desc="AI 분석 중...")
        
        # 모델 예측 전 메모리 체크
        if torch.cuda.is_available():
            free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # GB
            if free_memory < 1.0:
                print(f"Warning: Low GPU memory ({free_memory:.2f}GB)")
                MemoryManager.clear_gpu_memory()
        
        # 예측 수행
        with torch.no_grad():
            results = model.predict(
                source=np.array(input_resized),
                device=device,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=input_size,
                retina_masks=True
            )

        annotations = getattr(results[0].masks, 'data', None)
        if annotations is None or len(annotations) == 0:
            result_image = Image.fromarray(np.array(input_resized)).resize(original_size)
            counter.current_image = np.array(result_image)
            # 대용량 객체 명시적 해제
            del results
            MemoryManager.clear_gpu_memory()
            return np.array(result_image), "콜로니가 감지되지 않았습니다."

        progress(0.6, desc="마스크 처리 중...")
        
        # 향상된 배양 접시 필터링 처리
        dish_mask = None
        dish_annotation = None
        colony_annotations = list(annotations)
        
        if use_dish_filtering and len(annotations) > 1:
            # 향상된 배양접시 식별 알고리즘 사용
            dish_idx, colony_indices = identify_petri_dish(annotations, input_resized.size[::-1])
            
            if dish_idx is not None:
                dish_annotation = annotations[dish_idx]
                dish_mask_np = dish_annotation.cpu().numpy()
                if dish_mask_np.ndim == 3 and dish_mask_np.shape[0] == 1:
                    dish_mask_np = dish_mask_np[0]
                dish_mask = dish_mask_np > 0
                
                # 배양 접시를 제외한 콜로니 마스크
                colony_annotations = [annotations[idx] for idx in colony_indices]

        # 콜로니 필터링
        valid_colony_annotations = []
        counter.auto_points = []
        
        # 면적 필터링 준비
        if enable_area_filter and colony_annotations:
            colony_areas = [np.sum(ann.cpu().numpy() > 0) for ann in colony_annotations]
            if colony_areas:
                min_area = np.percentile(colony_areas, min_area_percentile)
                max_area = np.percentile(colony_areas, max_area_percentile)
            else:
                min_area, max_area = 0, float('inf')
        else:
            min_area, max_area = 0, float('inf')

        for ann in colony_annotations:
            ann_cpu = ann.cpu().numpy()
            if ann_cpu.ndim == 3 and ann_cpu.shape[0] == 1:
                ann_cpu = ann_cpu[0]
            mask = ann_cpu > 0
            area = np.sum(mask)

            # 면적 필터링
            if enable_area_filter and (area < min_area or area > max_area):
                continue

            # 배양 접시 내부 확인
            if use_dish_filtering and dish_mask is not None:
                overlap_ratio = np.sum(mask * dish_mask) / np.sum(mask)
                if overlap_ratio < dish_overlap_threshold:
                    continue

            # 향상된 원형도 계산
            circularity = calculate_circularity_robust(mask)
            if circularity >= circularity_threshold:
                valid_colony_annotations.append(ann)
                # 중심점 계산
                y_indices, x_indices = np.where(mask)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    # 원본 크기 좌표로 변환
                    original_x = int(center_x / scale)
                    original_y = int(center_y / scale)
                    counter.auto_points.append((original_x, original_y))

        progress(0.8, desc="결과 시각화 중...")
        processed_image = fast_process(
            colony_annotations=valid_colony_annotations,
            dish_annotation=dish_annotation,
            image=input_resized,
            mask_random_color=mask_random_color,
            withContours=withContours,
            original_size=original_size
        )

        # counter 업데이트
        counter.colony_masks = valid_colony_annotations
        counter.dish_mask = dish_mask
        counter.base_image = np.array(processed_image)
        counter.auto_detected_count = len(valid_colony_annotations)
        counter.current_image = counter.draw_points()

        # 대용량 객체 명시적 해제
        del results
        MemoryManager.clear_gpu_memory()

        progress(1.0, desc="완료!")
        return counter.current_image, counter.get_count_text()

    except torch.cuda.OutOfMemoryError:
        MemoryManager.clear_gpu_memory()
        return None, "GPU 메모리 부족. 이미지 크기를 줄이거나 배치 크기를 줄여주세요."
    except Exception as e:
        error_msg = f"분석 중 오류 발생: {str(e)}"
        print(error_msg)
        MemoryManager.clear_gpu_memory()
        if input_image is not None:
            return np.array(input_image), error_msg
        return None, error_msg


def save_results(original_image, processed_image):
    """결과를 저장합니다."""
    try:
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"result_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        # 파일 저장
        original_path = os.path.join(save_dir, "original.png")
        result_path = os.path.join(save_dir, "result.png")
        
        if isinstance(original_image, Image.Image):
            original_image.save(original_path)
        
        if isinstance(processed_image, np.ndarray):
            Image.fromarray(processed_image).save(result_path)
        
        # 결과 정보 저장
        result_info = {
            'datetime': datetime.now().isoformat(),
            'auto_count': counter.auto_detected_count,
            'manual_count': len(counter.manual_points),
            'total_count': counter.auto_detected_count + len(counter.manual_points),
            'analysis_method': counter.last_method,
            'device': str(device)
        }
        
        # JSON 저장
        json_path = os.path.join(save_dir, "result_info.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_info, f, indent=4, ensure_ascii=False)
        
        # 텍스트 결과 저장
        txt_path = os.path.join(save_dir, "count_results.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(counter.get_count_text())
            f.write(f"\n\n사용 디바이스: {device}")
            f.write(f"\n분석 방법: {counter.last_method}")

        return f"결과가 저장되었습니다:\n{save_dir}"
    except Exception as e:
        return f"저장 중 오류 발생: {str(e)}"


def process_single_file(file, **params):
    """단일 파일 처리 함수"""
    try:
        filename = os.path.basename(file.name)
        image = Image.open(file.name).convert("RGB")
        
        # 분석 수행
        result_image, count_text = segment_and_count_colonies(
            image,
            **params
        )
        
        # 결과 이미지가 numpy 배열인지 확인하고 안전하게 처리
        gallery_item = None
        if result_image is not None:
            try:
                # numpy 배열을 PIL Image로 변환하지 않고 그대로 유지
                # Gradio에서 numpy 배열을 직접 처리할 수 있음
                gallery_item = (result_image, f"{filename}\n{count_text}")
            except Exception as e:
                print(f"갤러리 아이템 생성 중 오류: {str(e)}")
                gallery_item = None
        
        # 카운트 값들을 Python 기본 타입으로 변환
        total_count = int(counter.auto_detected_count + len(counter.manual_points))
        auto_count = int(counter.auto_detected_count)
        manual_count = int(len(counter.manual_points))
        
        return {
            'filename': filename,
            'total_count': total_count,
            'auto_count': auto_count,
            'manual_count': manual_count,
            'status': 'success',
            'gallery_item': gallery_item
        }
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"파일 처리 중 상세 오류 ({file.name}): {error_detail}")
        
        return {
            'filename': os.path.basename(file.name) if hasattr(file, 'name') else 'unknown',
            'total_count': 0,
            'auto_count': 0,
            'manual_count': 0,
            'status': 'error',
            'error': str(e),
            'gallery_item': None
        }


def batch_process_optimized(files, chunk_size=5, **params):
    """청크 단위 배치 처리"""
    try:
        if not files:
            return "처리할 파일이 없습니다.", []
        
        # 결과 저장 디렉토리
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"batch_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)
        
        total_files = len(files)
        chunks = [files[i:i+chunk_size] for i in range(0, total_files, chunk_size)]
        
        results = []
        gallery_items = []
        
        progress = params.get('progress', lambda x, desc="": None)
        
        for chunk_idx, chunk in enumerate(chunks):
            # 청크 처리 전 메모리 정리
            gpu_monitor.cleanup()
            
            for file_idx, file in enumerate(chunk):
                global_idx = chunk_idx * chunk_size + file_idx
                progress((global_idx + 1) / total_files, desc=f"처리 중 ({global_idx + 1}/{total_files})")
                
                try:
                    # 파일 처리
                    result = process_single_file(file, **params)
                    
                    # JSON 안전 버전의 결과 생성 (gallery_item 제외)
                    json_safe_result = {
                        'filename': result['filename'],
                        'total_count': int(result['total_count']),  # numpy int를 Python int로 변환
                        'auto_count': int(result['auto_count']),
                        'manual_count': int(result['manual_count']),
                        'status': result['status']
                    }
                    
                    if result['status'] == 'error' and 'error' in result:
                        json_safe_result['error'] = str(result['error'])
                    
                    results.append(json_safe_result)
                    
                    # 갤러리 아이템은 별도로 관리
                    if result.get('gallery_item'):
                        gallery_items.append(result['gallery_item'])
                    
                    # 성공한 경우 파일 저장
                    if result['status'] == 'success':
                        filename = result['filename']
                        img_dir = os.path.join(batch_dir, os.path.splitext(filename)[0])
                        os.makedirs(img_dir, exist_ok=True)
                        
                        # 원본 이미지 저장
                        Image.open(file.name).save(os.path.join(img_dir, "original.png"))
                        
                        # 결과 이미지 저장 (numpy 배열 처리)
                        if result.get('gallery_item') and result['gallery_item'][0] is not None:
                            result_image = result['gallery_item'][0]
                            if isinstance(result_image, np.ndarray):
                                Image.fromarray(result_image).save(os.path.join(img_dir, "result.png"))
                            elif isinstance(result_image, Image.Image):
                                result_image.save(os.path.join(img_dir, "result.png"))
                        
                        # 카운트 결과 저장
                        count_info = f"총 콜로니 수: {result['total_count']}\n자동 감지: {result['auto_count']}\n수동 추가: {result['manual_count']}"
                        with open(os.path.join(img_dir, "count.txt"), 'w', encoding='utf-8') as f:
                            f.write(count_info)
                    
                except Exception as e:
                    error_result = {
                        'filename': os.path.basename(file.name),
                        'total_count': 0,
                        'auto_count': 0,
                        'manual_count': 0,
                        'status': 'error',
                        'error': str(e)
                    }
                    results.append(error_result)
                    print(f"파일 처리 중 오류 ({file.name}): {str(e)}")
                
                # 파일마다 메모리 체크
                gpu_monitor.check_memory()
            
            # 청크 완료 후 강제 메모리 정리
            gpu_monitor.cleanup()
            time.sleep(0.1)  # GPU 동기화 대기

        # CSV로 결과 저장 (JSON 안전 버전 사용)
        try:
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(batch_dir, "results.csv"), index=False, encoding='utf-8')
        except Exception as e:
            print(f"CSV 저장 중 오류: {str(e)}")
        
        # JSON으로 요약 저장 (JSON 안전 데이터만 포함)
        summary = {
            'timestamp': timestamp,
            'total_files': int(total_files),
            'successful': int(len([r for r in results if r['status'] == 'success'])),
            'failed': int(len([r for r in results if r['status'] == 'error'])),
            'results': results  # 이미 JSON 안전 버전
        }
        
        try:
            with open(os.path.join(batch_dir, "summary.json"), 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"JSON 저장 중 오류: {str(e)}")
            # JSON 저장 실패 시 텍스트로 대체
            with open(os.path.join(batch_dir, "summary.txt"), 'w', encoding='utf-8') as f:
                f.write(f"배치 처리 요약\n")
                f.write(f"처리 시간: {timestamp}\n")
                f.write(f"총 파일 수: {total_files}\n")
                f.write(f"성공: {summary['successful']}\n")
                f.write(f"실패: {summary['failed']}\n")

        summary_text = f"배치 처리 완료\n"
        summary_text += f"총 {total_files}개 파일 중 {summary['successful']}개 성공\n"
        summary_text += f"실패: {summary['failed']}개\n"
        summary_text += f"결과 저장: {batch_dir}"
        
        return summary_text, gallery_items

    except Exception as e:
        import traceback
        error_msg = f"배치 처리 중 오류: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, []


def batch_process(
    files,
    input_size=1024,
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    circularity_threshold=0.8,
    enable_area_filter=False,
    min_area_percentile=1,
    max_area_percentile=99,
    use_dish_filtering=True,
    dish_overlap_threshold=0.5,
    progress=gr.Progress()
):
    """최적화된 배치 처리 래퍼"""
    return batch_process_optimized(
        files=files,
        chunk_size=3,  # 메모리 안전을 위해 청크 크기 감소
        input_size=input_size,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        better_quality=better_quality,
        withContours=withContours,
        mask_random_color=True,
        circularity_threshold=circularity_threshold,
        enable_area_filter=enable_area_filter,
        min_area_percentile=min_area_percentile,
        max_area_percentile=max_area_percentile,
        use_dish_filtering=use_dish_filtering,
        dish_overlap_threshold=dish_overlap_threshold,
        progress=progress
    )


# CSS 스타일 (v99 디자인)
css = """
/* 기본 설정 */
:root {
    --primary-color: #3b82f6;
    --primary-dark: #1d4ed8;
    --primary-light: #93c5fd;
    --secondary-color: #10b981;
    --accent-color: #f59e0b;
    --danger-color: #ef4444;
    --text-color: #1f2937;
    --text-light: #6b7280;
    --bg-color: #f9fafb;
    --card-bg: #ffffff;
    --border-color: #e5e7eb;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    --radius-md: 0.375rem;
    --radius-lg: 0.5rem;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-color);
    background-color: var(--bg-color);
}

/* 헤더 스타일 */
.header {
    background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
    color: white;
    padding: 2rem;
    border-radius: var(--radius-lg);
    margin-bottom: 2rem;
    box-shadow: var(--shadow-md);
    text-align: center;
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.header h3 {
    font-size: 1.25rem;
    font-weight: 500;
    opacity: 0.9;
}

/* 버튼 스타일 */
button.primary {
    background: linear-gradient(to right, var(--primary-color), var(--primary-dark));
    color: white;
    font-weight: 600;
    border-radius: var(--radius-md);
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
    border: none;
}

button.primary:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

button.secondary {
    background-color: var(--card-bg);
    color: var(--primary-color);
    font-weight: 600;
    border: 1px solid var(--primary-color);
    border-radius: var(--radius-md);
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
}

button.secondary:hover {
    background-color: var(--primary-light);
    color: var(--primary-dark);
}

/* 섹션 타이틀 */
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--primary-dark);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* 카드 스타일 */
.card {
    background: var(--card-bg);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

/* 이미지 컨테이너 */
.input-image, .output-image {
    border: 2px solid var(--border-color);
    border-radius: var(--radius-lg);
    background-color: var(--card-bg);
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
    overflow: hidden;
}

.input-image:hover, .output-image:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--primary-light);
}

/* 결과 텍스트 */
.result-text {
    font-family: 'Roboto Mono', monospace;
    background-color: var(--card-bg);
    border-radius: var(--radius-md);
    border-left: 4px solid var(--secondary-color);
    padding: 1rem;
    box-shadow: var(--shadow-sm);
}

/* 아코디언 */
.accordion-header {
    background-color: var(--card-bg);
    color: var(--primary-dark);
    font-weight: 600;
    padding: 1rem;
    border-radius: var(--radius-md);
    border: 1px solid var(--border-color);
    transition: all 0.2s ease;
}

.accordion-header:hover {
    border-color: var(--primary-color);
    box-shadow: var(--shadow-sm);
}

/* 모드 인디케이터 */
.mode-indicator {
    text-align: center;
    padding: 0.5rem;
    border-radius: var(--radius-md);
    font-weight: 600;
}

/* 푸터 */
.footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    color: var(--text-light);
    border-top: 1px solid var(--border-color);
}

/* 로딩 애니메이션 */
.progress-bar {
    background: linear-gradient(90deg, var(--primary-light), var(--primary-color), var(--primary-dark));
    background-size: 200% 100%;
    animation: gradient-animation 2s ease infinite;
}

@keyframes gradient-animation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
"""

# Gradio UI 구성
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown(
        """
        <div class="header">
            <h1>🔬 Colony Counter</h1>
            <h3>AI 기반 콜로니 검출 및 카운팅</h3>
        </div>
        """
    )

    with gr.Tabs():
        # 단일 이미지 처리 탭
        with gr.Tab("🔍 단일 이미지 분석"):
            with gr.Row():
                # 입력 컬럼
                with gr.Column(scale=1):
                    gr.Markdown("<div class='section-title'>📸 이미지 업로드</div>")
                    input_image = gr.Image(
                        type="pil",
                        elem_classes=["input-image"],
                        show_label=False
                    )
                    
                    # 이미지 전처리
                    with gr.Accordion("🛠️ 이미지 전처리", open=False, elem_classes="accordion-header"):
                        to_grayscale = gr.Checkbox(label="흑백 변환", value=False)
                        binary = gr.Checkbox(label="바이너리 변환", value=False)
                        binary_threshold = gr.Slider(0, 255, 128, step=1, label="바이너리 임계값")
                        edge_detection = gr.Checkbox(label="에지 검출", value=False)
                        sharpen = gr.Checkbox(label="샤픈", value=False)
                        sharpen_amount = gr.Slider(0.5, 5.0, 1.0, step=0.1, label="샤픈 강도")
                        
                        with gr.Row():
                            preprocess_button = gr.Button("🔄 전처리 적용", variant="secondary")
                            reset_button = gr.Button("↺ 원본 복원", variant="secondary")
                            undo_button = gr.Button("↶ 실행 취소", variant="secondary")
                    
                    # 분석 설정
                    with gr.Accordion("⚙️ 분석 설정", open=False, elem_classes="accordion-header"):
                        with gr.Tab("기본"):
                            input_size_slider = gr.Slider(512, 2048, 1024, step=64, label="입력 크기")
                            withContours_checkbox = gr.Checkbox(label="윤곽선 표시", value=True)
                            better_quality_checkbox = gr.Checkbox(label="향상된 품질", value=False)
                            mask_random_color = gr.Checkbox(label="랜덤 색상", value=True)
                        
                        with gr.Tab("AI 탐지"):
                            iou_threshold_slider = gr.Slider(0.1, 0.9, 0.7, step=0.1, label="IOU 임계값")
                            conf_threshold_slider = gr.Slider(0.1, 0.9, 0.25, step=0.05, label="신뢰도 임계값")
                            circularity_threshold_slider = gr.Slider(0.0, 1.0, 0.8, step=0.01, label="원형도 임계값")
                        
                        with gr.Tab("면적 필터"):
                            enable_area_filter = gr.Checkbox(label="면적 필터링 사용", value=False)
                            min_area_percentile_slider = gr.Slider(0, 10, 1, step=1, label="최소 면적 백분위")
                            max_area_percentile_slider = gr.Slider(90, 100, 99, step=1, label="최대 면적 백분위")
                        
                        with gr.Tab("배양 접시 필터"):
                            use_dish_filtering = gr.Checkbox(label="배양 접시 필터링 사용", value=True)
                            dish_overlap_threshold = gr.Slider(0.1, 1.0, 0.5, step=0.1, label="접시 겹침 임계값")
                    
                    segment_button = gr.Button("🔍 이미지 분석", variant="primary", scale=2)

                # 출력 컬럼
                with gr.Column(scale=1):
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
                    
                    # 수동 편집
                    with gr.Row():
                        with gr.Column(scale=1):
                            remove_mode_button = gr.Button("🔄 편집 모드 전환", variant="secondary")
                            remove_mode_indicator = gr.Markdown(
                                value="<span style='color: green; font-weight: bold;'>🟢 ADD MODE</span>",
                                elem_classes="mode-indicator"
                            )
                        with gr.Column(scale=1):
                            remove_point_button = gr.Button("↩️ 마지막 포인트 취소", variant="secondary")
                            undo_removal_button = gr.Button("♻️ 삭제 복원", variant="secondary")
                    
                    # 저장
                    save_button = gr.Button("💾 결과 저장", variant="primary")
                    save_output = gr.Textbox(label="저장 결과", lines=2, elem_classes="result-text")

        # 배치 처리 탭
        with gr.Tab("📊 배치 처리"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("<div class='section-title'>📁 파일 선택</div>")
                    batch_files = gr.File(
                        label="이미지 파일 선택 (여러 개 가능)",
                        file_count="multiple",
                        file_types=["image"],
                        elem_classes=["card"]
                    )
                    
                    # 배치 처리 설정
                    with gr.Accordion("⚙️ 배치 처리 설정", open=True, elem_classes="accordion-header"):
                        with gr.Row():
                            batch_input_size = gr.Slider(512, 2048, 1024, step=64, label="입력 크기")
                            batch_iou_threshold = gr.Slider(0.1, 0.9, 0.7, step=0.1, label="IOU 임계값")
                        
                        with gr.Row():
                            batch_conf_threshold = gr.Slider(0.1, 0.9, 0.25, step=0.05, label="신뢰도 임계값")
                            batch_circularity = gr.Slider(0.0, 1.0, 0.8, step=0.01, label="원형도 임계값")
                        
                        with gr.Row():
                            batch_enable_area_filter = gr.Checkbox(label="면적 필터링", value=False)
                            batch_use_dish_filtering = gr.Checkbox(label="배양 접시 필터링", value=True)
                        
                        with gr.Row():
                            batch_min_area = gr.Slider(0, 10, 1, step=1, label="최소 면적 백분위")
                            batch_max_area = gr.Slider(90, 100, 99, step=1, label="최대 면적 백분위")
                        
                        batch_dish_overlap = gr.Slider(0.1, 1.0, 0.5, step=0.1, label="접시 겹침 임계값")
                    
                    batch_process_button = gr.Button("🚀 배치 처리 시작", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("<div class='section-title'>📊 처리 결과</div>")
                    batch_summary = gr.Textbox(
                        label="처리 요약",
                        lines=8,
                        elem_classes="result-text"
                    )
                    
                    # 결과 갤러리
                    batch_gallery = gr.Gallery(
                        label="처리된 이미지",
                        show_label=False,
                        columns=3,
                        height=400,
                        elem_classes=["card"]
                    )
                    
                    open_output_button = gr.Button("📂 결과 폴더 열기", variant="secondary")

    # 푸터
    gr.Markdown(
        """
        <div class="footer">
            <div>🔬 Colony Counter</div>
            <div>Produced By BDK &copy; 2025</div>
        </div>
        """
    )

    # 이벤트 핸들러 연결
    
    # 이미지 업로드
    input_image.upload(
        lambda img: (image_history.set_original(img), "이미지가 업로드되었습니다."),
        inputs=[input_image],
        outputs=[input_image, colony_count_text]
    )
    
    # 전처리
    preprocess_button.click(
        lambda img, gs, b, bt, ed, sh, sa: image_history.add_state(
            preprocess_image(img, gs, b, bt, ed, sh, sa)
        ),
        inputs=[input_image, to_grayscale, binary, binary_threshold, edge_detection, sharpen, sharpen_amount],
        outputs=[input_image]
    )
    
    reset_button.click(
        lambda: image_history.reset(),
        outputs=[input_image]
    )
    
    undo_button.click(
        lambda: image_history.undo(),
        outputs=[input_image]
    )
    
    # 분석
    segment_button.click(
        segment_and_count_colonies,
        inputs=[
            input_image,
            input_size_slider,
            iou_threshold_slider,
            conf_threshold_slider,
            better_quality_checkbox,
            withContours_checkbox,
            mask_random_color,
            circularity_threshold_slider,
            enable_area_filter,
            min_area_percentile_slider,
            max_area_percentile_slider,
            use_dish_filtering,
            dish_overlap_threshold
        ],
        outputs=[output_image, colony_count_text]
    )
    
    # 수동 편집
    output_image.select(
        counter.add_or_remove_point,
        inputs=[output_image],
        outputs=[output_image, colony_count_text]
    )
    
    remove_mode_button.click(
        counter.toggle_remove_mode,
        outputs=[output_image, remove_mode_indicator]
    )
    
    remove_point_button.click(
        counter.remove_last_point,
        inputs=[output_image],
        outputs=[output_image, colony_count_text]
    )
    
    undo_removal_button.click(
        counter.undo_last_removal,
        inputs=[output_image],
        outputs=[output_image, colony_count_text]
    )
    
    # 저장
    save_button.click(
        save_results,
        inputs=[input_image, output_image],
        outputs=[save_output]
    )
    
    # 배치 처리
    batch_process_button.click(
        batch_process,
        inputs=[
            batch_files,
            batch_input_size,
            batch_iou_threshold,
            batch_conf_threshold,
            gr.Checkbox(value=False, visible=False),  # better_quality
            gr.Checkbox(value=True, visible=False),   # withContours
            batch_circularity,
            batch_enable_area_filter,
            batch_min_area,
            batch_max_area,
            batch_use_dish_filtering,
            batch_dish_overlap
        ],
        outputs=[batch_summary, batch_gallery]
    )
    
    # 결과 폴더 열기
    def open_folder():
        try:
            if platform.system() == "Windows":
                os.startfile(DEFAULT_OUTPUT_DIR)
            elif platform.system() == "Darwin":
                subprocess.run(["open", DEFAULT_OUTPUT_DIR])
            else:
                subprocess.run(["xdg-open", DEFAULT_OUTPUT_DIR])
            return f"폴더 열기: {DEFAULT_OUTPUT_DIR}"
        except Exception as e:
            return f"폴더 열기 실패: {str(e)}"
    
    open_output_button.click(
        open_folder,
        outputs=[batch_summary]
    )

if __name__ == "__main__":
    demo.launch(share=False)