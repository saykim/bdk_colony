from ultralytics import YOLO
import gradio as gr
import torch
from PIL import Image
from PIL.Image import Resampling
import numpy as np
import cv2
import os
from datetime import datetime
import pandas as pd
from pathlib import Path
import json
import sys
import warnings

# FutureWarning 무시 (옵션, 권장하지 않음)
warnings.filterwarnings("ignore", category=FutureWarning)

# 기본 출력 디렉토리 설정
DEFAULT_OUTPUT_DIR = os.path.join(str(Path.home()), 'colony_counter_results')
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# FastSAM 모델 로드
try:
    model = YOLO('./weights/FastSAM-x.pt')
except Exception as e:
    print(f"Error loading YOLO model: {str(e)}")
    sys.exit(1)

# 장치 설정
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)

def fast_process(colony_annotations, dish_annotation, image, mask_random_color, withContours):
    """
    마스크 주석을 기반으로 이미지를 처리하고, 페트리 접시는 외곽선만 그리며 콜로니는 채우고 외곽선을 그립니다.
    """
    try:
        image_np = np.array(image).copy()

        # 콜로니 마스크 처리
        for ann in colony_annotations:
            ann_cpu = ann.cpu().numpy()
            if ann_cpu.ndim == 3 and ann_cpu.shape[0] == 1:
                ann_cpu = ann_cpu[0]  # (1, H, W) -> (H, W)
            
            mask = ann_cpu > 0
            if mask.ndim == 2:
                if mask_random_color:
                    color = np.random.randint(0, 255, (3,)).tolist()
                    image_np[mask] = color
                else:
                    image_np[mask] = (0, 255, 0)  # 기본 초록색

            if withContours:
                contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                cv2.drawContours(image_np, contours, -1, (255, 0, 0), 2)  # 파란색 경계선

        # 페트리 접시 마스크 처리 (외곽선만)
        if dish_annotation is not None:
            dish_mask = dish_annotation.cpu().numpy()
            if dish_mask.ndim == 3 and dish_mask.shape[0] == 1:
                dish_mask = dish_mask[0]
            if dish_mask.ndim == 2:
                dish_mask = dish_mask > 0
                if withContours:
                    contours = cv2.findContours(dish_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]
                    cv2.drawContours(image_np, contours, -1, (0, 0, 255), 3)  # 빨간색 외곽선

        processed_image = Image.fromarray(image_np)
        return processed_image
    except Exception as e:
        print(f"Error in fast_process: {str(e)}")
        return image

class ColonyCounter:
    """
    콜로니 수동 카운팅 도구 클래스
    - 자동 감지된 포인트와 수동 추가 포인트를 관리
    - 이미지에 포인트를 시각화
    - 제거 모드 지원
    """
    def __init__(self):
        self.manual_points = []  # 수동으로 추가된 포인트 목록
        self.auto_points = []  # 자동으로 감지된 포인트 목록
        self.auto_annotations = []  # 자동으로 감지된 콜로니 애노테이션
        self.dish_annotation = None  # 페트리 접시 애노테이션
        self.current_image = None  # 현재 표시 중인 이미지
        self.original_image = None  # 원본 이미지 (리사이징 후)
        self.base_image = None  # 세그멘테이션 결과 이미지 (오버레이 전)
        self.auto_detected_count = 0  # 자동 감지된 콜로니 수
        self.remove_mode = False  # 제거 모드 활성화 여부
        self.last_method = None  # 마지막으로 사용한 분석 방법
        self.scale_factor = 1.0  # 이미지 리사이징 비율 저장 변수 추가
        self.removed_history = []  # 삭제된 포인트 기록 (원복 기능을 위함)

    def reset(self):
        """모든 상태 초기화"""
        self.manual_points = []
        self.auto_points = []
        self.auto_annotations = []
        self.dish_annotation = None
        self.current_image = None
        self.base_image = None
        self.auto_detected_count = 0
        self.remove_mode = False
        self.original_image = None
        self.last_method = None
        self.scale_factor = 1.0  # 리셋 시 scale_factor도 초기화
        self.removed_history = []  # 삭제 기록도 초기화

    def set_original_image(self, image):
        """원본 이미지 설정"""
        if isinstance(image, Image.Image):
            self.original_image = np.array(image)
            # base_image도 함께 설정 (초기에는 원본과 동일)
            self.base_image = self.original_image.copy()
        else:
            self.original_image = image
            self.base_image = image.copy() if image is not None else None

    def toggle_remove_mode(self):
        """제거 모드 토글"""
        self.remove_mode = not self.remove_mode
        # 제거 모드가 변경되면 현재 이미지를 새로 그리기
        current_img = self.draw_points()
        mode_text = "🔴 Remove Mode" if self.remove_mode else "🟢 Add Mode"
        return current_img, mode_text

    def find_closest_point(self, x, y, threshold=30):
        # 자동 포인트와 수동 포인트 모두에서 가장 가까운 점 찾기
        all_points = self.auto_points + self.manual_points
        if not all_points:
            return None, None
        
        # 클릭 좌표는 이미 UI 좌표계이므로, 비교를 위해 원본 이미지 좌표계로 변환할 필요 없음
        scaled_x = x
        scaled_y = y

        distances = []
        for idx, (px, py) in enumerate(all_points):
            dist = np.sqrt((scaled_x - px) ** 2 + (scaled_y - py) ** 2)
            distances.append((dist, idx))

        if not distances:
            return None, None

        closest_dist, closest_idx = min(distances, key=lambda item: item[0])
        if closest_dist < threshold:
            # auto_points 길이를 기준으로 자동/수동을 구분
            is_auto = (closest_idx < len(self.auto_points))
            return closest_idx, is_auto
        return None, None

    def debug_find_closest(self, x, y, threshold=30):
        """디버깅용 함수: 가장 가까운 포인트 찾기 과정을 자세히 출력"""
        all_points = self.auto_points + self.manual_points
        if not all_points:
            print("포인트가 없습니다.")
            return None, None
        
        # 클릭 좌표는 이미 UI 좌표계이므로, 비교를 위해 원본 이미지 좌표계로 변환할 필요 없음
        scaled_x = x
        scaled_y = y
        print(f"클릭 좌표: ({x}, {y}) -> 변환 좌표: ({scaled_x}, {scaled_y})")
        print(f"스케일 팩터: {self.scale_factor}")
        
        if len(self.auto_points) > 0:
            print(f"자동 포인트 개수: {len(self.auto_points)}")
            for i, (px, py) in enumerate(self.auto_points[:5]):
                dist = np.sqrt((scaled_x - px) ** 2 + (scaled_y - py) ** 2)
                print(f"  자동 포인트 {i}: ({px}, {py}), 거리: {dist}")
            if len(self.auto_points) > 5:
                print(f"  ... 외 {len(self.auto_points)-5}개")
        
        if len(self.manual_points) > 0:
            print(f"수동 포인트 개수: {len(self.manual_points)}")
            for i, (px, py) in enumerate(self.manual_points[:5]):
                dist = np.sqrt((scaled_x - px) ** 2 + (scaled_y - py) ** 2)
                print(f"  수동 포인트 {i}: ({px}, {py}), 거리: {dist}")
            if len(self.manual_points) > 5:
                print(f"  ... 외 {len(self.manual_points)-5}개")

        distances = []
        for idx, (px, py) in enumerate(all_points):
            dist = np.sqrt((scaled_x - px) ** 2 + (scaled_y - py) ** 2)
            distances.append((dist, idx))

        if not distances:
            print("거리 계산 결과가 없습니다.")
            return None, None

        closest_dist, closest_idx = min(distances, key=lambda item: item[0])
        print(f"가장 가까운 포인트 인덱스: {closest_idx}, 거리: {closest_dist}")
        print(f"임계값: {threshold}")
        
        if closest_dist < threshold:
            is_auto = (closest_idx < len(self.auto_points))
            type_str = "자동" if is_auto else "수동"
            print(f"선택된 포인트: {type_str} 포인트 {closest_idx}, 임계값 범위 내에 있음")
            return closest_idx, is_auto
        else:
            print(f"가장 가까운 포인트도 임계값보다 멀리 있습니다.")
        return None, None

    def add_or_remove_point(self, image, evt: gr.SelectData):
        """
        이미지에 포인트 추가 또는 제거하는 함수
        - 클릭한 위치에 포인트 추가(일반 모드) 또는 가장 가까운 포인트 제거(제거 모드)
        """
        try:
            if self.base_image is None and image is not None:
                self.base_image = np.array(image)
                self.current_image = self.base_image.copy()

            x, y = evt.index

            if self.remove_mode:
                # 제거 모드인 경우, 가장 가까운 포인트 찾기
                closest_idx, is_auto = self.find_closest_point(x, y)
                
                if closest_idx is not None:
                    if is_auto:
                        # 자동 감지 포인트 제거 전 정보 저장
                        removed_point = self.auto_points[closest_idx]
                        removed_annotation = None
                        if len(self.auto_annotations) > closest_idx:
                            removed_annotation = self.auto_annotations[closest_idx]
                        # 삭제 정보 저장: (유형, 인덱스, 좌표, 애노테이션)
                        self.removed_history.append(("auto", closest_idx, removed_point, removed_annotation))
                        
                        # 자동 감지 포인트 제거
                        del self.auto_points[closest_idx]
                        # 해당 애노테이션도 함께 제거
                        if len(self.auto_annotations) > closest_idx:
                            del self.auto_annotations[closest_idx]
                        self.auto_detected_count -= 1
                    else:
                        # 수동 추가 포인트 제거 전 정보 저장
                        manual_idx = closest_idx - len(self.auto_points)
                        removed_point = self.manual_points[manual_idx]
                        # 삭제 정보 저장: (유형, 인덱스, 좌표)
                        self.removed_history.append(("manual", manual_idx, removed_point, None))
                        
                        # 수동 추가 포인트 제거
                        del self.manual_points[manual_idx]
            else:
                # 일반 모드인 경우, 클릭 위치에 수동 포인트 추가
                self.manual_points.append((x, y))

            # 포인트 반영한 이미지 다시 그리기
            img_with_points = self.draw_points()
            return img_with_points, self.get_count_text()
        except Exception as e:
            print(f"Error in add_or_remove_point: {str(e)}")
            import traceback
            traceback.print_exc()
            return image, self.get_count_text()

    def remove_last_point(self, image):
        try:
            if self.manual_points:
                self.manual_points.pop()
                img_with_points = self.draw_points()
                return img_with_points, self.get_count_text()
            return image, self.get_count_text()
        except Exception as e:
            print(f"Error in remove_last_point: {str(e)}")
            return image, self.get_count_text()

    def get_count_text(self):
        try:
            method_text = f"Method: {self.last_method}\n" if self.last_method != "NONE" else ""
            total = self.auto_detected_count + len(self.manual_points)
            return (f"{method_text}Total Colony Count: {total}\n"
                    f"🤖 Auto detected: {self.auto_detected_count}\n"
                    f"👆 Manually added: {len(self.manual_points)}")
        except Exception as e:
            print(f"Error in get_count_text: {str(e)}")
            return "Error calculating count"

    def draw_points(self):
        """
        이미지에 콜로니 포인트와 번호를 그리는 메서드
        
        기능:
        1. 자동 감지된 콜로니에 번호 표시 (흰색 텍스트 + 검은색 외곽선)
        2. 수동으로 추가된 포인트에 사각형과 번호 표시 (빨간색)
        3. 제거 모드일 때 상단에 표시
        
        [사용자 커스터마이징 가능한 부분]
        1. 마커 크기:
           - square_size: 수동 포인트의 사각형 크기 (기본값: 25)
           - font: 폰트 종류 (기본값: cv2.FONT_HERSHEY_SIMPLEX)
           - font_scale: 텍스트 크기 (기본값: 0.77)
           - font_thickness: 내부 텍스트 두께 (기본값: 1)
           - outline_thickness: 외곽선 두께 (기본값: 4)

        2. 색상 설정:
           - 자동 감지 콜로니:
             * 텍스트 색상: (255, 255, 255) # 흰색
             * 외곽선 색상: (0, 0, 0) # 검은색
           - 수동 추가 콜로니:
             * 사각형 색상: (255, 0, 0) # 빨간색
             * 테두리 색상: (0, 0, 0) # 검은색
             * 텍스트 색상: (255, 0, 0) # 빨간색

        3. 투명도 설정:
           - overlay_opacity: 오버레이 투명도 (기본값: 0.4)
           - remove_mode_opacity: 제거 모드 배너 투명도 (기본값: 0.3)
        
        반환값:
        - 포인트와 번호가 표시된 이미지
        - 오류 발생 시 원본 이미지 반환
        """
        try:
            if self.base_image is None:
                return None

            # 기반 이미지(세그멘테이션 결과)를 사용하여 매번 새롭게 그리기
            img_with_points = self.base_image.copy()
            overlay = np.zeros_like(img_with_points)

            ###########################################
            # Clear previous drawings by creating a fresh copy of the original image
            img_with_points = self.base_image.copy()
            overlay = np.zeros_like(img_with_points)

            # [사용자 설정 가능한 변수들]
            ###########################################
            # 1. 마커 크기 및 폰트 설정
            square_size = 25  # 수동 포인트의 사각형 크기
            font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 종류
            font_scale = 0.77  # 텍스트 크기
            font_thickness = 1  # 내부 텍스트 두께
            outline_thickness = 4  # 외곽선 두께

            # 2. 색상 설정 (B, G, R 형식)
            AUTO_TEXT_COLOR = (255, 255, 255)  # 자동 감지 텍스트 색상 (흰색)
            AUTO_OUTLINE_COLOR = (0, 0, 0)     # 자동 감지 외곽선 색상 (검은색)
            MANUAL_RECT_COLOR = (255, 0, 0)    # 수동 추가 사각형 색상 (빨간색)
            MANUAL_BORDER_COLOR = (0, 0, 0)    # 수동 추가 테두리 색상 (검은색)
            MANUAL_TEXT_COLOR = (255, 0, 0)    # 수동 추가 텍스트 색상 (빨간색)

            # 3. 투명도 설정
            OVERLAY_OPACITY = 0.4  # 오버레이 투명도
            REMOVE_MODE_OPACITY = 0.3  # 제거 모드 배너 투명도

            ###########################################
            # 1. 자동 감지된 콜로니 번호 표시
            ###########################################
            for idx, (x, y) in enumerate(self.auto_points, 1):
                # 좌표는 이미 원본 이미지 좌표계이므로 현재 화면 크기에 맞게 직접 표시
                display_x = x
                display_y = y
                
                text = str(idx)
                # 텍스트 크기 계산하여 중앙 정렬
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = int(display_x - text_width / 2)
                text_y = int(display_y - 10)

                # [중요] 8방향 검은색 외곽선으로 텍스트 가시성 향상
                # dx, dy로 8방향의 오프셋을 지정하여 외곽선 생성
                for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    cv2.putText(img_with_points, text,
                              (text_x + dx, text_y + dy),
                              font, font_scale, AUTO_OUTLINE_COLOR, outline_thickness)

                # 흰색 텍스트를 외곽선 위에 그리기
                cv2.putText(img_with_points, text,
                          (text_x, text_y),
                          font, font_scale, AUTO_TEXT_COLOR, font_thickness)

            ###########################################
            # 2. 수동으로 추가된 포인트 표시
            ###########################################
            for idx, (x, y) in enumerate(self.manual_points, len(self.auto_points) + 1):
                # 좌표는 이미 원본 이미지 좌표계이므로 현재 화면 크기에 맞게 직접 표시
                display_x = x
                display_y = y
                
                # 사각형 좌표 계산
                pt1 = (int(display_x - square_size / 2), int(display_y - square_size / 2))
                pt2 = (int(display_x + square_size / 2), int(display_y + square_size / 2))
                
                # [중요] 반투명 사각형 그리기
                cv2.rectangle(overlay, pt1, pt2, MANUAL_RECT_COLOR, -1)  # 색상 채우기
                cv2.rectangle(overlay, pt1, pt2, MANUAL_BORDER_COLOR, 2)  # 테두리

                # 번호 텍스트 추가
                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = int(display_x - text_width / 2)
                text_y = int(display_y - 10)

                # 8방향 검은색 외곽선
                for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    cv2.putText(overlay, text,
                              (text_x + dx, text_y + dy),
                              font, font_scale, AUTO_OUTLINE_COLOR, outline_thickness)

                # 빨간색 텍스트
                cv2.putText(img_with_points, text,
                          (text_x, text_y),
                          font, font_scale, MANUAL_TEXT_COLOR, font_thickness)

            # [중요] 오버레이 이미지를 원본과 블렌딩
            cv2.addWeighted(overlay, OVERLAY_OPACITY, img_with_points, 1.0, 0, img_with_points)

            ###########################################
            # 3. 제거 모드 표시
            ###########################################
            if self.remove_mode:
                # 상단에 빨간색 배너 추가
                overlay = img_with_points.copy()
                cv2.rectangle(overlay, (0, 0), (img_with_points.shape[1], 40), MANUAL_RECT_COLOR, -1)
                cv2.addWeighted(overlay, REMOVE_MODE_OPACITY, img_with_points, 0.7, 0, img_with_points)
                # "REMOVE MODE" 텍스트 표시
                cv2.putText(img_with_points, "REMOVE MODE", (10, 30),
                          font, 1, AUTO_TEXT_COLOR, 2)

            return img_with_points
        except Exception as e:
            print(f"Error in draw_points: {str(e)}")
            return self.base_image

    def save_results(self, output_dir, img_filename):
        """결과 이미지와 텍스트 저장"""
        try:
            image_name = os.path.splitext(img_filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(output_dir, f"{image_name}_{timestamp}")
            os.makedirs(save_dir, exist_ok=True)
            
            if self.original_image is not None:
                original_path = os.path.join(save_dir, "original.png")
                cv2.imwrite(original_path, cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR))
            
            # 포인트가 그려진 현재 이미지 저장
            current_img = self.draw_points()
            if current_img is not None:
                result_path = os.path.join(save_dir, "result.png")
                cv2.imwrite(result_path, cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR))
            
            result_txt = os.path.join(save_dir, "count_results.txt")
            with open(result_txt, 'w') as f:
                f.write(self.get_count_text())
                
            return f"Results saved to {save_dir}"
        except Exception as e:
            return f"Error saving results: {str(e)}"

    def undo_last_removal(self, image):
        """
        마지막으로 삭제된 포인트를 복원하는 함수
        - 삭제 기록이 없으면 현재 이미지를 그대로 반환
        - 삭제 기록이 있으면 마지막 삭제된 포인트 정보를 가져와 복원
        """
        try:
            # 삭제 기록이 없으면 복원할 것이 없음
            if not self.removed_history:
                return image, self.get_count_text() + "\n삭제 기록이 없습니다."
            
            # 마지막 삭제 기록 가져오기
            last_removal = self.removed_history.pop()
            removal_type, index, point, annotation = last_removal
            
            # 삭제된 포인트 유형에 따라 복원
            if removal_type == "auto":
                # 자동 포인트 복원
                self.auto_points.insert(index, point)
                if annotation is not None:
                    # 애노테이션도 함께 복원
                    self.auto_annotations.insert(index, annotation)
                self.auto_detected_count += 1
            else:
                # 수동 포인트 복원
                self.manual_points.insert(index, point)
            
            # 포인트가 반영된 이미지 다시 그리기
            img_with_points = self.draw_points()
            return img_with_points, self.get_count_text() + "\n마지막 삭제된 포인트가 복원되었습니다."
        except Exception as e:
            print(f"Error in undo_last_removal: {str(e)}")
            import traceback
            traceback.print_exc()
            return image, self.get_count_text() + f"\n복원 중 오류 발생: {str(e)}"

class ImagePreprocessHistory:
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

    def redo(self):
        """이전 상태로 복원"""
        if self.current_index > 0:
            self.current_index -= 1
            self.current_image = self.history[self.current_index]
            return self.current_image
        return self.current_image

# 전역 이미지 히스토리 객체
image_history = ImagePreprocessHistory()

def segment_and_count_colonies(
    input_image,
    method='fastsam',
    input_size=1024,
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    mask_random_color=True,
    min_area_percentile=1,
    max_area_percentile=99,
    circularity_threshold=0.8
):
    try:
        if input_image is None:
            return None, "No input image provided.", None

        # 새로운 counter 인스턴스 생성
        new_counter = ColonyCounter()
        new_counter.reset()
        new_counter.set_original_image(input_image)
        new_counter.last_method = method.upper()

        # 입력 이미지 리사이즈
        input_size = int(input_size)
        w, h = input_image.size
        scale = input_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        input_resized = input_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 스케일 비율 저장
        new_counter.scale_factor = scale

        # FastSAM 모델 예측
        input_array = np.array(input_resized)
        results = model.predict(
            source=input_array,
            device=device,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=input_size,
            retina_masks=True  # FastSAM 마스크를 위해 필요
        )

        annotations = getattr(results[0].masks, 'data', None)
        if annotations is None or len(annotations) == 0:
            new_counter.current_image = np.array(input_resized)
            return np.array(input_resized), "No colonies detected", new_counter

        # 각 마스크 면적
        areas = [np.sum(ann.cpu().numpy() > 0) for ann in annotations]

        # 가장 큰 마스크 = 페트리 접시(dish)
        dish_idx = np.argmax(areas)
        dish_annotation = annotations[dish_idx]
        colony_annotations = [ann for idx, ann in enumerate(annotations) if idx != dish_idx]

        # --- 면적 필터링 로직 추가 시작 ---
        if not colony_annotations:
            # 콜로니 후보가 없으면 바로 반환
            new_counter.current_image = np.array(input_resized)
            return np.array(input_resized), "No colonies detected after dish removal", new_counter

        # 모든 콜로니 후보의 면적 계산
        all_colony_areas = []
        for ann in colony_annotations:
            ann_cpu_area = ann.cpu().numpy()
            if ann_cpu_area.ndim == 3 and ann_cpu_area.shape[0] == 1:
                ann_cpu_area = ann_cpu_area[0]
            mask_area = ann_cpu_area > 0
            all_colony_areas.append(np.sum(mask_area))

        # 면적 리스트가 비어있지 않다면 백분위수 기반 임계값 계산
        if all_colony_areas:
            min_area_threshold = np.percentile(all_colony_areas, min_area_percentile)
            max_area_threshold = np.percentile(all_colony_areas, max_area_percentile)
        else:
            # 면적 계산이 불가능한 경우 (모든 후보 면적이 0 등), 기본값 설정 또는 오류 처리
            min_area_threshold = 0
            max_area_threshold = float('inf') # 사실상 면적 필터링 안함
        # --- 면적 필터링 로직 추가 끝 ---

        # 필터링 + 원형도 계산
        valid_colony_annotations = []
        new_counter.auto_points = []  # 자동 감지된 점 초기화

        for ann in colony_annotations:
            ann_cpu = ann.cpu().numpy()
            if ann_cpu.ndim == 3 and ann_cpu.shape[0] == 1:
                ann_cpu = ann_cpu[0]
            mask = ann_cpu > 0
            area = np.sum(mask)

            # 면적 필터링 조건 확인
            is_area_valid = (area >= min_area_threshold) and (area <= max_area_threshold)

            # 원형도 계산 및 필터링 조건 확인 (면적 조건 통과 시)
            is_circularity_valid = False
            if is_area_valid:
                contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                if contours and len(contours) > 0:
                    perimeter = cv2.arcLength(contours[0], True)
                    # area = cv2.contourArea(contours[0]) # 필요시 contourArea 사용
                    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
                    if circularity >= circularity_threshold:
                        is_circularity_valid = True

            # 최종 필터링: 면적과 원형도 모두 유효한 경우
            if is_area_valid and is_circularity_valid:
                valid_colony_annotations.append(ann)
                # 마스크 중심점
                y_indices, x_indices = np.where(mask)
                center_x = int(np.mean(x_indices))
                center_y = int(np.mean(y_indices))
                
                # 리사이즈된 이미지 좌표를 UI 좌표계로 변환
                # UI 좌표계는 리사이즈된 이미지를 기준으로 함
                new_counter.auto_points.append((center_x, center_y))

        if valid_colony_annotations:
            processed_image = fast_process(
                colony_annotations=valid_colony_annotations,
                dish_annotation=dish_annotation,
                image=input_resized,
                mask_random_color=mask_random_color,
                withContours=withContours
            )
        else:
            processed_image = input_resized

        # counter 객체에 결과 반영
        if isinstance(processed_image, Image.Image):
            new_counter.base_image = np.array(processed_image)
        else:
            new_counter.base_image = processed_image

        new_counter.auto_detected_count = len(new_counter.auto_points)
        new_counter.current_image = new_counter.draw_points()

        if new_counter is not None:
            counter.auto_points = new_counter.auto_points
            counter.auto_annotations = new_counter.auto_annotations
            counter.auto_detected_count = new_counter.auto_detected_count
            counter.current_image = new_counter.current_image
            counter.base_image = new_counter.base_image
            counter.original_image = new_counter.original_image
            counter.last_method = new_counter.last_method
            counter.scale_factor = new_counter.scale_factor

        return new_counter.current_image, new_counter.get_count_text(), new_counter
    except Exception as e:
        print(f"Error in segment_and_count_colonies: {str(e)}")
        if input_image is not None:
            return np.array(input_image), f"Error processing image: {str(e)}", None
        return None, f"Error processing image: {str(e)}", None

def batch_process(
    input_dir,
    output_dir='',
    method='fastsam',
    input_size=1024,
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    mask_random_color=True,
    min_area_percentile=1,
    max_area_percentile=99,
    circularity_threshold=0.8
):
    """여러 이미지 일괄 처리"""
    try:
        if not isinstance(input_dir, str) or not os.path.isdir(input_dir):
            return "Invalid input directory.", []

        if not output_dir or output_dir.strip() == "":
            output_dir = DEFAULT_OUTPUT_DIR

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            return f"Error creating output directory: {str(e)}", []

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(input_dir) 
                       if f.lower().endswith(valid_extensions)]
        
        if not image_files:
            return "No valid image files found in the input directory.", []

        results = []
        progress_text = (f"Processing images from: {input_dir}\n"
                         f"Saving results to: {output_dir}\n\n")
        gallery_items = []
        
        for idx, img_file in enumerate(image_files, 1):
            try:
                progress_text += f"Processing image {idx}/{len(image_files)}: {img_file}\n"
                img_path = os.path.join(input_dir, img_file)

                try:
                    input_image = Image.open(img_path).convert('RGB')
                except Exception as e:
                    raise Exception(f"Error loading image: {str(e)}")
                
                # 이미지 분석
                result_image, count_text, counter = segment_and_count_colonies(
                    input_image=input_image,
                    method=method,
                    input_size=input_size,
                    iou_threshold=iou_threshold,
                    conf_threshold=conf_threshold,
                    better_quality=better_quality,
                    withContours=withContours,
                    mask_random_color=mask_random_color,
                    min_area_percentile=min_area_percentile,
                    max_area_percentile=max_area_percentile,
                    circularity_threshold=circularity_threshold
                )
                
                if counter is not None:
                    save_result = counter.save_results(output_dir, img_file)
                else:
                    save_result = "Failed to process image."

                results.append({
                    'filename': img_file,
                    'count_text': count_text,
                    'save_result': save_result
                })
                
                if result_image is not None:
                    gallery_items.append({
                        'image': result_image,
                        'caption': f"{img_file}\n{count_text}"
                    })
                else:
                    gallery_items.append({
                        'image': None,
                        'caption': f"{img_file}\n{count_text}"
                    })
                
                progress_text += f"Processed {img_file}: {count_text}\n{save_result}\n\n"
            except Exception as e:
                error_msg = f"Error processing {img_file}: {str(e)}\n"
                print(error_msg)
                results.append({
                    'filename': img_file,
                    'count_text': f"Error: {str(e)}",
                    'save_result': "Failed to save results"
                })
                gallery_items.append({
                    'image': None,
                    'caption': f"{img_file}\nError: {str(e)}"
                })
                progress_text += error_msg

        summary_path = os.path.join(output_dir, "batch_summary.csv")
        try:
            pd.DataFrame(results).to_csv(summary_path, index=False)
        except Exception as e:
            progress_text += f"Error saving summary CSV: {str(e)}\n"

        # summary.json 생성
        try:
            create_summary(results, output_dir)
            progress_text += f"\nBatch processing complete. Summary saved to {summary_path} and summary.json"
        except Exception as e:
            progress_text += f"\nError creating summary: {str(e)}"

        return progress_text, gallery_items

    except Exception as e:
        return f"Error in batch processing: {str(e)}", []

def get_output_subdir(output_dir):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    return os.path.join(output_dir, date_str, time_str)

def create_summary(results, output_dir):
    summary = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'time': datetime.now().strftime("%H:%M:%S"),
        'total_images': len(results),
        'successful': sum(1 for r in results if 'Error' not in r['count_text']),
        'failed': sum(1 for r in results if 'Error' in r['count_text']),
        'results': results
    }
    
    summary_json_path = os.path.join(output_dir, "summary.json")
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)

def preprocess_image(image, method='none'):
    """
    이미지 전처리 함수
    Args:
        image: PIL Image 또는 numpy array
        method: 전처리 방법 ('grayscale', 'binary', 'edge', 'sharpen', 'none')
    Returns:
        처리된 PIL Image
    """
    try:
        # PIL Image를 numpy array로 변환
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()

        if method == 'none':
            return Image.fromarray(image_np)

        if method == 'grayscale':
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
            return Image.fromarray(image_np)

        elif method == 'binary':
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))

        elif method == 'edge':
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
            edges = cv2.Canny(gray, 100, 200)
            return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

        elif method == 'sharpen':
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            sharpened = cv2.filter2D(image_np, -1, kernel)
            return Image.fromarray(sharpened)

        return Image.fromarray(image_np)
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        return image

# CSS 스타일 수정
css = """
.container {max-width: 1100px; margin: auto; padding: 20px;}
.header {text-align: center; margin-bottom: 30px;}
.result-text {font-size: 1.2em; font-weight: bold;}
.instruction-box {
    background-color: #000000;
    color: #ffffff;
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
}
.input-image {
    border: 2px solid #ddd;
    margin-top: 10px;
    border-radius: 8px;
    background-color: #f8f9fa;
    max-height: 600px;
}
.output-image {
    border: 2px solid #ddd;
    margin-top: 10px;
    border-radius: 8px;
    background-color: #f8f9fa;
    min-height: 600px;
}
.image-label {
    font-size: 1.1em;
    font-weight: bold;
    margin-bottom: 5px;
    color: #444;
}
.footer {
    text-align: center;
    padding: 20px;
    margin-top: 30px;
    color: #666;
    font-size: 0.9em;
}
"""

# 전역 counter 객체
counter = ColonyCounter()

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown(
        """
        <div class="header">
            <h1>🔬 Advanced Colony Counter</h1>
            <h3>AI-Powered Colony Detection & Analysis</h3>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("Colony Counter"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        """<p class="image-label">Upload Image</p>""",
                        elem_classes="image-label"
                    )
                    input_image = gr.Image(
                        label="",
                        type="pil",
                        elem_classes="input-image",
                        show_label=False,
                        show_download_button=False
                    )
                    
                    # 이미지 전처리 옵션
                    with gr.Row():
                        preprocess_method = gr.Radio(
                            choices=['none', 'grayscale', 'binary', 'edge', 'sharpen'],
                            value='none',
                            label="Image Preprocessing",
                            info="Select preprocessing method"
                        )
                    
                    with gr.Row():
                        preprocess_button = gr.Button(
                            "🎨 Apply Preprocessing",
                            variant="secondary",
                            scale=1
                        )
                        reset_button = gr.Button(
                            "↺ Reset to Original",
                            variant="secondary",
                            scale=1
                        )
                        redo_button = gr.Button(
                            "↶ Undo Preprocessing",
                            variant="secondary",
                            scale=1
                        )

                    with gr.Row():
                        method_select = gr.Radio(
                            choices=['AI Detection'],
                            value='AI Detection',
                            label="Detection Method",
                            info="Advanced AI-based colony detection"
                        )
                        segment_button = gr.Button(
                            "🔍 Analyze Image",
                            variant="primary",
                            scale=2
                        )

                    # 분석 설정 추가
                    with gr.Accordion("⚙️ Analysis Settings", open=False):
                        with gr.Row():
                            input_size_slider = gr.Slider(
                                512, 1024, 1024,
                                step=64,
                                label='Input Size',
                                info='Larger size = better accuracy but slower'
                            )
                            better_quality_checkbox = gr.Checkbox(
                                label="Enhanced Quality",
                                value=True,
                                info="Improve output quality"
                            )
                        
                        with gr.Row():
                            withContours_checkbox = gr.Checkbox(
                                label="Show Contours",
                                value=True,
                                info="Display colony boundaries"
                            )
                            iou_threshold_slider = gr.Slider(
                                0.1, 0.9, 0.7,
                                label="IOU Threshold",
                                info="Higher = stricter detection"
                            )
                        
                        with gr.Row():
                            conf_threshold_slider = gr.Slider(
                                0.1, 0.9, 0.25,
                                label="Confidence Threshold",
                                info="Higher = more confident detection"
                            )
                            circularity_threshold_slider = gr.Slider(
                                0.0, 1.0, 0.8,
                                label="Circularity",
                                info="Higher = more circular colonies"
                            )
                        
                        with gr.Row():
                            min_area_percentile_slider = gr.Slider(
                                0, 10, 1,
                                label="Min Size %",
                                info="Filter smaller colonies"
                            )
                            max_area_percentile_slider = gr.Slider(
                                90, 100, 99,
                                label="Max Size %",
                                info="Filter larger objects"
                            )

                with gr.Column(scale=1):
                    gr.Markdown(
                        """<p class="image-label">Analysis Result</p>""",
                        elem_classes="image-label"
                    )
                    output_image = gr.Image(
                        label="",
                        type="numpy",
                        interactive=True,
                        elem_classes="output-image",
                        show_label=False,
                        show_download_button=True
                    )
                    colony_count_text = gr.Textbox(
                        label="Count Result",
                        lines=3,
                        elem_classes="result-text"
                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            remove_mode_button = gr.Button(
                                "🔄 Toggle Edit Mode",
                                variant="secondary"
                            )
                            remove_mode_text = gr.Textbox(
                                label="Current Mode",
                                value="🟢 Add Mode",
                                lines=1
                            )
                        with gr.Column(scale=1):
                            remove_point_button = gr.Button(
                                "↩️ Undo Last Point",
                                variant="secondary"
                            )
                            undo_removal_button = gr.Button(
                                "♻️ 삭제 복원",
                                variant="secondary"
                            )

        with gr.Tab("Batch Processing"):
            with gr.Row():
                input_dir = gr.Textbox(
                    label="Input Directory",
                    placeholder="Enter the full path to the input directory",
                    lines=1
                )
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value=DEFAULT_OUTPUT_DIR,
                    info="Results will be saved here",
                    placeholder="Enter path or use default",
                    lines=1
                )
            
            with gr.Row():
                batch_process_btn = gr.Button("🔍 Process All Images", scale=2)
                open_folder_btn = gr.Button("📂 Open Output Folder", scale=1)

            with gr.Row():
                batch_progress = gr.Textbox(
                    label="Progress",
                    lines=10,
                    max_lines=15
                )
            
            with gr.Row():
                batch_gallery = gr.Gallery(
                    label="Batch Processing Results",
                    show_label=True,
                    elem_id="gallery",
                    columns=3
                )

            with gr.Row():
                with gr.Accordion("📋 Batch Processing Instructions", open=False):
                    gr.Markdown("""
                    ### How to use batch processing:
                    1. Enter the full path to the input directory containing your images.
                    2. Enter the output directory path or use the default.
                    3. Click 'Process All Images' to start.
                    4. Progress will be shown in the text box below.
                    5. Results for each image will be saved in separate folders and displayed in the gallery.
                    6. A summary CSV and JSON file will be created in the output directory.
                    """)

    # Footer 추가
    gr.Markdown(
        """
        <div class="footer">
            Produced By BDK&copy;
        </div>
        """
    )

    # 전처리 함수 정의
    def handle_image_upload(image):
        """이미지 업로드 처리 최적화"""
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
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # 이미지 히스토리에 저장
            processed_image = image_history.set_original(image)
            return processed_image, "Image uploaded successfully"
        except Exception as e:
            print(f"Error in handle_image_upload: {str(e)}")
            return None, f"Error uploading image: {str(e)}"

    # 전처리 함수 수정
    def apply_preprocessing(image, method):
        """이미지 전처리 적용"""
        if image is None:
            return None
        
        try:
            processed = preprocess_image(image, method)
            if processed is not None:
                return image_history.add_state(processed)
            return image
        except Exception as e:
            print(f"Error in apply_preprocessing: {str(e)}")
            return image

    # Reset 함수
    def reset_image():
        """이미지 초기화"""
        try:
            return image_history.reset()
        except Exception as e:
            print(f"Error in reset_image: {str(e)}")
            return None

    # Redo 함수
    def redo_image():
        """이전 상태로 복원"""
        try:
            return image_history.redo()
        except Exception as e:
            print(f"Error in redo_image: {str(e)}")
            return None

    # 이벤트 핸들러 연결
    input_image.upload(
        handle_image_upload,
        inputs=[input_image],
        outputs=[input_image, colony_count_text]
    )

    preprocess_button.click(
        apply_preprocessing,
        inputs=[input_image, preprocess_method],
        outputs=[input_image]
    )

    reset_button.click(
        reset_image,
        inputs=[],
        outputs=[input_image]
    )

    redo_button.click(
        redo_image,
        inputs=[],
        outputs=[input_image]
    )

    # analyze_image 함수 수정
    def analyze_image(
        input_image, method, input_size=1024, iou_threshold=0.7,
        conf_threshold=0.25, better_quality=True, withContours=True,
        min_area_percentile=1, max_area_percentile=99,
        circularity_threshold=0.8
    ):
        """이미지 분석 함수 최적화"""
        if input_image is None:
            return None, "No input image provided."

        try:
            # 이미지 크기 최적화
            if isinstance(input_image, Image.Image):
                w, h = input_image.size
                if max(w, h) > input_size:
                    scale = input_size / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    input_image = input_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # FastSAM을 내부적으로 사용
            method = 'fastsam'
            processed_image, count_text, new_counter = segment_and_count_colonies(
                input_image=input_image,
                method=method,
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

            if new_counter is not None:
                counter.auto_points = new_counter.auto_points
                counter.auto_annotations = new_counter.auto_annotations
                counter.auto_detected_count = new_counter.auto_detected_count
                counter.current_image = new_counter.current_image
                counter.base_image = new_counter.base_image
                counter.original_image = new_counter.original_image
                counter.last_method = new_counter.last_method
                counter.scale_factor = new_counter.scale_factor

            return processed_image, count_text
        except Exception as e:
            print(f"Error in analyze_image: {str(e)}")
            return None, f"Error analyzing image: {str(e)}"

    segment_button.click(
        analyze_image,
        inputs=[
            input_image,
            method_select,
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

    undo_removal_button.click(
        counter.undo_last_removal,
        inputs=[output_image],
        outputs=[output_image, colony_count_text]
    )

    # 배치 처리 이벤트
    def handle_batch_process(
        input_dir, output_dir, method, input_size, iou_threshold,
        conf_threshold, better_quality, withContours,
        min_area_percentile, max_area_percentile, circularity_threshold
    ):
        # batch_process 내부에서 mask_random_color=True로 하드코딩되어 있으므로
        # 원하면 inputs에서 mask_random_color를 인자로 받아 연동 가능
        progress, gallery = batch_process(
            input_dir=input_dir,
            output_dir=output_dir,
            method=method,
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
        return progress, gallery

    batch_process_btn.click(
        handle_batch_process,
        inputs=[
            input_dir,
            output_dir,
            method_select,
            input_size_slider,
            iou_threshold_slider,
            conf_threshold_slider,
            better_quality_checkbox,
            withContours_checkbox,
            min_area_percentile_slider,
            max_area_percentile_slider,
            circularity_threshold_slider
        ],
        outputs=[batch_progress, batch_gallery]
    )

    def open_output_folder(folder_to_open):
        try:
            if os.path.exists(folder_to_open):
                if os.name == 'nt':  # Windows
                    os.startfile(folder_to_open)
                elif os.name == 'posix':  # macOS, Linux
                    if sys.platform == 'darwin':
                        os.system(f'open "{folder_to_open}"')
                    else:
                        os.system(f'xdg-open "{folder_to_open}"')
                return f"Opened output folder: {folder_to_open}"
            else:
                return f"Output folder does not exist: {folder_to_open}"
        except Exception as e:
            return f"Error opening folder: {str(e)}"

    open_folder_btn.click(
        open_output_folder,
        inputs=[output_dir],
        outputs=[batch_progress]
    )

if __name__ == "__main__":
    demo.launch()
