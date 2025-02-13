from ultralytics import YOLO
import gradio as gr
import torch
from PIL import Image
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
    콜로니 카운터 클래스
    자동/수동 콜로니 감지 및 카운팅을 위한 클래스입니다.
    """
    def __init__(self):
        """
        콜로니 카운터 초기화
        - manual_points: 수동으로 추가된 포인트 목록
        - auto_points: AI가 자동 감지한 colony 중심점 목록
        - current_image: 현재 처리 중인 이미지
        - auto_detected_count: 자동 감지된 콜로니 수
        - remove_mode: 포인트 제거 모드 여부
        - last_method: 마지막으로 사용된 감지 방법
        """
        self.manual_points = []
        self.auto_points = []
        self.current_image = None
        self.auto_detected_count = 0
        self.remove_mode = False
        self.original_image = None
        self.last_method = None

    def reset(self):
        self.manual_points = []
        self.auto_points = []
        self.current_image = None
        self.auto_detected_count = 0
        self.remove_mode = False
        self.original_image = None
        self.last_method = None

    def set_original_image(self, image):
        self.original_image = np.array(image)

    def toggle_remove_mode(self):
        self.remove_mode = not self.remove_mode
        img_with_points = self.draw_points()
        mode_text = "🔴 Remove Mode" if self.remove_mode else "🟢 Add Mode"
        return img_with_points, mode_text

    def find_closest_point(self, x, y, threshold=20):
        """
        클릭한 좌표에서 가장 가까운 포인트를 찾습니다.

        Args:
            x (int): 클릭한 x 좌표
            y (int): 클릭한 y 좌표
            threshold (int): 최대 허용 거리

        Returns:
            tuple: (인덱스, 포인트 타입) 또는 (None, None)
        """
        min_dist = float('inf')
        closest_idx = None
        point_type = None

        # 자동 감지 포인트 확인
        for idx, (px, py) in enumerate(self.auto_points):
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if dist < min_dist and dist <= threshold:
                min_dist = dist
                closest_idx = idx
                point_type = 'auto'

        # 수동 추가 포인트 확인
        for idx, (px, py) in enumerate(self.manual_points):
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if dist < min_dist and dist <= threshold:
                min_dist = dist
                closest_idx = idx
                point_type = 'manual'

        return closest_idx, point_type

    def add_or_remove_point(self, image, evt: gr.SelectData):
        """
        이미지를 클릭하여 포인트를 추가하거나 제거합니다.

        Args:
            image (numpy.ndarray): 현재 이미지
            evt (gr.SelectData): 클릭 이벤트 데이터

        Returns:
            tuple: (업데이트된 이미지, 카운트 텍스트)
        """
        try:
            if self.current_image is None and image is not None:
                self.current_image = np.array(image)

            x, y = evt.index

            if self.remove_mode:
                # 가장 가까운 포인트 찾기
                closest_idx, point_type = self.find_closest_point(x, y)
                
                if closest_idx is not None:
                    if point_type == 'auto':
                        self.auto_points.pop(closest_idx)
                        self.auto_detected_count = len(self.auto_points)
                        print(f"자동 감지 포인트 제거: {closest_idx}")
                    elif point_type == 'manual':
                        self.manual_points.pop(closest_idx)
                        print(f"수동 추가 포인트 제거: {closest_idx}")
                else:
                    print("근처에서 제거할 포인트를 찾을 수 없습니다.")
            else:
                # 포인트 추가
                self.manual_points.append((x, y))
                print(f"새 포인트 추가: ({x}, {y})")

            img_with_points = self.draw_points()
            return img_with_points, self.get_count_text()
        except Exception as e:
            print(f"포인트 추가/제거 중 오류 발생: {str(e)}")
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
            method_text = f"Method: {self.last_method}\n" if self.last_method else ""
            total = self.auto_detected_count + len(self.manual_points)
            return (f"{method_text}Total Colony Count: {total}\n"
                    f"🤖 Auto detected: {self.auto_detected_count}\n"
                    f"👆 Manually added: {len(self.manual_points)}")
        except Exception as e:
            print(f"Error in get_count_text: {str(e)}")
            return "Error calculating count"

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
            font_scale = 1.5
            font_thickness = 2
            
            # 텍스트 크기 계산
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, font_thickness
            )
            
            # 텍스트 위치 계산 (좌하단)
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
            
            if self.current_image is not None:
                result_path = os.path.join(save_dir, "result.png")
                cv2.imwrite(result_path, cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
            
            result_txt = os.path.join(save_dir, "count_results.txt")
            with open(result_txt, 'w') as f:
                f.write(self.get_count_text())
                
            return f"Results saved to {save_dir}"
        except Exception as e:
            return f"Error saving results: {str(e)}"

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
        input_resized = input_image.resize((new_w, new_h))

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

        # 필터링 + 원형도 계산
        valid_colony_annotations = []
        new_counter.auto_points = []  # 자동 감지된 점 초기화

        for ann in colony_annotations:
            ann_cpu = ann.cpu().numpy()
            if ann_cpu.ndim == 3 and ann_cpu.shape[0] == 1:
                ann_cpu = ann_cpu[0]
            mask = ann_cpu > 0
            area = np.sum(mask)

            contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            if contours and len(contours) > 0:
                perimeter = cv2.arcLength(contours[0], True)
                circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
                if circularity >= circularity_threshold:
                    valid_colony_annotations.append(ann)
                    # 마스크 중심점
                    y_indices, x_indices = np.where(mask)
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
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
            new_counter.current_image = np.array(processed_image)
        else:
            new_counter.current_image = processed_image

        new_counter.auto_detected_count = len(new_counter.auto_points)
        new_counter.current_image = new_counter.draw_points()

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
                image = image.resize((new_w, new_h), Image.LANCZOS)
            
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
        circularity_threshold=0.8,
        progress=gr.Progress()
    ):
        """이미지 분석 함수 최적화"""
        if input_image is None:
            return None, "No input image provided."

        try:
            progress(0, desc="초기화 중...")
            
            # 이미지 크기 최적화
            if isinstance(input_image, Image.Image):
                w, h = input_image.size
                if max(w, h) > input_size:
                    progress(0.1, desc="이미지 크기 조정 중...")
                    scale = input_size / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    input_image = input_image.resize((new_w, new_h), Image.LANCZOS)

            progress(0.2, desc="AI 모델 준비 중...")
            # FastSAM을 내부적으로 사용
            method = 'fastsam'
            
            progress(0.3, desc="콜로니 분석 중...")
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

            progress(0.8, desc="결과 처리 중...")
            if new_counter is not None:
                counter.auto_points = new_counter.auto_points
                counter.auto_detected_count = new_counter.auto_detected_count
                counter.current_image = new_counter.current_image
                counter.original_image = new_counter.original_image
                counter.last_method = "AI Detection"

            progress(1.0, desc="완료!")
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
        outputs=[output_image, colony_count_text],
        show_progress=True
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

    # 배치 처리 이벤트
    def handle_batch_process(
        input_dir, output_dir, method, input_size, iou_threshold,
        conf_threshold, better_quality, withContours,
        min_area_percentile, max_area_percentile, circularity_threshold,
        progress=gr.Progress()
    ):
        try:
            progress(0, desc="초기화 중...")
            
            if not isinstance(input_dir, str) or not os.path.isdir(input_dir):
                return "잘못된 입력 디렉토리입니다.", []

            if not output_dir or output_dir.strip() == "":
                output_dir = DEFAULT_OUTPUT_DIR

            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                return f"출력 디렉토리 생성 오류: {str(e)}", []

            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            image_files = [f for f in os.listdir(input_dir) 
                          if f.lower().endswith(valid_extensions)]
            
            if not image_files:
                return "입력 디렉토리에 유효한 이미지 파일이 없습니다.", []

            results = []
            progress_text = (f"처리 중인 디렉토리: {input_dir}\n"
                            f"결과 저장 위치: {output_dir}\n\n")
            gallery_items = []
            
            total_files = len(image_files)
            for idx, img_file in enumerate(image_files, 1):
                try:
                    progress((idx-1)/total_files, desc=f"이미지 처리 중 ({idx}/{total_files}): {img_file}")
                    progress_text += f"처리 중인 이미지 {idx}/{total_files}: {img_file}\n"
                    
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
                        mask_random_color=True,
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

                except Exception as e:
                    error_msg = f"이미지 처리 오류 {img_file}: {str(e)}\n"
                    print(error_msg)
                    progress_text += error_msg
                    results.append({
                        'filename': img_file,
                        'count_text': f"Error: {str(e)}",
                        'save_result': "Failed to save results"
                    })
                    gallery_items.append({
                        'image': None,
                        'caption': f"{img_file}\nError: {str(e)}"
                    })

            progress(0.9, desc="결과 저장 중...")
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

            progress(1.0, desc="완료!")
            return progress_text, gallery_items

        except Exception as e:
            return f"일괄 처리 오류: {str(e)}", []

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
        outputs=[batch_progress, batch_gallery],
        show_progress=True
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