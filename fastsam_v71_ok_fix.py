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
# warnings.filterwarnings("ignore", category=FutureWarning)

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
    def __init__(self):
        self.manual_points = []  # 수동으로 추가된 포인트 저장
        self.auto_points = []    # 자동 감지된 colony 중심점 저장
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
        # 자동 포인트와 수동 포인트 모두에서 가장 가까운 점 찾기
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
            # auto_points 길이를 기준으로 자동/수동을 구분
            is_auto = (closest_idx < len(self.auto_points))
            return closest_idx, is_auto
        return None, None

    def add_or_remove_point(self, image, evt: gr.SelectData):
        try:
            if self.current_image is None and image is not None:
                self.current_image = np.array(image)

            x, y = evt.index

            if self.remove_mode:
                closest_idx, is_auto = self.find_closest_point(x, y)
                if closest_idx is not None:
                    if is_auto:
                        self.auto_points.pop(closest_idx)
                        self.auto_detected_count -= 1
                    else:
                        manual_idx = closest_idx - len(self.auto_points)
                        self.manual_points.pop(manual_idx)
            else:
                self.manual_points.append((x, y))

            img_with_points = self.draw_points()
            return img_with_points, self.get_count_text()
        except Exception as e:
            print(f"Error in add_or_remove_point: {str(e)}")
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
        try:
            if self.current_image is None:
                return None

            img_with_points = self.current_image.copy()
            overlay = np.zeros_like(img_with_points)
            square_size = 25

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 1
            outline_thickness = 3

            # 자동 감지된 colony 번호 표시
            for idx, (x, y) in enumerate(self.auto_points, 1):
                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = int(x - text_width / 2)
                text_y = int(y - 10)

                # 검은색 외곽선 (4방향)
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    cv2.putText(img_with_points, text,
                                (text_x + dx, text_y + dy),
                                font, font_scale, (0, 0, 0), outline_thickness)

                # 흰색 텍스트
                cv2.putText(img_with_points, text,
                            (text_x, text_y),
                            font, font_scale, (255, 255, 255), font_thickness)

            # 수동 포인트
            for idx, (x, y) in enumerate(self.manual_points, len(self.auto_points) + 1):
                pt1 = (int(x - square_size / 2), int(y - square_size / 2))
                pt2 = (int(x + square_size / 2), int(y + square_size / 2))
                cv2.rectangle(overlay, pt1, pt2, (255, 0, 0), -1)  # 붉은색
                cv2.rectangle(overlay, pt1, pt2, (0, 0, 0), 2)     # 검은색 외곽선

                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = int(x - text_width / 2)
                text_y = int(y - 10)

                # 검은색 외곽선 (4방향)
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    cv2.putText(overlay, text,
                                (text_x + dx, text_y + dy),
                                font, font_scale, (0, 0, 0), outline_thickness)

                # 흰색 텍스트
                cv2.putText(img_with_points, text,
                            (text_x, text_y),
                            font, font_scale, (255, 0, 0), font_thickness)

            cv2.addWeighted(overlay, 0.4, img_with_points, 1.0, 0, img_with_points)

            # 제거 모드 표시
            if self.remove_mode:
                overlay = img_with_points.copy()
                cv2.rectangle(overlay, (0, 0), (img_with_points.shape[1], 40), (255, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, img_with_points, 0.7, 0, img_with_points)
                cv2.putText(img_with_points, "REMOVE MODE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            return img_with_points
        except Exception as e:
            print(f"Error in draw_points: {str(e)}")
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

# CSS 스타일 정의
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
.input-image {border: 2px solid #ddd;}
.output-image {border: 2px solid #ddd;}
"""

# 전역 counter 객체
counter = ColonyCounter()

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown(
        """
        <div class="header">
            <h1>🔬 Advanced Colony Counter</h1>
            <h3>Automated Colony Detection with Manual Correction</h3>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("Colony Counter"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        elem_classes="input-image"
                    )
                    with gr.Row():
                        method_select = gr.Radio(
                            choices=['FastSAM'],
                            value='FastSAM',
                            label="Detection Method",
                            info="Choose AI-based detection method"
                        )
                        segment_button = gr.Button(
                            "🔍 Analyze Image",
                            variant="primary",
                            scale=2
                        )

                    with gr.Accordion("⚙️ Analysis Settings", open=False):
                        with gr.Tab("General"):
                            input_size_slider = gr.Slider(
                                512, 1024, 1024,
                                step=64,
                                label="Input Size",
                                info="Larger size = better accuracy but slower"
                            )
                            better_quality_checkbox = gr.Checkbox(
                                label="Enhanced Quality",
                                value=True,
                                info="Improve output quality at the cost of speed"
                            )
                            withContours_checkbox = gr.Checkbox(
                                label="Show Contours",
                                value=True,
                                info="Display colony boundaries"
                            )

                        with gr.Tab("FastSAM"):
                            iou_threshold_slider = gr.Slider(
                                0.1, 0.9, 0.7,
                                label="IOU Threshold",
                                info="Higher = stricter overlap detection"
                            )
                            conf_threshold_slider = gr.Slider(
                                0.1, 0.9, 0.25,
                                label="Confidence Threshold",
                                info="Higher = more confident detections"
                            )

                        with gr.Tab("Size Filters"):
                            min_area_percentile_slider = gr.Slider(
                                0, 10, 1,
                                label="Min Size Percentile",
                                info="Filter out smaller colonies"
                            )
                            max_area_percentile_slider = gr.Slider(
                                90, 100, 99,
                                label="Max Size Percentile",
                                info="Filter out larger objects"
                            )

                        with gr.Tab("Shape Filters"):
                            circularity_threshold_slider = gr.Slider(
                                0.0, 1.0, 0.8,
                                label="Circularity Threshold",
                                info="Set threshold for circular colonies"
                            )

                with gr.Column(scale=1):
                    output_image = gr.Image(
                        label="Analysis Result",
                        type="numpy",
                        interactive=True,
                        elem_classes="output-image"
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

    # 전역 counter 업데이트를 위한 함수
    def analyze_image(
        input_image, method, input_size, iou_threshold, conf_threshold,
        better_quality, withContours, min_area_percentile, max_area_percentile,
        circularity_threshold
    ):
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
        # 자동 카운팅 결과를 전역 counter에 동기화
        if new_counter is not None:
            counter.auto_points = new_counter.auto_points
            counter.auto_detected_count = new_counter.auto_detected_count
            counter.current_image = new_counter.current_image
            counter.original_image = new_counter.original_image
            counter.last_method = new_counter.last_method
            # 수동 포인트는 유지하거나 초기화할지 결정 가능
            # counter.manual_points = []  # 자동 분석 시 매번 수동 포인트를 리셋하려면 사용

        return processed_image, count_text

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