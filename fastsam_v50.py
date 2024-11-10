from ultralytics import YOLO
import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# FastSAM 모델 로드
model = YOLO('./weights/FastSAM-x.pt')

# 장치 설정
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)

def fast_process(colony_annotations, dish_annotation, image, device, scale, better_quality, mask_random_color, bbox, use_retina, withContours):
    """
    마스크 주석을 기반으로 이미지를 처리하고, 페트리 접시는 외곽선만 그리며 콜로니는 채우고 외곽선을 그립니다.
    """
    image_np = np.array(image).copy()

    # 콜로니 마스크 처리
    for ann in colony_annotations:
        mask = ann.cpu().numpy()
        if mask.ndim == 2:
            mask = mask > 0
            if mask_random_color:
                color = np.random.randint(0, 255, (3,)).tolist()
                image_np[mask] = color
            else:
                image_np[mask] = (0, 255, 0)  # 기본 초록색

        if withContours:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_np, contours, -1, (255, 0, 0), 2)  # 파란색 경계선

    # 페트리 접시 마스크 처리 (외곽선만)
    if dish_annotation is not None:
        dish_mask = dish_annotation.cpu().numpy()
        if dish_mask.ndim == 2:
            dish_mask = dish_mask > 0
            if withContours:
                contours, _ = cv2.findContours(dish_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_np, contours, -1, (0, 0, 255), 3)  # 빨간색 외곽선

    processed_image = Image.fromarray(image_np)
    return processed_image

class ColonyCounter:
    def __init__(self):
        self.manual_points = []  # 수동으로 추가된 포인트 저장
        self.auto_points = []    # 자동 감지된 colony의 중심점 저장
        self.current_image = None
        self.auto_detected_count = 0
        self.remove_mode = False
        self.original_image = None
        self.last_method = None

    def reset(self):
        self.manual_points = []
        self.auto_points = []  # 자동 포인트 초기화 추가
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

    def add_or_remove_point(self, image, evt: gr.SelectData):
        try:
            if self.current_image is None and image is not None:
                self.current_image = np.array(image)

            x, y = evt.index

            if self.remove_mode:
                # 제거 모드: 가장 가까운 점 찾아서 제거
                if self.manual_points:
                    closest_idx = self.find_closest_point(x, y)
                    if closest_idx is not None:
                        self.manual_points.pop(closest_idx)
            else:
                # 추가 모드: 새로운 점 추가
                self.manual_points.append((x, y))

            img_with_points = self.draw_points()
            return img_with_points, self.get_count_text()
        except Exception as e:
            print(f"Error in add_or_remove_point: {str(e)}")
            return image, self.get_count_text()

    def find_closest_point(self, x, y, threshold=20):
        if not self.manual_points:
            return None

        distances = []
        for idx, (px, py) in enumerate(self.manual_points):
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            distances.append((dist, idx))

        closest_dist, closest_idx = min(distances, key=lambda item: item[0])
        return closest_idx if closest_dist < threshold else None

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
            return (f"{method_text}Total Colony Count: {self.auto_detected_count + len(self.manual_points)}\n"
                    f"🤖 Auto detected: {self.auto_detected_count}\n"
                    f"👆 Manually added: {len(self.manual_points)}")
        except Exception as e:
            print(f"Error in get_count_text: {str(e)}")
            return "Error calculating count"

    def draw_points(self):
        try:
            if self.current_image is None:
                return None

            img_with_points = self.current_image.copy()  # 한 번만 복사
            overlay = np.zeros_like(img_with_points)  # 오버레이 레이어 생성
            square_size = 25  # 수동 포인트의 크기 설정

            # Font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 1        # 흰색 글씨를 얇게
            outline_thickness = 3     # 검은색 외곽선을 두껍게

            # 자동 감지된 colony에 번호 표시
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

                #흰색 텍스트
                cv2.putText(img_with_points, text,
                            (text_x, text_y),
                            font, font_scale, (255, 255, 255), font_thickness)
                


            # 수동 포인트 모두 한번에 그리기
            for x, y in self.manual_points:
                pt1 = (int(x - square_size / 2), int(y - square_size / 2))
                pt2 = (int(x + square_size / 2), int(y + square_size / 2))
                cv2.rectangle(overlay, pt1, pt2, (255, 0, 0), -1)

            # 한 번의 addWeighted 연산으로 처리
            cv2.addWeighted(overlay, 0.4, img_with_points, 1.0, 0, img_with_points)

            # 수동 포인트에 번호 표시
            for idx, (x, y) in enumerate(self.manual_points, len(self.auto_points) + 1):
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
                            font, font_scale, (255, 0, 0), font_thickness) # 흰색은 255, 255, 255

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

counter = ColonyCounter()

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
            return None, "No input image provided."

        counter.reset()
        counter.set_original_image(input_image)
        counter.last_method = method.upper()

        # 입력 이미지 크기 조정
        input_size = int(input_size)
        w, h = input_image.size
        scale = input_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        input_resized = input_image.resize((new_w, new_h))

        # FastSAM 모델 실행
        results = model.predict(
            input_resized,
            device=device,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=input_size,
            retina_masks=True
        )

        # 마스크가 없는 경우 처리
        if not hasattr(results[0], 'masks') or results[0].masks is None:
            counter.current_image = np.array(input_resized)
            return np.array(input_resized), "No colonies detected"

        annotations = results[0].masks.data

        if len(annotations) == 0:
            counter.current_image = np.array(input_resized)
            return np.array(input_resized), "No colonies detected"

        # 모든 마스크의 면적 계산
        areas = [np.sum(ann.cpu().numpy() > 0) for ann in annotations]

        # 페트리 접시 마스크 찾기 (가장 큰 마스크)
        dish_idx = np.argmax(areas)
        dish_annotation = annotations[dish_idx]
        colony_annotations = [ann for idx, ann in enumerate(annotations) if idx != dish_idx]

        # 면적 필터링 및 원형도 계산
        valid_colony_annotations = []
        counter.auto_points = []  # 중심점 리스트 초기화

        for ann in colony_annotations:
            mask = ann.cpu().numpy()
            area = np.sum(mask > 0)

            # 원형도 계산
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                perimeter = cv2.arcLength(contours[0], True)
                circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
                if circularity >= circularity_threshold:
                    valid_colony_annotations.append(ann)
                    # 마스크의 중심점 계산
                    y_indices, x_indices = np.where(mask > 0)
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    counter.auto_points.append((center_x, center_y))

        # 이미지 처리
        if valid_colony_annotations:
            fig = fast_process(
                colony_annotations=valid_colony_annotations,
                dish_annotation=dish_annotation,
                image=input_resized,
                device=device,
                scale=(1024 // input_size),
                better_quality=better_quality,
                mask_random_color=mask_random_color,
                bbox=None,
                use_retina=False,
                withContours=withContours
            )
        else:
            fig = input_resized

        # numpy 배열로 변환 및 auto_detected_count 설정
        if isinstance(fig, Image.Image):
            counter.current_image = np.array(fig)
        else:
            counter.current_image = fig

        counter.auto_detected_count = len(counter.auto_points)

        # draw_points를 호출하여 숫자 표시
        counter.current_image = counter.draw_points()

        return counter.current_image, counter.get_count_text()

    except Exception as e:
        print(f"Error in segment_and_count_colonies: {str(e)}")
        if input_image is not None:
            return np.array(input_image), f"Error processing image: {str(e)}"
        return None, f"Error processing image: {str(e)}"

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

# Gradio 인터페이스 설정
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown(
        """
        <div class="header">
            <h1>🔬 Advanced Colony Counter</h1>
            <h3>Automated Colony Detection with Manual Correction</h3>
        </div>
        """
    )

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
                            info="Filter out smaller colonies (1 means smallest 1% filtered)"
                        )
                        max_area_percentile_slider = gr.Slider(
                            90, 100, 99,
                            label="Max Size Percentile",
                            info="Filter out larger objects (99 means largest 1% filtered)"
                        )

                    with gr.Tab("Shape Filters"):
                        circularity_threshold_slider = gr.Slider(
                            0.0, 1.0, 0.8,
                            label="Circularity Threshold",
                            info="Set a threshold to detect only circular colonies (1 = perfect circle)"
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

        with gr.Row(elem_classes="instruction-box"):
            gr.Markdown(
                """
                ### 📝 빠른 가이드
                1. **이미지 업로드**: 분석할 콜로니 이미지를 업로드하세요.
                2. **분석 방법 선택**: FastSAM - AI 기반 탐지 방식을 사용합니다.
                3. **이미지 분석**: Analyze Image 버튼을 눌러 이미지를 분석하세요.
                4. **수동 수정**: 
                   - 👆 이미지를 클릭하여 누락된 콜로니를 추가하세요.
                   - 🔄 'Toggle Edit Mode' 버튼을 사용하여 추가/제거 모드를 전환하세요.
                   - ↩️ 'Undo Last Point' 버튼을 사용하여 최근 추가된 포인트를 제거하세요.
                5. **분석 설정 조정**:
                   - **Input Size**: 입력 이미지 크기를 설정하세요 (크면 정확도 증가, 속도 감소).
                   - **IOU Threshold**: 겹침에 대한 민감도를 설정하세요 (높을수록 겹침이 엄격하게 판단됨).
                   - **Confidence Threshold**: 탐지 신뢰도를 설정하세요 (높을수록 신뢰도가 높은 탐지만 표시됨).
                   - **Min/Max Size Percentile**: 크기 필터를 설정하여 너무 작거나 큰 콜로니를 필터링합니다.
                   - **Circularity Threshold**: 원형도 필터를 설정하여 원형에 가까운 콜로니만 탐지합니다.
                """
            )

    # 이벤트 핸들러 설정
    segment_button.click(
        segment_and_count_colonies,
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

# 앱 실행
demo.launch()
