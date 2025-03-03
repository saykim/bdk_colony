import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
from fastsam import FastSAM, FastSAMPrompt
import torch
import logging
import sys


# 장치 설정
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 모델 경로 설정 및 예외 처리
MODEL_PATH = 'weight/FastSAM-x.pt'
try:
    if not os.path.exists(MODEL_PATH):
        logging.error(f"모델 파일이 존재하지 않습니다: {MODEL_PATH}")
        print(f"오류: 모델 파일을 찾을 수 없습니다. {MODEL_PATH} 경로를 확인하세요.")
        # 모델 파일이 없으면 실행 중단
        sys.exit(1)
    model = FastSAM(MODEL_PATH)
    logging.info(f"모델 로드 성공: {MODEL_PATH}")
except Exception as e:
    logging.error(f"모델 로드 중 오류 발생: {e}")
    print(f"모델 로드 중 오류가 발생했습니다: {e}")
    sys.exit(1)

# 전역 변수로 상태 관리
current_image = None
current_points = []
edit_mode = "Add"  # "Add" 또는 "Remove"
processed_image = None

# 이미지 전처리 함수
def preprocess_image(image, grayscale, binary_threshold, edge_detection, sharpen):
    if image is None:
        return None
    
    try:
        img = np.array(image)
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 다시 RGB로 변환
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
    except Exception as e:
        logging.error(f"이미지 전처리 중 오류 발생: {e}")
        return None

# 자동 카운팅 함수 (FastSAM 기반)
def analyze_image(image, conf_threshold, iou_threshold, circularity_threshold, draw_contours, random_colors):
    global current_image, current_points, processed_image
    
    if image is None:
        return None, "이미지가 로드되지 않았습니다."
    
    try:
        current_image = np.array(image)
        processed_image = current_image.copy()

        # FastSAM으로 객체 탐지
        everything_results = model(current_image, device='cuda' if torch.cuda.is_available() else 'cpu', 
                                  retina_masks=True, imgsz=1024, conf=conf_threshold, iou=iou_threshold)
        
        current_points = []
        for idx, mask in enumerate(everything_results[0].masks.data):
            mask_np = mask.cpu().numpy()
            contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity >= circularity_threshold:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        current_points.append((cX, cY))
                        if draw_contours:
                            color = tuple(np.random.randint(0, 255, 3).tolist()) if random_colors else (0, 255, 0)
                            cv2.drawContours(processed_image, [contour], -1, color, 2)
                        cv2.putText(processed_image, str(len(current_points)), (cX, cY), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return Image.fromarray(processed_image), f"Total colonies: {len(current_points)}"
    except Exception as e:
        logging.error(f"이미지 분석 중 오류 발생: {e}")
        return None, f"이미지 분석 중 오류 발생: {e}"

# 수동 편집 함수 (Gradio 이벤트 처리 방식 수정)
def manual_edit(image, evt: gr.SelectData):
    global current_points, processed_image, edit_mode, current_image
    
    if current_image is None:
        return image, "이미지가 로드되지 않았습니다."
    
    try:
        x, y = evt.index
        
        if processed_image is None:
            processed_image = current_image.copy()
            
        if edit_mode == "Add":
            current_points.append((x, y))
            processed_image = current_image.copy()
            for idx, (px, py) in enumerate(current_points):
                cv2.circle(processed_image, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(processed_image, str(idx + 1), (px, py), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        elif edit_mode == "Remove":
            if current_points:
                distances = [np.sqrt((px - x) ** 2 + (py - y) ** 2) for (px, py) in current_points]
                closest_idx = np.argmin(distances)
                if distances[closest_idx] < 20:  # 근접 거리 임계값
                    current_points.pop(closest_idx)
                    processed_image = current_image.copy()
                    for idx, (px, py) in enumerate(current_points):
                        cv2.circle(processed_image, (px, py), 5, (0, 255, 0), -1)
                        cv2.putText(processed_image, str(idx + 1), (px, py), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return Image.fromarray(processed_image), f"Total colonies: {len(current_points)}"
    except Exception as e:
        logging.error(f"수동 편집 중 오류 발생: {e}")
        return image, f"수동 편집 중 오류 발생: {e}"

# 모드 전환 함수
def toggle_mode(mode):
    global edit_mode
    edit_mode = mode
    return f"Current mode: {edit_mode}"

# 결과 저장 함수
def save_results():
    if processed_image is None:
        return "저장할 결과가 없습니다!"
    
    try:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "result.png")
        cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        with open(os.path.join(output_dir, "count.txt"), "w") as f:
            f.write(f"Total colonies: {len(current_points)}")
        return f"결과가 {output_dir} 폴더에 저장되었습니다."
    except Exception as e:
        logging.error(f"결과 저장 중 오류 발생: {e}")
        return f"결과 저장 중 오류 발생: {e}"

# 배치 처리 함수
def batch_process(folder_path, conf_threshold, iou_threshold, circularity_threshold):
    if not folder_path:
        return "폴더 경로를 입력해주세요."
    
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return f"유효하지 않은 폴더 경로입니다: {folder_path}"
    
    try:
        output_dir = "batch_output"
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not image_files:
            return f"폴더에 이미지 파일이 없습니다: {folder_path}"
        
        for idx, filename in enumerate(image_files):
            try:
                img_path = os.path.join(folder_path, filename)
                image = Image.open(img_path).convert("RGB")
                processed_img, count_text = analyze_image(image, conf_threshold, iou_threshold, 
                                                        circularity_threshold, True, True)
                if processed_img:
                    output_path = os.path.join(output_dir, f"result_{filename}")
                    processed_img.save(output_path)
                    count = int(count_text.split()[-1])
                    results.append({"Image": filename, "Count": count})
                    logging.info(f"이미지 처리 완료 ({idx+1}/{len(image_files)}): {filename}, 카운트: {count}")
                
                # 메모리 관리
                if idx % 10 == 0:
                    import gc
                    gc.collect()
                    
            except Exception as e:
                logging.error(f"이미지 처리 중 오류 발생 ({filename}): {e}")
                results.append({"Image": filename, "Count": "오류"})
        
        if results:
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
            return f"배치 처리 완료. 결과가 {output_dir} 폴더에 저장되었습니다."
        else:
            return "처리된 이미지가 없습니다."
    except Exception as e:
        logging.error(f"배치 처리 중 오류 발생: {e}")
        return f"배치 처리 중 오류 발생: {e}"

# Gradio UI 구성
with gr.Blocks(title="Colony Counter PoC") as demo:
    gr.Markdown("# Colony Counter PoC")
    
    with gr.Row():
        with gr.Column():
            # 이미지 업로드
            image_input = gr.Image(label="Upload Image", type="pil")
            folder_input = gr.Textbox(label="Folder Path (for Batch)", placeholder="e.g., /path/to/folder")
            
            # 전처리 옵션
            grayscale = gr.Checkbox(label="Grayscale", value=False)
            binary_threshold = gr.Slider(0, 255, value=0, label="Binary Threshold")
            edge_detection = gr.Checkbox(label="Edge Detection", value=False)
            sharpen = gr.Slider(0, 1, value=0, label="Sharpen Strength")
            preprocess_btn = gr.Button("Apply Preprocessing")
            
            # 파라미터 설정
            conf_threshold = gr.Slider(0.1, 0.9, value=0.5, label="Confidence Threshold")
            iou_threshold = gr.Slider(0.1, 0.9, value=0.7, label="IOU Threshold")
            circularity_threshold = gr.Slider(0, 1, value=0.5, label="Circularity Threshold")
            draw_contours = gr.Checkbox(label="Draw Contours", value=True)
            random_colors = gr.Checkbox(label="Random Color Masks", value=False)
            
            # 버튼
            analyze_btn = gr.Button("Analyze Image")
            batch_btn = gr.Button("Process All Images")
        
        with gr.Column():
            # 결과 출력
            output_image = gr.Image(label="Analysis Results", interactive=True)
            output_text = gr.Textbox(label="Colony Count")
            
            # 수동 편집
            mode_dropdown = gr.Dropdown(choices=["Add", "Remove"], value="Add", label="Edit Mode")
            save_btn = gr.Button("Save Results")
            status_text = gr.Textbox(label="Status")
    
    # 이벤트 연결 (최신 Gradio 버전에 맞게 수정)
    preprocess_btn.click(
        fn=preprocess_image, 
        inputs=[image_input, grayscale, binary_threshold, edge_detection, sharpen], 
        outputs=image_input
    )
    
    analyze_btn.click(
        fn=analyze_image, 
        inputs=[image_input, conf_threshold, iou_threshold, circularity_threshold, draw_contours, random_colors], 
        outputs=[output_image, output_text]
    )
    
    # 이미지 클릭 이벤트 처리 방식 수정
    output_image.select(
        fn=manual_edit,
        inputs=[output_image],
        outputs=[output_image, output_text]
    )
    
    mode_dropdown.change(
        fn=toggle_mode, 
        inputs=mode_dropdown, 
        outputs=status_text
    )
    
    save_btn.click(
        fn=save_results, 
        inputs=None, 
        outputs=status_text
    )
    
    batch_btn.click(
        fn=batch_process, 
        inputs=[folder_input, conf_threshold, iou_threshold, circularity_threshold], 
        outputs=status_text
    )

if __name__ == "__main__":
    demo.launch()