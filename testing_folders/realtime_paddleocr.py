import cv2
import numpy as np
from paddleocr import PaddleOCR

# PaddleOCR 3.x correct API — no text_recognition_lang
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="PP-OCRv5_server_rec",
    device="gpu:0",
)

cap = cv2.VideoCapture(0)
print("Press SPACE to scan, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("PaddleOCR v3 - Press SPACE to scan", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        print("\n--- Scanning... ---")

        # predict() returns a list of result dicts
        results = ocr.predict(frame)

        for res in results:
            if res is None:
                continue
            # In PaddleOCR 3.x, each result has these keys:
            boxes   = res.get("dt_boxes", [])
            texts   = res.get("rec_text", [])
            scores  = res.get("rec_score", [])

            # Handle both flat and nested result formats
            if isinstance(texts, str):
                texts  = [texts]
                scores = [scores]
                boxes  = [boxes]

            for box, text, score in zip(boxes, texts, scores):
                if score > 0.7:
                    print(f"[{score:.2f}] {text}")
                    pts = np.array(box, dtype=np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                    cv2.putText(frame, f"{text}",
                                tuple(pts[0].astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

        cv2.imshow("Result", frame)

cap.release()
cv2.destroyAllWindows()