import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import os
import shutil


BASE_DIR = Path(__file__).resolve().parent
OCR_MODEL_DIR = BASE_DIR / "models" / "ocr"


def _clean_text(text: str) -> str:
    cleaned = " ".join((text or "").strip().split())
    return cleaned


def _text_quality_score(text: str) -> float:
    """
    Heuristic score in [0, 1] to suppress OCR gibberish.
    Tesseract can emit random punctuation on blank/noisy frames.
    """
    t = _clean_text(text)
    if not t:
        return 0.0
    if len(t) < 3:
        return 0.0

    letters = sum(ch.isalpha() for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    spaces = sum(ch.isspace() for ch in t)
    total = max(1, len(t) - spaces)
    non_alnum = sum((not ch.isalnum()) and (not ch.isspace()) for ch in t)

    letter_ratio = letters / float(total)
    alnum_ratio = (letters + digits) / float(total)
    symbol_ratio = non_alnum / float(total)

    # Hard rejects
    if letters < 3 and digits < 4:
        return 0.0
    if symbol_ratio > 0.45:
        return 0.0
    if alnum_ratio < 0.55:
        return 0.0

    # Penalize strings that are mostly single-character tokens.
    tokens = [tok for tok in t.split(" ") if tok]
    if tokens:
        single_char = sum(1 for tok in tokens if len(tok) == 1)
        if single_char / float(len(tokens)) > 0.6 and letters < 6:
            return 0.0

    # Soft score: prefer higher letter ratio and longer strings.
    length_bonus = min(1.0, len(t) / 18.0)
    score = (0.65 * letter_ratio + 0.35 * alnum_ratio) * length_bonus
    return float(np.clip(score, 0.0, 1.0))


class LazyOCRRecognizer:
    """
    Lazy OCR recognizer with multi-backend support:
    1) Local HuggingFace vision-text OCR model in backend/models/ocr
    2) pytesseract fallback (if installed and Tesseract binary is available)

    Returns (text, confidence) where confidence is [0, 1].
    """

    def __init__(self):
        self._backend: Optional[str] = None
        self._processor = None
        self._model = None
        self._model_type: Optional[str] = None
        self._paddle_ocr = None
        self._pytesseract = None
        self._easyocr = None
        self._easyocr_reader = None

    def _ensure_loaded(self):
        if self._backend is not None:
            return

        prefer = (os.getenv("GABAI_OCR_BACKEND") or "").strip().lower()
        # Supported values: "paddleocr", "glm", "deepseek", "pytesseract", "easyocr", "auto"
        # Default to pytesseract (lightweight) unless explicitly overridden.
        if not prefer:
            prefer = "pytesseract"
        if prefer not in ("paddleocr", "glm", "deepseek", "pytesseract", "easyocr", "auto"):
            prefer = "pytesseract"

        def _pick_device_and_dtype(torch):
            use_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
            device = "cuda" if use_cuda else "cpu"
            # bfloat16 is often unsupported on consumer Windows GPUs; prefer fp16 on CUDA.
            dtype = torch.float16 if use_cuda else torch.float32
            return device, dtype

        def _load_local_hf_model(force_type: Optional[str] = None) -> bool:
            """
            Loads a local HuggingFace model bundle from OCR_MODEL_DIR.
            Supports:
            - GLM-OCR (model_type: glm_ocr)
            - DeepSeek OCR variants (legacy folder content)
            """
            try:
                import torch  # type: ignore
                from transformers import AutoConfig, AutoProcessor  # type: ignore

                cfg = AutoConfig.from_pretrained(
                    str(OCR_MODEL_DIR),
                    local_files_only=True,
                    trust_remote_code=True,
                )
                model_type = str(getattr(cfg, "model_type", "") or "").strip().lower()
                if force_type:
                    model_type = force_type

                self._processor = AutoProcessor.from_pretrained(
                    str(OCR_MODEL_DIR),
                    local_files_only=True,
                    trust_remote_code=True,
                )

                device, dtype = _pick_device_and_dtype(torch)

                # Force eager attention to avoid flash-attn issues on Windows.
                if hasattr(cfg, "_attn_implementation"):
                    try:
                        setattr(cfg, "_attn_implementation", "eager")
                    except Exception:
                        pass

                if model_type == "glm_ocr":
                    from transformers import AutoModelForImageTextToText  # type: ignore

                    self._model = AutoModelForImageTextToText.from_pretrained(
                        str(OCR_MODEL_DIR),
                        local_files_only=True,
                        trust_remote_code=True,
                        config=cfg,
                        torch_dtype=dtype,
                        device_map="auto" if device == "cuda" else None,
                    )
                    if device != "cuda":
                        self._model = self._model.to(device=device, dtype=dtype)
                    self._model.eval()
                    self._backend = "glm_ocr_local"
                    self._model_type = "glm_ocr"
                    print(f"[OCR] Loaded local GLM-OCR model from: {OCR_MODEL_DIR} ({device})")
                    return True

                # Default: legacy DeepSeek-style model code.
                from transformers import AutoModel  # type: ignore

                self._model = AutoModel.from_pretrained(
                    str(OCR_MODEL_DIR),
                    local_files_only=True,
                    trust_remote_code=True,
                    config=cfg,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                )
                if device != "cuda":
                    self._model = self._model.to(device=device, dtype=dtype)
                self._model.eval()
                self._backend = "deepseek_ocr_local"
                self._model_type = model_type or "unknown"
                print(f"[OCR] Loaded local DeepSeek OCR model from: {OCR_MODEL_DIR} ({device})")
                return True
            except Exception as exc:
                print(f"[OCR] Local OCR model load failed: {exc}")
                return False

        def _try_deepseek():
            return _load_local_hf_model(force_type=None)

        def _try_glm():
            return _load_local_hf_model(force_type="glm_ocr")

        def _try_pytesseract():
            try:
                import pytesseract  # type: ignore
                from pytesseract.pytesseract import TesseractNotFoundError  # type: ignore

                # Auto-detect tesseract.exe on Windows to avoid PATH confusion.
                # If we can't find it, treat pytesseract as unavailable so we can fall back cleanly.
                def _configure_tesseract_cmd():
                    # 1) If already in PATH, we're done.
                    exe = shutil.which("tesseract")
                    if exe:
                        pytesseract.pytesseract.tesseract_cmd = exe
                        return

                    # 2) Common installer locations.
                    candidates = [
                        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                        # Some installers use a versioned folder:
                        r"C:\Program Files\Tesseract-OCR\bin\tesseract.exe",
                    ]
                    for c in candidates:
                        if os.path.exists(c):
                            pytesseract.pytesseract.tesseract_cmd = c
                            return

                _configure_tesseract_cmd()
                # Validate that the binary is actually reachable.
                try:
                    _ = pytesseract.get_tesseract_version()
                except TesseractNotFoundError as exc:
                    raise RuntimeError(str(exc))

                self._pytesseract = pytesseract
                self._backend = "pytesseract"
                print("[OCR] Using pytesseract fallback")
                return True
            except Exception as exc:
                print(f"[OCR] pytesseract fallback unavailable: {exc}")
                return False

        def _try_easyocr():
            try:
                import easyocr  # type: ignore
                try:
                    import torch  # type: ignore
                    use_gpu = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
                except Exception:
                    use_gpu = False

                self._easyocr = easyocr
                # English-only by default; extend as needed.
                self._easyocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
                self._backend = "easyocr"
                print(f"[OCR] Using EasyOCR fallback ({'GPU' if use_gpu else 'CPU'})")
                return True
            except Exception as exc:
                print(f"[OCR] EasyOCR fallback unavailable: {exc}")
                return False

        def _try_paddleocr():
            """
            PaddleOCR pipeline (det+rec) using PP-OCRv5 mobile models.
            Requires `paddleocr` + `paddlepaddle(-gpu)` installed in the environment.
            """
            try:
                from paddleocr import PaddleOCR  # type: ignore
                # Device format in PaddleOCR is like "gpu:0" or "cpu".
                device = (os.getenv("GABAI_PADDLE_DEVICE") or "gpu:0").strip()
                if not device:
                    device = "gpu:0"

                det_name = (os.getenv("GABAI_PADDLE_DET") or "PP-OCRv5_mobile_det").strip()
                rec_name = (os.getenv("GABAI_PADDLE_REC") or "PP-OCRv5_mobile_rec").strip()

                # Keep optional modules off for speed; enable textline orientation (helps rotated lines).
                self._paddle_ocr = PaddleOCR(
                    text_detection_model_name=det_name,
                    text_recognition_model_name=rec_name,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    device=device,
                )
                self._backend = "paddleocr"
                print(f"[OCR] Using PaddleOCR ({det_name} + {rec_name}) on {device}")
                return True
            except Exception as exc:
                print(f"[OCR] PaddleOCR backend unavailable: {exc}")
                return False

        # Respect explicit user preference.
        if prefer == "easyocr":
            if _try_easyocr():
                return
        elif prefer == "paddleocr":
            if _try_paddleocr():
                return
        elif prefer == "pytesseract":
            if _try_pytesseract():
                return
        elif prefer == "glm":
            if _try_glm():
                return
        elif prefer == "deepseek":
            if _try_deepseek():
                return

        # Auto mode (and safety net): prefer lightweight OCR first.
        if _try_paddleocr():
            return
        if _try_pytesseract():
            return

        # This avoids requiring the separate Tesseract binary on Windows.
        if _try_easyocr():
            return

        # Heavy local models last (kept optional).
        if _try_glm():
            return
        if _try_deepseek():
            return

        self._backend = "none"
        print("[OCR] No OCR backend available")

    def infer(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        self._ensure_loaded()

        if self._backend == "paddleocr" and self._paddle_ocr is not None:
            try:
                # PaddleOCR expects images as numpy arrays (OpenCV BGR is ok).
                img = frame_bgr

                texts = []
                scores = []

                # Newer PaddleOCR inference package exposes `.predict(...)`.
                if hasattr(self._paddle_ocr, "predict"):
                    output = self._paddle_ocr.predict(input=img, batch_size=1)
                    for res in output or []:
                        # `res` can be a result object/dict depending on version.
                        payload = getattr(res, "res", None) or res.get("res") if isinstance(res, dict) else None
                        if not payload:
                            continue
                        rec_texts = payload.get("rec_texts") if isinstance(payload, dict) else None
                        rec_scores = payload.get("rec_scores") if isinstance(payload, dict) else None
                        if rec_texts:
                            for i, t in enumerate(list(rec_texts)):
                                t = _clean_text(str(t))
                                if not t:
                                    continue
                                texts.append(t)
                                try:
                                    s = float(rec_scores[i]) if rec_scores is not None else 0.8
                                except Exception:
                                    s = 0.8
                                scores.append(s)
                else:
                    # Classic PaddleOCR API: `.ocr(img, cls=..., det=..., rec=...)`
                    ocr_res = self._paddle_ocr.ocr(img, cls=False)
                    # Expected: [ [ [box], (text, score) ], ... ] or nested by page.
                    flat = ocr_res
                    if isinstance(ocr_res, list) and ocr_res and isinstance(ocr_res[0], list) and ocr_res and ocr_res and ocr_res and (
                        len(ocr_res) == 1 and isinstance(ocr_res[0], list)
                    ):
                        flat = ocr_res[0]
                    for item in flat or []:
                        if not item or len(item) < 2:
                            continue
                        rec = item[1]
                        if not rec or len(rec) < 1:
                            continue
                        t = _clean_text(str(rec[0]))
                        if not t:
                            continue
                        texts.append(t)
                        try:
                            scores.append(float(rec[1]) if len(rec) > 1 else 0.8)
                        except Exception:
                            scores.append(0.8)

                merged = _clean_text(" ".join(texts))
                if not merged:
                    return None, 0.0

                base_conf = float(np.clip(float(np.mean(scores)) if scores else 0.7, 0.0, 1.0))
                quality = _text_quality_score(merged)
                confidence = float(np.clip(base_conf * (0.7 + 0.3 * quality), 0.0, 1.0))
                # Suppress rare gibberish outputs.
                if quality < 0.25 and confidence < 0.35:
                    return None, 0.0
                return merged, confidence
            except Exception as exc:
                print(f"[OCR] PaddleOCR inference failed: {exc}")
                return None, 0.0

        if self._backend == "glm_ocr_local" and self._processor is not None and self._model is not None:
            try:
                import torch  # type: ignore
                from PIL import Image  # type: ignore

                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            # GLM-OCR processors vary across versions; try in-memory image first.
                            {"type": "image", "image": pil},
                            {"type": "text", "text": "Text Recognition:"},
                        ],
                    }
                ]

                # Prefer chat-template path when available (matches upstream README).
                if hasattr(self._processor, "apply_chat_template"):
                    try:
                        inputs = self._processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt",
                        )
                    except Exception:
                        # Some processors only support URL/path images in templates.
                        # Fall back to direct processor call below.
                        inputs = None
                else:
                    inputs = None

                if inputs is None:
                    # Fallback: direct image+text call (widest compatibility).
                    inputs = self._processor(images=pil, text="Text Recognition:", return_tensors="pt")

                # Move tensors to model device (BatchEncoding-safe).
                device = getattr(self._model, "device", None)
                if device is not None and hasattr(inputs, "to"):
                    inputs = inputs.to(device)
                elif device is not None and isinstance(inputs, dict):
                    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                # Some GLM processors include token_type_ids which can be None or unsupported.
                try:
                    if isinstance(inputs, dict) and inputs.get("token_type_ids", "missing") is None:
                        inputs.pop("token_type_ids", None)
                except Exception:
                    pass

                with torch.inference_mode():
                    generated = self._model.generate(**inputs, max_new_tokens=128)

                # Decode only the newly generated tokens when possible.
                if isinstance(inputs, dict) and "input_ids" in inputs:
                    prompt_len = int(inputs["input_ids"].shape[1])
                    gen_ids = generated[0][prompt_len:]
                else:
                    try:
                        prompt_len = int(inputs["input_ids"].shape[1])  # type: ignore[index]
                        gen_ids = generated[0][prompt_len:]
                    except Exception:
                        gen_ids = generated[0]

                text = self._processor.decode(gen_ids, skip_special_tokens=True)
                text = _clean_text(text)
                if not text:
                    return None, 0.0
                return text, 0.9
            except Exception as exc:
                print(f"[OCR] Local GLM-OCR inference failed: {exc}")
                return None, 0.0

        if self._backend == "deepseek_ocr_local" and self._processor is not None and self._model is not None:
            try:
                import torch  # type: ignore
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                inputs = self._processor(images=rgb, return_tensors="pt")
                if hasattr(self._model, "device"):
                    inputs = {k: v.to(self._model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                generated = self._model.generate(**inputs, max_new_tokens=48)
                text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
                text = _clean_text(text)
                if not text:
                    return None, 0.0
                return text, 0.9
            except Exception as exc:
                print(f"[OCR] Local DeepSeek OCR inference failed: {exc}")
                return None, 0.0

        if self._backend == "pytesseract" and self._pytesseract is not None:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                data = self._pytesseract.image_to_data(
                    th,
                    output_type=self._pytesseract.Output.DICT,
                    # PSM 6 assumes a block of text; works decently for signs/labels.
                    config="--oem 3 --psm 6",
                )
                words = []
                confs = []
                for txt, conf in zip(data.get("text", []), data.get("conf", [])):
                    txt = _clean_text(str(txt))
                    if not txt:
                        continue
                    try:
                        c = float(conf)
                    except Exception:
                        c = -1.0
                    if c < 0:
                        continue
                    # Suppress low-confidence fragments (common on blank frames).
                    if c < 55:
                        continue
                    words.append(txt)
                    confs.append(c / 100.0)

                text = _clean_text(" ".join(words))
                if not text:
                    return None, 0.0
                base_conf = float(np.clip(np.mean(confs), 0.0, 1.0)) if confs else 0.5
                quality = _text_quality_score(text)
                confidence = float(np.clip(base_conf * quality, 0.0, 1.0))
                # If it looks like gibberish, treat as no text.
                if quality < 0.35 or confidence < 0.25:
                    return None, 0.0
                return text, confidence
            except Exception as exc:
                msg = str(exc)
                print(f"[OCR] pytesseract inference failed: {exc}")
                # If Tesseract binary is missing, don't spam every frame—fall back to EasyOCR.
                if "tesseract is not installed" in msg.lower() or "tesseractnotfounderror" in msg.lower():
                    self._backend = None
                    self._pytesseract = None
                return None, 0.0

        if self._backend == "easyocr" and self._easyocr_reader is not None:
            try:
                # EasyOCR expects RGB or grayscale; we pass RGB.
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # detail=1 returns [ [bbox, text, conf], ... ]
                results = self._easyocr_reader.readtext(rgb, detail=1, paragraph=True)
                if not results:
                    return None, 0.0

                texts = []
                confs = []
                for item in results:
                    if not item or len(item) < 3:
                        continue
                    text = _clean_text(str(item[1]))
                    if not text:
                        continue
                    try:
                        conf = float(item[2])
                    except Exception:
                        conf = 0.0
                    texts.append(text)
                    confs.append(conf)

                merged = _clean_text(" ".join(texts))
                if not merged:
                    return None, 0.0
                confidence = float(np.clip(np.mean(confs), 0.0, 1.0)) if confs else 0.5
                return merged, confidence
            except Exception as exc:
                print(f"[OCR] EasyOCR inference failed: {exc}")
                return None, 0.0

        return None, 0.0


@dataclass
class OCRDecision:
    nav_command: str
    confidence: float
    should_speak: bool


class OCRAssistant:
    """
    Stabilizes OCR text so the app doesn't spam noisy frame-by-frame changes.
    """

    def __init__(
        self,
        speak_cooldown_s: float = 1.4,
        hold_frames: int = 2,
        # Printed text on-camera can be noisy; allow lower confidence but rely on stability.
        min_conf: float = 0.20,
        unknown_label: str = "No text",
    ):
        self._last_spoken: str = ""
        self._last_spoken_at: float = 0.0

        self._candidate: str = ""
        self._candidate_frames: int = 0
        self._candidate_conf: float = 0.0

        self._speak_cooldown_s = speak_cooldown_s
        self._hold_frames = hold_frames
        self._min_conf = min_conf
        self._unknown_label = unknown_label

    def decide(self, text: Optional[str], confidence: float) -> OCRDecision:
        now = time.monotonic()

        nav_command = text if (text and confidence >= self._min_conf) else self._unknown_label
        nav_command = _clean_text(nav_command)

        if nav_command == self._candidate:
            self._candidate_frames += 1
            self._candidate_conf = confidence
        else:
            self._candidate = nav_command
            self._candidate_frames = 1
            self._candidate_conf = confidence

        stable = self._candidate_frames >= self._hold_frames
        cooldown_ok = (now - self._last_spoken_at) >= self._speak_cooldown_s
        changed = nav_command != self._last_spoken
        should_speak = stable and cooldown_ok and changed and nav_command != self._unknown_label

        if should_speak:
            self._last_spoken = nav_command
            self._last_spoken_at = now

        return OCRDecision(
            nav_command=nav_command,
            confidence=float(self._candidate_conf),
            should_speak=should_speak,
        )


ocr_recognizer = LazyOCRRecognizer()
ocr_assistant = OCRAssistant()
