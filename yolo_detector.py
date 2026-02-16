import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

YOLO_ONNX_URLS = [
    "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx",
]
YOLO_ONNX_FILENAME = "yolov5n.onnx"

DARKNET_CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
DARKNET_WEIGHTS_URL = (
    "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
)
DARKNET_CFG_FILENAME = "yolov4-tiny.cfg"
DARKNET_WEIGHTS_FILENAME = "yolov4-tiny.weights"

DetectionTuple = Tuple[int, int, int, int, float, int]
TrackedDetectionTuple = Tuple[int, int, int, int, float, int, int]


def _default_model_dir() -> Path:
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / "liza_codex_app" / "models"
    return Path(__file__).resolve().parent / "models"


def _letterbox(
    image: np.ndarray,
    new_shape: int | Tuple[int, int] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, int, int]:
    if isinstance(new_shape, tuple):
        new_h, new_w = int(new_shape[0]), int(new_shape[1])
    else:
        new_h = int(new_shape)
        new_w = int(new_shape)

    shape = image.shape[:2]  # (h, w)
    ratio = min(new_h / shape[0], new_w / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))  # (w, h)
    dw = new_w - new_unpad[0]
    dh = new_h - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, left, top


def _download_to_file(url: str, target: Path, min_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=f"{target.name}.", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        urllib.request.urlretrieve(url, str(tmp_path))
        if tmp_path.stat().st_size < min_size:
            head = tmp_path.read_bytes()[:256].lower()
            if b"<html" in head or b"doctype html" in head:
                raise RuntimeError("downloaded HTML page instead of model file")
            raise RuntimeError(f"downloaded file is too small: {tmp_path.stat().st_size} bytes")
        os.replace(str(tmp_path), str(target))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _ensure_file(target: Path, urls: List[str], min_size: int, force: bool = False) -> None:
    if not force and target.exists() and target.stat().st_size >= min_size:
        return

    last_error: Optional[Exception] = None
    for url in urls:
        try:
            _download_to_file(url, target, min_size)
            return
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = exc

    if last_error is None:
        raise RuntimeError(f"No URLs provided for {target.name}")
    raise RuntimeError(f"Failed to download {target.name}: {last_error}") from last_error


def _flatten_indices(indices) -> List[int]:
    if indices is None:
        return []
    if isinstance(indices, np.ndarray):
        return [int(x) for x in indices.flatten().tolist()]
    if not indices:
        return []
    flat: List[int] = []
    for idx in indices:
        if isinstance(idx, (list, tuple, np.ndarray)):
            flat.append(int(idx[0]))
        else:
            flat.append(int(idx))
    return flat


class YoloObjectDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        auto_download: bool = True,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.classes = COCO_CLASSES
        self.backend = ""
        self.net = None
        self.ort_session = None
        self.ort_input_name = ""
        self.ort_input_height = int(input_size)
        self.ort_input_width = int(input_size)
        self.ort_input_dtype = np.float32
        self.ort_normalize = True
        self.ort_output_names: List[str] = []
        self.darknet_output_names: List[str] = []

        if model_path is None:
            model_path = str(_default_model_dir() / YOLO_ONNX_FILENAME)

        self.onnx_path = Path(model_path)
        self.model_dir = self.onnx_path.parent
        self.model_dir.mkdir(parents=True, exist_ok=True)

        onnx_file_error: Optional[Exception] = None
        ort_error: Optional[Exception] = None
        onnx_error: Optional[Exception] = None

        # Ensure ONNX file exists once. Both ORT and OpenCV backends use it.
        try:
            if auto_download:
                _ensure_file(self.onnx_path, YOLO_ONNX_URLS, min_size=1_000_000)
            elif not self.onnx_path.exists():
                raise FileNotFoundError(f"YOLO model not found: {self.onnx_path}")
        except Exception as exc:
            onnx_file_error = exc

        # 1) ONNX Runtime first (best fit for your CPU Windows case).
        if onnx_file_error is None and ort is not None:
            try:
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort_session = ort.InferenceSession(
                    str(self.onnx_path),
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
                ort_input = self.ort_session.get_inputs()[0]
                self.ort_input_name = ort_input.name
                self.ort_input_dtype, self.ort_normalize = self._ort_dtype_from_type(
                    str(getattr(ort_input, "type", "tensor(float)"))
                )
                ort_shape = list(getattr(ort_input, "shape", []))
                if len(ort_shape) >= 4:
                    h_dim = ort_shape[-2]
                    w_dim = ort_shape[-1]
                    if isinstance(h_dim, int) and isinstance(w_dim, int) and h_dim > 0 and w_dim > 0:
                        self.ort_input_height = int(h_dim)
                        self.ort_input_width = int(w_dim)
                        if self.ort_input_height == self.ort_input_width:
                            self.input_size = self.ort_input_height
                self.ort_output_names = [out.name for out in self.ort_session.get_outputs()]
                self.backend = "ort"
            except Exception as exc:
                ort_error = exc

        # 2) OpenCV DNN ONNX fallback.
        if self.backend == "" and onnx_file_error is None:
            try:
                self.net = cv2.dnn.readNetFromONNX(str(self.onnx_path))
                self.backend = "onnx"
            except Exception as exc:
                onnx_error = exc
                if auto_download:
                    try:
                        _ensure_file(self.onnx_path, YOLO_ONNX_URLS, min_size=1_000_000, force=True)
                        self.net = cv2.dnn.readNetFromONNX(str(self.onnx_path))
                        self.backend = "onnx"
                        onnx_error = None
                    except Exception as retry_exc:
                        onnx_error = retry_exc

        # 3) Darknet YOLOv4-tiny fallback.
        if self.backend == "":
            cfg_path = self.model_dir / DARKNET_CFG_FILENAME
            weights_path = self.model_dir / DARKNET_WEIGHTS_FILENAME
            try:
                if auto_download:
                    _ensure_file(cfg_path, [DARKNET_CFG_URL], min_size=1_000)
                    _ensure_file(weights_path, [DARKNET_WEIGHTS_URL], min_size=5_000_000)
                elif not cfg_path.exists() or not weights_path.exists():
                    raise FileNotFoundError(
                        f"Darknet files not found: {cfg_path} and {weights_path}"
                    )

                self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
                self.darknet_output_names = self.net.getUnconnectedOutLayersNames()
                self.backend = "darknet"
            except Exception as darknet_error:
                reason = []
                if onnx_file_error is not None:
                    reason.append(f"Model file step failed: {onnx_file_error}")
                if ort_error is not None:
                    reason.append(f"ONNX Runtime failed: {ort_error}")
                if onnx_error is not None:
                    reason.append(f"OpenCV ONNX failed: {onnx_error}")
                reason.append(f"Darknet fallback failed: {darknet_error}")
                raise RuntimeError("; ".join(reason)) from darknet_error

        if self.net is not None:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @staticmethod
    def _ort_dtype_from_type(type_name: str) -> Tuple[np.dtype, bool]:
        t = type_name.lower()
        if "float16" in t:
            return np.float16, True
        if "float" in t:
            return np.float32, True
        if "uint8" in t:
            return np.uint8, False
        # Fallback works for most YOLO exports.
        return np.float32, True

    def _try_switch_ort_to_onnx(self) -> bool:
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.onnx_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.backend = "onnx"
            self.ort_session = None
            return True
        except Exception:
            return False

    def force_onnx_backend(self) -> bool:
        return self._try_switch_ort_to_onnx()

    def _postprocess_onnx(
        self,
        outputs: np.ndarray,
        ratio: float,
        pad_left: int,
        pad_top: int,
        frame_shape: Tuple[int, int, int],
    ) -> List[DetectionTuple]:
        frame_h, frame_w = frame_shape[:2]

        pred = outputs
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] <= 84 and pred.shape[1] > pred.shape[0]:
            pred = pred.T

        boxes: List[List[int]] = []
        scores: List[float] = []
        class_ids: List[int] = []

        for row in pred:
            if row.shape[0] < 6:
                continue

            # YOLOv5: [cx, cy, w, h, obj_conf, class_probs...]
            # YOLOv8-like ONNX: [cx, cy, w, h, class_probs...]
            if row.shape[0] >= 85:
                obj_conf = float(row[4])
                cls_scores = row[5:]
                class_id = int(np.argmax(cls_scores))
                score = obj_conf * float(cls_scores[class_id])
            else:
                cls_scores = row[4:]
                class_id = int(np.argmax(cls_scores))
                score = float(cls_scores[class_id])

            if score < self.conf_threshold:
                continue

            cx, cy, w, h = map(float, row[:4])
            x1 = (cx - w / 2.0 - pad_left) / ratio
            y1 = (cy - h / 2.0 - pad_top) / ratio
            x2 = (cx + w / 2.0 - pad_left) / ratio
            y2 = (cy + h / 2.0 - pad_top) / ratio

            x1 = max(0, min(int(round(x1)), frame_w - 1))
            y1 = max(0, min(int(round(y1)), frame_h - 1))
            x2 = max(0, min(int(round(x2)), frame_w - 1))
            y2 = max(0, min(int(round(y2)), frame_h - 1))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            boxes.append([x1, y1, bw, bh])
            scores.append(score)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        flat_idx = _flatten_indices(indices)
        if not flat_idx:
            return []

        detections: List[DetectionTuple] = []
        for idx in flat_idx:
            x, y, w, h = boxes[idx]
            detections.append((x, y, x + w, y + h, scores[idx], class_ids[idx]))
        return detections

    def _postprocess_darknet(
        self,
        outputs: List[np.ndarray],
        frame_shape: Tuple[int, int, int],
    ) -> List[DetectionTuple]:
        frame_h, frame_w = frame_shape[:2]

        boxes: List[List[int]] = []
        scores: List[float] = []
        class_ids: List[int] = []

        for output in outputs:
            for det in output:
                if det.shape[0] < 6:
                    continue
                cls_scores = det[5:]
                class_id = int(np.argmax(cls_scores))
                score = float(cls_scores[class_id])
                if score < self.conf_threshold:
                    continue

                cx, cy, w, h = det[:4]
                x1 = int((cx - w / 2.0) * frame_w)
                y1 = int((cy - h / 2.0) * frame_h)
                bw = int(w * frame_w)
                bh = int(h * frame_h)

                x1 = max(0, min(x1, frame_w - 1))
                y1 = max(0, min(y1, frame_h - 1))
                bw = max(1, min(bw, frame_w - x1))
                bh = max(1, min(bh, frame_h - y1))

                boxes.append([x1, y1, bw, bh])
                scores.append(score)
                class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        flat_idx = _flatten_indices(indices)
        if not flat_idx:
            return []

        detections: List[DetectionTuple] = []
        for idx in flat_idx:
            x, y, w, h = boxes[idx]
            detections.append((x, y, x + w, y + h, scores[idx], class_ids[idx]))
        return detections

    @staticmethod
    def _filter_detections(
        detections: List[DetectionTuple],
        frame_shape: Tuple[int, int, int],
    ) -> List[DetectionTuple]:
        if not detections:
            return detections

        h, w = frame_shape[:2]
        frame_area = float(max(1, h * w))
        filtered: List[DetectionTuple] = []
        for x1, y1, x2, y2, score, class_id in detections:
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            area = float(bw * bh)
            area_ratio = area / frame_area
            aspect = float(bw) / float(bh)

            # Remove noisy boxes that cause flicker.
            if area_ratio < 0.00010:
                continue
            if area_ratio > 0.92 and score < 0.90:
                continue
            if aspect < 0.05 or aspect > 20.0:
                continue

            filtered.append((x1, y1, x2, y2, score, class_id))
        return filtered

    @staticmethod
    def _smooth_detections_temporal(
        detections: List[DetectionTuple],
        prev_map: Dict[int, np.ndarray],
        alpha: float = 0.40,
    ) -> Tuple[List[DetectionTuple], Dict[int, np.ndarray]]:
        if not detections:
            return [], {}

        alpha = float(np.clip(alpha, 0.0, 1.0))
        cur_map: Dict[int, np.ndarray] = {}
        out: List[DetectionTuple] = []
        for x1, y1, x2, y2, score, class_id in detections:
            cur = np.array([x1, y1, x2, y2], dtype=np.float32)
            prev = prev_map.get(int(class_id))
            if prev is not None:
                prev_cx = 0.5 * (prev[0] + prev[2])
                prev_cy = 0.5 * (prev[1] + prev[3])
                cur_cx = 0.5 * (cur[0] + cur[2])
                cur_cy = 0.5 * (cur[1] + cur[3])
                move = float(np.hypot(cur_cx - prev_cx, cur_cy - prev_cy))
                prev_w = max(1.0, float(prev[2] - prev[0]))
                prev_h = max(1.0, float(prev[3] - prev[1]))
                prev_diag = float(np.hypot(prev_w, prev_h))
                move_ratio = move / prev_diag

                # Small motion is mostly detector jitter -> stronger smoothing.
                if move_ratio < 0.02:
                    local_alpha = min(alpha, 0.18)
                elif move_ratio < 0.12:
                    local_alpha = alpha
                else:
                    # Fast motion: follow object quickly.
                    local_alpha = max(alpha, 0.78)
                sm = prev * (1.0 - local_alpha) + cur * local_alpha
            else:
                sm = cur
            cur_map[int(class_id)] = sm
            sx1, sy1, sx2, sy2 = [int(round(v)) for v in sm.tolist()]
            out.append((sx1, sy1, sx2, sy2, score, class_id))
        return out, cur_map

    def detect(self, frame: np.ndarray) -> List[DetectionTuple]:
        if self.backend == "ort":
            inp, ratio, pad_left, pad_top = _letterbox(
                frame, (self.ort_input_height, self.ort_input_width)
            )
            # ORT expects NCHW float32 in RGB order.
            ort_input = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            if self.ort_normalize:
                ort_input = ort_input.astype(np.float32) / 255.0
                ort_input = ort_input.astype(self.ort_input_dtype, copy=False)
            else:
                ort_input = ort_input.astype(self.ort_input_dtype, copy=False)
            ort_input = np.transpose(ort_input, (2, 0, 1))[None, ...]
            try:
                outputs = self.ort_session.run(
                    self.ort_output_names if self.ort_output_names else None,
                    {self.ort_input_name: ort_input},
                )
            except Exception as exc:
                # Runtime mismatch (shape/type/provider): fall back to OpenCV ONNX path.
                if self._try_switch_ort_to_onnx():
                    return self.detect(frame)
                raise RuntimeError(f"ORT inference failed: {exc}") from exc
            if not outputs:
                return []
            detections = self._postprocess_onnx(outputs[0], ratio, pad_left, pad_top, frame.shape)
            return self._filter_detections(detections, frame.shape)

        if self.backend == "onnx":
            inp, ratio, pad_left, pad_top = _letterbox(frame, self.input_size)
            blob = cv2.dnn.blobFromImage(
                inp,
                scalefactor=1.0 / 255.0,
                size=(self.input_size, self.input_size),
                swapRB=True,
            )
            self.net.setInput(blob)
            outputs = self.net.forward()
            detections = self._postprocess_onnx(outputs, ratio, pad_left, pad_top, frame.shape)
            return self._filter_detections(detections, frame.shape)

        if self.backend == "darknet":
            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=1.0 / 255.0,
                size=(416, 416),
                swapRB=True,
                crop=False,
            )
            self.net.setInput(blob)
            outputs = self.net.forward(self.darknet_output_names)
            detections = self._postprocess_darknet(outputs, frame.shape)
            return self._filter_detections(detections, frame.shape)

        return []

    def draw(
        self,
        frame: np.ndarray,
        detections: List[Tuple[int, ...]],
        max_boxes: int = 20,
        show_track_id: bool = False,
    ) -> None:
        shown = 0
        for det in detections:
            if shown >= max_boxes:
                break
            if len(det) == 7:
                x1, y1, x2, y2, score, class_id, track_id = det
                id_prefix = f"#{track_id} " if show_track_id else ""
            elif len(det) == 6:
                x1, y1, x2, y2, score, class_id = det
                id_prefix = ""
            else:
                continue
            color = (255, 180, 30)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = self.classes[class_id] if 0 <= class_id < len(self.classes) else str(class_id)
            txt = f"{id_prefix}{label} {score:.2f}"
            cv2.putText(frame, txt, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            shown += 1


class NorfairObjectTracker:
    def __init__(
        self,
        distance_threshold: float = 0.72,
        hit_counter_max: int = 20,
        smoothing: float = 0.30,
        max_missing_updates: int = 3,
        match_iou_threshold: float = 0.15,
        fast_smoothing: float = 0.85,
        fast_motion_ratio: float = 0.35,
    ) -> None:
        try:
            from norfair import Detection as NfDetection
            from norfair import Tracker as NfTracker
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError(
                "Norfair is not installed. Run: pip install norfair"
            ) from exc

        self._Detection = NfDetection
        self._Tracker = NfTracker
        self._distance_threshold = distance_threshold
        self._hit_counter_max = hit_counter_max
        self._max_missing_updates = max(0, int(max_missing_updates))
        self._match_iou_threshold = float(match_iou_threshold)
        self._fast_smoothing = float(np.clip(fast_smoothing, 0.0, 1.0))
        self._fast_motion_ratio = max(0.0, float(fast_motion_ratio))
        self._tracker = self._Tracker(
            distance_function="iou",
            distance_threshold=self._distance_threshold,
            hit_counter_max=self._hit_counter_max,
            initialization_delay=0,
        )
        self._smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self._smoothed_bbox: Dict[int, np.ndarray] = {}
        self._missing_by_id: Dict[int, int] = {}
        self._class_by_id: Dict[int, int] = {}
        self._score_by_id: Dict[int, float] = {}

    def reset(self) -> None:
        self._tracker = self._Tracker(
            distance_function="iou",
            distance_threshold=self._distance_threshold,
            hit_counter_max=self._hit_counter_max,
            initialization_delay=0,
        )
        self._smoothed_bbox.clear()
        self._missing_by_id.clear()
        self._class_by_id.clear()
        self._score_by_id.clear()

    def _clamp_bbox(
        self,
        bbox: np.ndarray,
        frame_shape: Optional[Tuple[int, int, int]],
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox.tolist()]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if frame_shape is not None:
            h, w = frame_shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
        return x1, y1, x2, y2

    @staticmethod
    def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)

    def update(
        self,
        detections: Optional[List[DetectionTuple]],
        frame_shape: Optional[Tuple[int, int, int]] = None,
    ) -> List[TrackedDetectionTuple]:
        detections = detections or []
        norfair_detections = []
        for x1, y1, x2, y2, score, class_id in detections:
            points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            det = self._Detection(
                points=points,
                scores=np.array([score, score], dtype=np.float32),
                data={"class_id": int(class_id), "score": float(score)},
            )
            norfair_detections.append(det)

        tracked = self._tracker.update(detections=norfair_detections)
        results: List[TrackedDetectionTuple] = []
        active_ids: set[int] = set()
        det_boxes = [np.array([d[0], d[1], d[2], d[3]], dtype=np.float32) for d in detections]
        track_info: List[Tuple[int, int, np.ndarray]] = []

        for track_idx, track in enumerate(tracked):
            track_id = int(track.id)
            active_ids.add(track_id)

            estimate = np.array(track.estimate, dtype=np.float32)
            if estimate.shape[0] < 2:
                continue
            raw_bbox = np.array(
                [
                    float(estimate[0][0]),
                    float(estimate[0][1]),
                    float(estimate[1][0]),
                    float(estimate[1][1]),
                ],
                dtype=np.float32,
            )
            track_info.append((track_idx, track_id, raw_bbox))

        # Greedy one-to-one matching track->detection by IoU.
        track_to_det: Dict[int, int] = {}
        if det_boxes and track_info:
            candidates: List[Tuple[float, int, int]] = []
            for track_idx, _, raw_bbox in track_info:
                for det_idx, det_bbox in enumerate(det_boxes):
                    iou = self._iou_xyxy(raw_bbox, det_bbox)
                    if iou >= self._match_iou_threshold:
                        candidates.append((iou, track_idx, det_idx))
            candidates.sort(key=lambda x: x[0], reverse=True)
            used_tracks: set[int] = set()
            used_dets: set[int] = set()
            for _, track_idx, det_idx in candidates:
                if track_idx in used_tracks or det_idx in used_dets:
                    continue
                used_tracks.add(track_idx)
                used_dets.add(det_idx)
                track_to_det[track_idx] = det_idx

        for track_idx, track_id, raw_bbox in track_info:
            matched_det_idx = track_to_det.get(track_idx)
            if matched_det_idx is None:
                miss = self._missing_by_id.get(track_id, 0) + 1
            else:
                miss = 0
            self._missing_by_id[track_id] = miss

            # Keep the track alive internally, but hide stale boxes quickly.
            if miss > self._max_missing_updates:
                continue

            bbox = raw_bbox
            prev = self._smoothed_bbox.get(track_id)
            if prev is None:
                smoothed = bbox
            else:
                prev_cx = 0.5 * (prev[0] + prev[2])
                prev_cy = 0.5 * (prev[1] + prev[3])
                new_cx = 0.5 * (bbox[0] + bbox[2])
                new_cy = 0.5 * (bbox[1] + bbox[3])
                move = float(np.hypot(new_cx - prev_cx, new_cy - prev_cy))
                prev_w = max(1.0, float(prev[2] - prev[0]))
                prev_h = max(1.0, float(prev[3] - prev[1]))
                prev_diag = float(np.hypot(prev_w, prev_h))
                move_ratio = move / max(prev_diag, 1.0)

                alpha = self._fast_smoothing if move_ratio >= self._fast_motion_ratio else self._smoothing
                smoothed = prev * (1.0 - alpha) + bbox * alpha
            self._smoothed_bbox[track_id] = smoothed
            x1, y1, x2, y2 = self._clamp_bbox(smoothed, frame_shape)

            class_id = self._class_by_id.get(track_id, 0)
            score = self._score_by_id.get(track_id, 0.0)
            if matched_det_idx is not None:
                _, _, _, _, det_score, det_class_id = detections[matched_det_idx]
                class_id = int(det_class_id)
                score = float(det_score)
                self._class_by_id[track_id] = class_id
                self._score_by_id[track_id] = score

            results.append((x1, y1, x2, y2, score, class_id, track_id))

        stale_ids = set(self._smoothed_bbox.keys()) - active_ids
        for stale_id in stale_ids:
            self._smoothed_bbox.pop(stale_id, None)
            self._missing_by_id.pop(stale_id, None)
            self._class_by_id.pop(stale_id, None)
            self._score_by_id.pop(stale_id, None)

        return results
