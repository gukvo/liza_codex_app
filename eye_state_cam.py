import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit(
        "mediapipe is not installed. Run: pip install mediapipe"
    ) from exc


def _get_face_mesh_module():
    if hasattr(mp, "solutions"):
        return mp.solutions.face_mesh
    try:
        import importlib

        mp_solutions = importlib.import_module("mediapipe.solutions")
        return mp_solutions.face_mesh
    except Exception:
        pass
    try:
        from mediapipe.python import solutions as mp_solutions
    except Exception as exc:  # pragma: no cover - environment-specific
        raise SystemExit(
            "mediapipe import succeeded, but FaceMesh 'solutions' is unavailable. "
            "This can happen on Windows with Python 3.12 or if the install is broken. "
            "Try reinstalling mediapipe and pin numpy==1.26.4. "
            "If you're on 3.12, use Python 3.11 or 3.10."
        ) from exc
    return mp_solutions.face_mesh


def _get_hands_module():
    if hasattr(mp, "solutions"):
        return mp.solutions.hands
    try:
        import importlib

        mp_solutions = importlib.import_module("mediapipe.solutions")
        return mp_solutions.hands
    except Exception:
        pass
    try:
        from mediapipe.python import solutions as mp_solutions
    except Exception as exc:  # pragma: no cover - environment-specific
        raise SystemExit(
            "mediapipe import succeeded, but Hands 'solutions' is unavailable. "
            "Try reinstalling mediapipe and pin numpy==1.26.4."
        ) from exc
    return mp_solutions.hands


try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

_FONT_PATH: Optional[str] = None
_PIL_FONT: Optional["ImageFont.FreeTypeFont"] = None
_FREETYPE = None
SHOW_HELMET_BOX = False


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _ear(pts: np.ndarray, idxs: list[int]) -> float:
    p1 = tuple(pts[idxs[0]])
    p2 = tuple(pts[idxs[1]])
    p3 = tuple(pts[idxs[2]])
    p4 = tuple(pts[idxs[3]])
    p5 = tuple(pts[idxs[4]])
    p6 = tuple(pts[idxs[5]])
    denom = 2.0 * _dist(p1, p4)
    if denom == 0.0:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / denom


def _landmarks_to_xy(landmarks, w: int, h: int) -> np.ndarray:
    pts = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))
    return np.array(pts, dtype=np.int32)


def _draw_eye_points(frame, pts: np.ndarray, idxs: list[int]) -> None:
    for i in idxs:
        cv2.circle(frame, tuple(pts[i]), 2, (0, 255, 0), -1)


def _find_font_path() -> Optional[str]:
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _get_pil_font(size: int = 20):
    global _FONT_PATH, _PIL_FONT
    if _PIL_FONT is not None:
        return _PIL_FONT
    if not PIL_AVAILABLE:
        return None
    if _FONT_PATH is None:
        _FONT_PATH = _find_font_path()
    if _FONT_PATH is None:
        return None
    try:
        _PIL_FONT = ImageFont.truetype(_FONT_PATH, size=size)
        return _PIL_FONT
    except Exception:
        return None


def _get_freetype():
    global _FREETYPE
    if _FREETYPE is not None:
        return _FREETYPE
    if not hasattr(cv2, "freetype"):
        _FREETYPE = False
        return _FREETYPE
    font_path = _find_font_path()
    if not font_path:
        _FREETYPE = False
        return _FREETYPE
    try:
        ft = cv2.freetype.createFreeType2()
        ft.loadFontData(fontFileName=font_path, id=0)
        _FREETYPE = ft
    except Exception:
        _FREETYPE = False
    return _FREETYPE


def _queue_text(texts: list[tuple[str, int, tuple[int, int, int]]], text: str, y: int,
                color=(255, 255, 255)) -> None:
    texts.append((text, y, color))


def _draw_texts(frame, texts: list[tuple[str, int, tuple[int, int, int]]]) -> None:
    if not texts:
        return
    ft = _get_freetype()
    if ft:
        for text, y, color in texts:
            org = (10, y)
            ft.putText(frame, text, org, fontHeight=20, color=(0, 0, 0),
                       thickness=3, line_type=cv2.LINE_AA, bottomLeftOrigin=False)
            ft.putText(frame, text, org, fontHeight=20, color=color,
                       thickness=1, line_type=cv2.LINE_AA, bottomLeftOrigin=False)
        return
    font = _get_pil_font(size=20)
    if PIL_AVAILABLE and font is not None:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for text, y, color in texts:
            x = 10
            draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
            draw.text((x, y), text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
        frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return
    for text, y, color in texts:
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def _eye_bbox(pts: np.ndarray, idxs: list[int]) -> Tuple[int, int, int, int]:
    xs = pts[idxs, 0]
    ys = pts[idxs, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _eye_center(pts: np.ndarray, idxs: list[int]) -> Tuple[int, int]:
    xs = pts[idxs, 0]
    ys = pts[idxs, 1]
    return int(xs.mean()), int(ys.mean())


def _face_bbox(pts: np.ndarray) -> Tuple[int, int, int, int]:
    x1 = int(pts[:, 0].min())
    y1 = int(pts[:, 1].min())
    x2 = int(pts[:, 0].max())
    y2 = int(pts[:, 1].max())
    return x1, y1, x2, y2


def _select_primary_face(face_pts_list: list[np.ndarray], width: int, height: int) -> int:
    frame_cx = width * 0.5
    frame_cy = height * 0.5
    frame_area = float(width * height)
    frame_diag = max(float(np.hypot(width, height)), 1.0)

    best_idx = 0
    best_score = -1e18
    for idx, pts in enumerate(face_pts_list):
        x1, y1, x2, y2 = _face_bbox(pts)
        face_w = max(1.0, float(x2 - x1))
        face_h = max(1.0, float(y2 - y1))
        area = face_w * face_h
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        dist_norm = float(np.hypot(cx - frame_cx, cy - frame_cy)) / frame_diag

        # Prefer larger and more centered face.
        score = area - dist_norm * (0.35 * frame_area)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _hands_to_xy(hand_landmarks, w: int, h: int) -> list[Tuple[int, int]]:
    points = []
    for hand in hand_landmarks:
        for lm in hand.landmark:
            points.append((int(lm.x * w), int(lm.y * h)))
    return points


def _is_eye_covered(
    eye_bbox: Tuple[int, int, int, int],
    eye_center: Tuple[int, int],
    eye_width: float,
    hand_points: list[Tuple[int, int]],
) -> bool:
    if not hand_points:
        return False
    x1, y1, x2, y2 = eye_bbox
    margin = max(int(eye_width * 0.6), 8)
    x1 -= margin
    y1 -= margin
    x2 += margin
    y2 += margin
    cx, cy = eye_center
    radius = max(eye_width * 0.75, 12.0)
    for hx, hy in hand_points:
        if x1 <= hx <= x2 and y1 <= hy <= y2:
            return True
        if _dist((hx, hy), (cx, cy)) <= radius:
            return True
    return False


def _clamp_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return x1, y1, x2, y2


def _build_helmet_roi(pts: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x_min = int(pts[:, 0].min())
    x_max = int(pts[:, 0].max())
    y_min = int(pts[:, 1].min())
    y_max = int(pts[:, 1].max())
    face_w = x_max - x_min
    face_h = y_max - y_min
    if face_w < 40 or face_h < 40:
        return None

    # Focus on central area above forehead where helmet shell is expected.
    rx1 = int(x_min + face_w * 0.20)
    rx2 = int(x_max - face_w * 0.20)
    ry1 = int(y_min - face_h * 0.62)
    ry2 = int(y_min + face_h * 0.08)
    return _clamp_box(rx1, ry1, rx2, ry2, width, height)


def _helmet_mask_from_hsv(hsv: np.ndarray) -> np.ndarray:
    # Common hard-hat colors: yellow/orange/red/blue/white.
    yellow = cv2.inRange(hsv, (18, 80, 70), (40, 255, 255))
    orange = cv2.inRange(hsv, (5, 90, 70), (17, 255, 255))
    red1 = cv2.inRange(hsv, (0, 90, 70), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 90, 70), (180, 255, 255))
    blue = cv2.inRange(hsv, (90, 70, 60), (130, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 185), (180, 65, 255))
    mask = yellow | orange | red1 | red2 | blue | white
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _skin_mask(roi_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    skin_hsv = cv2.inRange(hsv, (0, 20, 50), (25, 220, 255))
    skin_ycrcb = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    skin = cv2.bitwise_and(skin_hsv, skin_ycrcb)
    kernel = np.ones((3, 3), dtype=np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel, iterations=1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=1)
    return skin


def _detect_helmet(frame: np.ndarray, pts: np.ndarray) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
    h, w = frame.shape[:2]
    roi_box = _build_helmet_roi(pts, w, h)
    if roi_box is None:
        return False, 0.0, None

    x1, y1, x2, y2 = roi_box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0, roi_box

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = _helmet_mask_from_hsv(hsv)
    skin = _skin_mask(roi)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin))

    total = float(mask.shape[0] * mask.shape[1])
    color_ratio = float(np.count_nonzero(mask)) / total
    top_rows = max(1, int(mask.shape[0] * 0.55))
    top_ratio = float(np.count_nonzero(mask[:top_rows, :])) / float(top_rows * mask.shape[1])
    bottom_rows = max(1, int(mask.shape[0] * 0.35))
    bottom_ratio = float(np.count_nonzero(mask[-bottom_rows:, :])) / float(bottom_rows * mask.shape[1])

    center_x1 = int(mask.shape[1] * 0.20)
    center_x2 = int(mask.shape[1] * 0.80)
    center_y2 = int(mask.shape[0] * 0.75)
    center = mask[:center_y2, center_x1:center_x2]
    center_ratio = float(np.count_nonzero(center)) / float(center.size) if center.size else 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area_ratio = 0.0
    max_width_ratio = 0.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        max_area = cv2.contourArea(largest)
        max_area_ratio = float(max_area) / total
        _, _, cw, _ = cv2.boundingRect(largest)
        max_width_ratio = float(cw) / float(mask.shape[1])

    # Aggregate score from color coverage and connected region size.
    score = (
        0.28 * color_ratio
        + 0.24 * top_ratio
        + 0.18 * bottom_ratio
        + 0.20 * center_ratio
        + 0.20 * max_area_ratio
    )
    helmet_found = (
        color_ratio >= 0.08
        and top_ratio >= 0.10
        and center_ratio >= 0.09
        and max_area_ratio >= 0.04
        and max_width_ratio >= 0.38
        and score >= 0.12
    )
    return helmet_found, score, roi_box


def run() -> int:
    window_name = "Eye State"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found or cannot be opened.")
        return 1
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    mp_face_mesh = _get_face_mesh_module()
    mp_hands = _get_hands_module()
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    calib_secs = 2.0
    calib_active = True
    calib_start = time.time()
    calib_samples = []
    threshold: Optional[float] = None
    helmet_score_smooth = 0.0

    status = "НЕТ ЛИЦА"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from camera.")
            break

        texts = []
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)
        hands_results = hands.process(rgb)
        multi_faces = results.multi_face_landmarks or []
        face_count = len(multi_faces)
        face_found = face_count > 0
        hand_points = []
        if hands_results.multi_hand_landmarks:
            hand_points = _hands_to_xy(hands_results.multi_hand_landmarks, w, h)

        ear_value: Optional[float] = None
        left_status = "НЕТ ЛИЦА"
        right_status = "НЕТ ЛИЦА"
        helmet_status = "НЕТ ЛИЦА"
        helmet_score = 0.0
        helmet_box: Optional[Tuple[int, int, int, int]] = None
        extra_person = False

        if face_found:
            face_pts_list = [
                _landmarks_to_xy(face_landmarks.landmark, w, h)
                for face_landmarks in multi_faces
            ]
            primary_idx = _select_primary_face(face_pts_list, w, h)
            pts = face_pts_list[primary_idx]
            extra_person = face_count > 1

            helmet_found, helmet_score, helmet_box = _detect_helmet(frame, pts)
            helmet_score_smooth = 0.72 * helmet_score_smooth + 0.28 * helmet_score
            helmet_confirmed = helmet_found or helmet_score_smooth >= 0.34
            helmet_status = "НАДЕТА" if helmet_confirmed else "НЕ ОБНАРУЖЕНА"
            if SHOW_HELMET_BOX and helmet_box is not None:
                x1, y1, x2, y2 = helmet_box
                box_color = (0, 200, 0) if helmet_confirmed else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            left_ear = _ear(pts, LEFT_EYE_IDX)
            right_ear = _ear(pts, RIGHT_EYE_IDX)
            ear_value = (left_ear + right_ear) / 2.0

            _draw_eye_points(frame, pts, LEFT_EYE_IDX)
            _draw_eye_points(frame, pts, RIGHT_EYE_IDX)

            if calib_active:
                calib_samples.append(ear_value)
                status = "КАЛИБРОВКА"
                left_status = "КАЛИБРОВКА"
                right_status = "КАЛИБРОВКА"
            elif threshold is not None:
                left_width = _dist(tuple(pts[LEFT_EYE_IDX[0]]), tuple(pts[LEFT_EYE_IDX[3]]))
                right_width = _dist(tuple(pts[RIGHT_EYE_IDX[0]]), tuple(pts[RIGHT_EYE_IDX[3]]))
                left_bbox = _eye_bbox(pts, LEFT_EYE_IDX)
                right_bbox = _eye_bbox(pts, RIGHT_EYE_IDX)
                left_center = _eye_center(pts, LEFT_EYE_IDX)
                right_center = _eye_center(pts, RIGHT_EYE_IDX)
                left_covered = _is_eye_covered(left_bbox, left_center, left_width, hand_points)
                right_covered = _is_eye_covered(right_bbox, right_center, right_width, hand_points)

                if left_covered:
                    left_status = "ЗАКРЫТ РУКОЙ"
                else:
                    left_status = "ОТКРЫТ" if left_ear >= threshold else "ЗАКРЫТ"

                if right_covered:
                    right_status = "ЗАКРЫТ РУКОЙ"
                else:
                    right_status = "ОТКРЫТ" if right_ear >= threshold else "ЗАКРЫТ"

                status = f"Л: {left_status}  П: {right_status}"
        else:
            status = "НЕТ ЛИЦА"

        if calib_active:
            elapsed = time.time() - calib_start
            if face_found:
                _queue_text(texts, f"Калибровка... {elapsed:.1f}с", 25, (0, 255, 255))
            else:
                _queue_text(texts, "Покажите лицо", 25, (0, 255, 255))

            if elapsed >= calib_secs:
                if calib_samples:
                    open_mean = float(np.mean(calib_samples))
                    threshold = open_mean * 0.75
                    calib_active = False
                else:
                    calib_start = time.time()
                    calib_samples = []

        if ear_value is not None:
            thr_text = f"{threshold:.3f}" if threshold is not None else "n/a"
            _queue_text(
                texts,
                f"EAR: {ear_value:.3f}  THR: {thr_text}",
                25 if not calib_active else 50,
                (255, 255, 255),
            )
        if face_found:
            _queue_text(texts, f"Левый: {left_status}", 75 if calib_active else 50, (0, 255, 255))
            _queue_text(texts, f"Правый: {right_status}", 100 if calib_active else 75, (0, 255, 255))
            _queue_text(
                texts,
                f"Каска: {helmet_status} ({helmet_score_smooth:.2f})",
                125 if calib_active else 100,
                (0, 255, 255),
            )
            if extra_person:
                _queue_text(
                    texts,
                    "Второй человек: ОБНАРУЖЕН",
                    150 if calib_active else 125,
                    (0, 180, 255),
                )
        else:
            _queue_text(texts, f"Статус: {status}", 75 if calib_active else 50, (0, 255, 255))

        _draw_texts(frame, texts)
        cv2.imshow(window_name, frame)

        # Allow close by window X button.
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        key = cv2.waitKey(1) & 0xFF
        # Q/q, Russian Й/й (common layout switch), and Esc.
        if key in (ord("q"), ord("Q"), 201, 233, 27):
            break
        if key in (ord("r"), ord("R")):
            calib_active = True
            calib_start = time.time()
            calib_samples = []
            threshold = None

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    hands.close()
    return 0


if __name__ == "__main__":
    sys.exit(run())
