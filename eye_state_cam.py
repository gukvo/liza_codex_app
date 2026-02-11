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


def run() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found or cannot be opened.")
        return 1

    mp_face_mesh = _get_face_mesh_module()
    mp_hands = _get_hands_module()
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
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
        face_found = results.multi_face_landmarks is not None
        hand_points = []
        if hands_results.multi_hand_landmarks:
            hand_points = _hands_to_xy(hands_results.multi_hand_landmarks, w, h)

        ear_value: Optional[float] = None
        left_status = "НЕТ ЛИЦА"
        right_status = "НЕТ ЛИЦА"

        if face_found:
            landmarks = results.multi_face_landmarks[0].landmark
            pts = _landmarks_to_xy(landmarks, w, h)

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
        else:
            _queue_text(texts, f"Статус: {status}", 75 if calib_active else 50, (0, 255, 255))

        _draw_texts(frame, texts)

        cv2.imshow("Eye State", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
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
