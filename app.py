from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import (
    extract_features,
    extract_hog,
    extract_lbp,
    preprocess_array,
)

DATA_ROOT = Path("processed_dataset")
SOURCE_ROOT = Path("craters")
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

METHODS = [
    ("HOG only", "hog", False, False),
    ("CLAHE + HOG", "hog", True, False),
    ("CLAHE + Blur + HOG", "hog", True, True),
    ("CLAHE + Blur + HOG + LBP", "hog_lbp", True, True),
]


@st.cache_resource
def train_method_models():
    models = {}
    for name, feature_type, use_clahe, use_blur in METHODS:
        X_train, y_train = load_split("train", feature_type, use_clahe, use_blur)
        X_test, y_test = load_split("test", feature_type, use_clahe, use_blur)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="linear",
                        C=1.0,
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = float(np.mean(pred == y_test))
        models[name] = {
            "model": model,
            "feature_type": feature_type,
            "use_clahe": use_clahe,
            "use_blur": use_blur,
            "test_accuracy": acc,
        }
    return models


def extract_features_array(image_array, feature_type: str, use_clahe: bool, use_blur: bool):
    image = preprocess_array(image_array, use_clahe=use_clahe, use_blur=use_blur)
    if feature_type == "hog":
        return extract_hog(image)
    if feature_type == "lbp":
        return extract_lbp(image)
    if feature_type == "hog_lbp":
        return np.concatenate([extract_hog(image), extract_lbp(image)])
    raise ValueError("Unsupported feature type")


def load_split(split, feature_type, use_clahe, use_blur):
    X, y = [], []
    for label_name, label in [("non_crater", 0), ("crater", 1)]:
        class_dir = DATA_ROOT / split / label_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing {class_dir}. Run prepare_dataset.py first.")

        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in VALID_EXT:
                continue
            X.append(extract_features(path, feature_type, use_clahe, use_blur))
            y.append(label)
    return np.asarray(X), np.asarray(y)


def list_sample_paths():
    sample_paths = []
    for folder in ["crater", "non_crater"]:
        class_dir = DATA_ROOT / "test" / folder
        if class_dir.exists():
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in VALID_EXT:
                    sample_paths.append(path)
    return sample_paths


def find_original_source_for_patch(patch_path: Path):
    patch_name = patch_path.name
    stem = patch_name
    for token in ("_crater_", "_non_"):
        if token in stem:
            stem = stem.split(token)[0]
            break

    candidates = list((SOURCE_ROOT / "test" / "images").glob(f"{stem}.*"))
    if candidates:
        return candidates[0]

    for image_path in sorted((SOURCE_ROOT / "test" / "images").iterdir()):
        if image_path.suffix.lower() in VALID_EXT and image_path.stem.startswith(stem):
            return image_path

    return None


def find_label_for_source(source_path: Path):
    label_path = SOURCE_ROOT / "test" / "labels" / f"{source_path.stem}.txt"
    if label_path.exists():
        return label_path
    return None


def draw_roi_over_image(image_bgr, label_path: Path):
    image = image_bgr.copy()
    if not label_path or not label_path.exists():
        return image

    h, w = image.shape[:2]
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, xc, yc, bw, bh = map(float, parts)
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return image


def classify_patch(image_path: Path, feature_type: str, use_clahe: bool, use_blur: bool, model):
    features = np.asarray(extract_features(image_path, feature_type, use_clahe, use_blur)).reshape(1, -1)
    pred = int(model.predict(features)[0])
    probs = model.predict_proba(features)[0]
    conf = float(probs[pred])
    label = "Crater" if pred == 1 else "Non-crater"
    return label, conf


def classify_array(image_array, feature_type: str, use_clahe: bool, use_blur: bool, model):
    features = np.asarray(extract_features_array(image_array, feature_type, use_clahe, use_blur)).reshape(1, -1)
    pred = int(model.predict(features)[0])
    probs = model.predict_proba(features)[0]
    conf = float(probs[pred])
    label = "Crater" if pred == 1 else "Non-crater"
    return label, conf


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(detections, iou_threshold=0.3):
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        detections = [
            d for d in detections
            if iou(
                [current["x"], current["y"], current["x"] + current["w"], current["y"] + current["h"]],
                [d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]],
            )
            < iou_threshold
        ]
    return keep


def detect_craters_in_image(
    image_bgr,
    model_bundle,
    patch_sizes=None,
    stride=64,
    confidence_threshold=0.60,
    iou_threshold=0.30,
):
    if patch_sizes is None:
        patch_sizes = [128]
    if isinstance(patch_sizes, int):
        patch_sizes = [patch_sizes]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    detections = []

    for patch_size in patch_sizes:
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch = gray[y:y + patch_size, x:x + patch_size]
                if patch.shape[:2] != (patch_size, patch_size):
                    patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

                label, conf = classify_array(
                    patch,
                    model_bundle["feature_type"],
                    model_bundle["use_clahe"],
                    model_bundle["use_blur"],
                    model_bundle["model"],
                )

                if label == "Crater" and conf >= confidence_threshold:
                    detections.append({
                        "x": x,
                        "y": y,
                        "w": patch_size,
                        "h": patch_size,
                        "confidence": conf,
                    })

    return non_max_suppression(detections, iou_threshold=iou_threshold)


def build_heatmap(image_bgr, detections, alpha=0.45):
    image = image_bgr.copy()
    if not detections:
        return image

    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for det in detections:
        x0 = max(0, int(det["x"]))
        y0 = max(0, int(det["y"]))
        x1 = min(w, int(det["x"]) + int(det["w"]))
        y1 = min(h, int(det["y"]) + int(det["h"]))
        heat[y0:y1, x0:x1] += det["confidence"] * 255.0

    heat = np.clip(heat, 0, 255)
    heat_img = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, heat_img, alpha, 0)


def draw_detection_boxes(image_bgr, detections):
    image = image_bgr.copy()
    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            image,
            f"{det['confidence'] * 100:.1f}%",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    return image


def preprocess_for_full_image_pipeline(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return gray, blurred, edges


def draw_hough_candidates(image_bgr, circles):
    image = image_bgr.copy()
    if circles is None:
        return image

    for cx, cy, r in circles[0]:
        cv2.circle(image, (int(cx), int(cy)), int(r), (0, 255, 255), 2)
        cv2.circle(image, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    return image


def detect_candidate_circles(image_bgr, min_radius=12, max_radius=110, min_dist=30, dp=1.2, threshold=18):
    _, blurred, edges = preprocess_for_full_image_pipeline(image_bgr)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=150,
        param2=threshold,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    return {
        "gray": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
        "blurred": blurred,
        "edges": edges,
        "circles": circles,
    }


def verify_hough_candidates(image_bgr, model_bundle, circles, confidence_threshold=0.60):
    if circles is None:
        return []

    verified = []
    for cx, cy, r in circles[0]:
        x0 = max(0, int(cx - r * 1.3))
        y0 = max(0, int(cy - r * 1.3))
        x1 = min(image_bgr.shape[1], int(cx + r * 1.3))
        y1 = min(image_bgr.shape[0], int(cy + r * 1.3))
        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        label, conf = classify_array(
            crop,
            model_bundle["feature_type"],
            model_bundle["use_clahe"],
            model_bundle["use_blur"],
            model_bundle["model"],
        )

        if label == "Crater" and conf >= confidence_threshold:
            verified.append({
                "x": x0,
                "y": y0,
                "w": x1 - x0,
                "h": y1 - y0,
                "radius": float(r),
                "diameter": float(r * 2),
                "size_px": float(np.pi * r * r),
                "confidence": float(conf),
            })

    return non_max_suppression(verified, iou_threshold=0.25)


def compute_hog_visual(image_array, use_clahe=True, use_blur=True):
    image = preprocess_array(image_array, use_clahe=use_clahe, use_blur=use_blur)
    _, hog_visual = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=False,
    )

    hog_visual = np.asarray(hog_visual, dtype=np.float32)
    hog_visual = (hog_visual - hog_visual.min()) / (hog_visual.max() - hog_visual.min() + 1e-8)
    hog_visual = (hog_visual * 255.0).astype(np.uint8)
    return image, hog_visual


def render_method_card(method_name: str, feature_type: str, use_clahe: bool, use_blur: bool, selected_patch: Path, model_bundle):
    proc = preprocess_array(cv2.imread(str(selected_patch), cv2.IMREAD_GRAYSCALE), use_clahe, use_blur)
    label, conf = classify_patch(selected_patch, feature_type, use_clahe, use_blur, model_bundle["model"])

    st.markdown(f"### {method_name}")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image(proc, channels="GRAY", caption="After preprocessing", width=300)
        if "hog" in feature_type:
            _, hog_visual = compute_hog_visual(cv2.imread(str(selected_patch), cv2.IMREAD_GRAYSCALE), use_clahe, use_blur)
            st.image(hog_visual, caption="HOG gradient map / orientation summary", channels="GRAY", width=300)
    with col2:
        st.metric("Prediction", label)
        st.metric("Confidence", f"{conf * 100:.2f}%")
        st.caption(f"Preprocessing: CLAHE={use_clahe}, Blur={use_blur}")
        st.caption(f"Model accuracy on test set: {model_bundle['test_accuracy'] * 100:.2f}%")


st.set_page_config(page_title="Lunar Crater Analyzer", layout="wide")
st.title("Lunar Crater Classification Dashboard")

if not DATA_ROOT.exists() or not (DATA_ROOT / "test").exists():
    st.error("The processed_dataset/test folder was not found. Run prepare_dataset.py first.")
    st.stop()

sample_paths = list_sample_paths()
if not sample_paths:
    st.error("No processed patch samples were found. Please generate the dataset first.")
    st.stop()

models = train_method_models()

st.sidebar.header("Controls")
use_existing_sample = st.sidebar.radio("Input source", ["Sample patch", "Upload image"], index=0)

if use_existing_sample == "Sample patch":
    selected_patch = st.sidebar.selectbox("Pick a sample patch to inspect", sample_paths, format_func=lambda p: p.name)
    selected_patch = Path(selected_patch)
    source_image = find_original_source_for_patch(selected_patch)
    source_label = find_label_for_source(source_image) if source_image else None

    st.subheader("Patch preview")
    orig_patch = cv2.imread(str(selected_patch), cv2.IMREAD_COLOR)
    if orig_patch is None:
        st.error(f"Could not read {selected_patch}")
        st.stop()

    left, right = st.columns(2)
    with left:
        st.image(orig_patch, channels="BGR", caption="Original patch", use_container_width=True)
    with right:
        if source_image is not None:
            src_img = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if src_img is not None:
                roi_img = draw_roi_over_image(src_img, source_label)
                st.image(roi_img, channels="BGR", caption=f"Source image with crater ROI: {source_image.name}", use_container_width=True)
            else:
                st.info("Source image could not be loaded")
        else:
            st.info("No matching source image found for this patch, so ROI overlay is unavailable.")

    st.subheader("Method comparison")
    for method_name, feature_type, use_clahe, use_blur in METHODS:
        render_method_card(method_name, feature_type, use_clahe, use_blur, selected_patch, models[method_name])

    st.subheader("Final class decision")
    final_predictions = []
    for method_name, feature_type, use_clahe, use_blur in METHODS:
        label, conf = classify_patch(selected_patch, feature_type, use_clahe, use_blur, models[method_name]["model"])
        final_predictions.append((method_name, label, conf))

    for method_name, label, conf in final_predictions:
        st.write(f"- {method_name}: {label} ({conf * 100:.2f}% confidence)")

    if source_image is not None and source_label is not None:
        st.subheader("ROI status")
        if any(label == "Crater" for _, label, _ in final_predictions):
            st.success("At least one model classified the patch as a crater, and the ROI is highlighted on the source image above.")
        else:
            st.info("No method classified this sample as a crater, so no crater ROI is highlighted.")

else:
    uploaded_file = st.sidebar.file_uploader("Upload lunar image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if uploaded_image is None:
            st.error("The uploaded file could not be read as an image.")
            st.stop()

        confidence_threshold = st.sidebar.slider("Confidence threshold", 0.40, 0.95, 0.60, 0.01)
        nms_threshold = st.sidebar.slider("NMS IoU threshold", 0.10, 0.80, 0.30, 0.05)
        scales = st.sidebar.multiselect(
            "Detection scales",
            [64, 128, 192, 256],
            default=[128, 192],
        )
        if not scales:
            scales = [128]

        st.subheader("Uploaded image overview")
        original_col, heatmap_col = st.columns(2)
        with original_col:
            st.image(uploaded_image, channels="BGR", caption="Original uploaded image", use_container_width=True)

        gradient_method_name = st.sidebar.selectbox("Gradient map method", [m[0] for m in METHODS if "hog" in m[1]], index=0)
        grad_bundle = models[gradient_method_name]
        grad_gray = preprocess_array(cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY), grad_bundle["use_clahe"], grad_bundle["use_blur"])
        _, grad_map = hog(
            grad_gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True,
            feature_vector=False,
        )
        grad_map = np.asarray(grad_map, dtype=np.float32)
        grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-8)
        grad_map = (grad_map * 255.0).astype(np.uint8)

        with heatmap_col:
            st.image(grad_map, channels="GRAY", caption=f"Full-image HOG gradient map ({gradient_method_name})", use_container_width=True)

        st.subheader("Full-image crater detection pipeline")
        pipeline_params = {
            "min_radius": st.sidebar.slider("Hough min radius", 8, 40, 12),
            "max_radius": st.sidebar.slider("Hough max radius", 40, 200, 110),
            "min_dist": st.sidebar.slider("Hough min distance", 10, 120, 30),
            "dp": st.sidebar.slider("Hough dp", 1.0, 2.0, 1.2, 0.1),
            "threshold": st.sidebar.slider("Hough accumulator threshold", 5, 40, 18),
        }
        pipeline_bundle = detect_candidate_circles(
            uploaded_image,
            min_radius=pipeline_params["min_radius"],
            max_radius=pipeline_params["max_radius"],
            min_dist=pipeline_params["min_dist"],
            dp=pipeline_params["dp"],
            threshold=pipeline_params["threshold"],
        )
        stage_cols = st.columns(4)
        with stage_cols[0]:
            st.image(pipeline_bundle["gray"], channels="GRAY", caption="1. CLAHE", use_container_width=True)
        with stage_cols[1]:
            st.image(pipeline_bundle["blurred"], channels="GRAY", caption="2. Gaussian blur", use_container_width=True)
        with stage_cols[2]:
            st.image(pipeline_bundle["edges"], channels="GRAY", caption="3. Canny edges", use_container_width=True)
        with stage_cols[3]:
            st.image(draw_hough_candidates(uploaded_image, pipeline_bundle["circles"]), channels="BGR", caption="4. Hough circle candidates", use_container_width=True)

        best_method_name = max(
            [method for method in METHODS if "hog" in method[1]],
            key=lambda item: models[item[0]]["test_accuracy"],
        )[0]
        verified_detections = verify_hough_candidates(
            uploaded_image,
            models[best_method_name],
            pipeline_bundle["circles"],
            confidence_threshold=confidence_threshold,
        )

        st.subheader("Verified crater candidates")
        if verified_detections:
            verified_overlay = draw_detection_boxes(uploaded_image, verified_detections)
            st.image(verified_overlay, channels="BGR", caption=f"Final SVM verification using {best_method_name}", use_container_width=True)
            st.table([
                {
                    "ROI": f"#{idx + 1}",
                    "Center": f"({int(det['x'] + det['w'] / 2)}, {int(det['y'] + det['h'] / 2)})",
                    "Radius (px)": round(det['radius'], 1),
                    "Diameter (px)": round(det['diameter'], 1),
                    "Area (px²)": round(det['size_px'], 1),
                    "Confidence": f"{det['confidence'] * 100:.2f}%",
                }
                for idx, det in enumerate(verified_detections)
            ])
        else:
            st.info("No crater candidate passed the Hough + SVM verification stage for the current settings.")

        method_results = []
        for method_name, feature_type, use_clahe, use_blur in METHODS:
            model_bundle = models[method_name]
            detections = detect_craters_in_image(
                uploaded_image,
                model_bundle,
                patch_sizes=scales,
                stride=max(32, min(96, max(scales) // 2)),
                confidence_threshold=confidence_threshold,
                iou_threshold=nms_threshold,
            )
            method_results.append((method_name, detections, model_bundle))

        best_method_name, best_detections, _ = max(method_results, key=lambda item: (len(item[1]), np.mean([d["confidence"] for d in item[1]]) if item[1] else 0.0))
        st.success(f"Best method on this image: {best_method_name} with {len(best_detections)} crater ROI(s).")

        with heatmap_col:
            heatmap_image = build_heatmap(uploaded_image, best_detections)
            st.image(heatmap_image, channels="BGR", caption="Crater confidence heatmap", use_container_width=True)

        st.subheader("Method comparison")
        cols = st.columns(2)
        for idx, (method_name, detections, model_bundle) in enumerate(method_results):
            with cols[idx % 2]:
                annotated = draw_detection_boxes(uploaded_image, detections)
                st.image(annotated, channels="BGR", caption=f"{method_name}: {len(detections)} ROI(s)", use_container_width=True)
                st.caption(f"Accuracy: {model_bundle['test_accuracy'] * 100:.2f}% | Confidence threshold: {confidence_threshold:.2f}")

        st.subheader("Selected region preview")
        if best_detections:
            det = best_detections[0]
            crop = uploaded_image[det["y"]:det["y"] + det["h"], det["x"]:det["x"] + det["w"]]
            proc = preprocess_array(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), models[best_method_name]["use_clahe"], models[best_method_name]["use_blur"])
            st.image(proc, channels="GRAY", caption="Preprocessed ROI crop", use_container_width=True)
            label, conf = classify_array(crop, models[best_method_name]["feature_type"], models[best_method_name]["use_clahe"], models[best_method_name]["use_blur"], models[best_method_name]["model"])
            st.metric("Final classification", label)
            st.metric("Confidence", f"{conf * 100:.2f}%")
        else:
            st.info("No crater passed the confidence threshold for the selected settings.")
    else:
        st.info("Upload a lunar image to scan it for crater regions using the trained SVM model.")
