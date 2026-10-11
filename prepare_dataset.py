from pathlib import Path
import argparse
import random
import shutil

import cv2
import numpy as np


IMAGE_SIZE = 128
MIN_CRATER_SIDE = 16       # ignore extremely tiny boxes
POSITIVE_PADDING = 0.15    # add context around crater
NEGATIVE_IOU_LIMIT = 0.02
RANDOM_SEED = 42
VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def yolo_to_box(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    _, xc, yc, w, h = map(float, parts)
    bw, bh = w * img_w, h * img_h
    cx, cy = xc * img_w, yc * img_h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return [x1, y1, x2, y2]


def clip_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return [
        max(0, min(img_w - 1, x1)),
        max(0, min(img_h - 1, y1)),
        max(1, min(img_w, x2)),
        max(1, min(img_h, y2)),
    ]


def square_with_padding(box, img_w, img_h, padding=POSITIVE_PADDING):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * padding)
    side = max(side, MIN_CRATER_SIDE)

    nx1 = cx - side / 2
    ny1 = cy - side / 2
    nx2 = cx + side / 2
    ny2 = cy + side / 2
    return clip_box([nx1, ny1, nx2, ny2], img_w, img_h)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def crop_and_resize(image, box):
    x1, y1, x2, y2 = map(int, box)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def matching_label_path(image_path, labels_dir):
    return labels_dir / (image_path.stem + '.txt')


def is_lunar_file(path):
    # Dataset mixes Moon and Mars. Mars files are explicitly named mars_crater--...
    return 'mars_crater' not in path.name.lower()


def prepare_split(source_root, output_root, split, rng):
    images_dir = source_root / split / 'images'
    labels_dir = source_root / split / 'labels'

    out_crater = output_root / split / 'crater'
    out_non = output_root / split / 'non_crater'
    out_crater.mkdir(parents=True, exist_ok=True)
    out_non.mkdir(parents=True, exist_ok=True)

    pos_count = 0
    neg_count = 0
    skipped_small = 0

    image_paths = [
        p for p in sorted(images_dir.iterdir())
        if p.suffix.lower() in VALID_EXT and is_lunar_file(p)
    ]

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        h, w = image.shape[:2]
        label_path = matching_label_path(image_path, labels_dir)
        if not label_path.exists():
            continue

        raw_boxes = []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            box = yolo_to_box(line, w, h)
            if box is not None:
                raw_boxes.append(clip_box(box, w, h))

        valid_boxes = []
        for box in raw_boxes:
            x1, y1, x2, y2 = box
            if min(x2 - x1, y2 - y1) < MIN_CRATER_SIDE:
                skipped_small += 1
                continue
            valid_boxes.append(box)

        # Positive crater crops
        positive_sizes = []
        for i, box in enumerate(valid_boxes):
            padded = square_with_padding(box, w, h)
            crop = crop_and_resize(image, padded)
            if crop is None:
                continue
            cv2.imwrite(str(out_crater / f'{image_path.stem}_crater_{i:03d}.png'), crop)
            pos_count += 1
            positive_sizes.append(max(box[2]-box[0], box[3]-box[1]))

        # Balanced negative crops from terrain not overlapping any annotated crater.
        # We sample roughly one negative for every positive from the SAME original image,
        # but save it within the same train/valid/test split to avoid leakage.
        target_neg = len(positive_sizes)
        created = 0
        attempts = 0
        max_attempts = max(300, target_neg * 120)

        while created < target_neg and attempts < max_attempts:
            attempts += 1
            if positive_sizes:
                side = int(rng.choice(positive_sizes) * 1.30)
            else:
                side = 96
            side = max(32, min(side, min(w, h, 192)))

            if w - side <= 1 or h - side <= 1:
                break

            x1 = rng.randint(0, w - side)
            y1 = rng.randint(0, h - side)
            candidate = [x1, y1, x1 + side, y1 + side]

            if any(iou(candidate, box) > NEGATIVE_IOU_LIMIT for box in raw_boxes):
                continue

            crop = crop_and_resize(image, candidate)
            if crop is None:
                continue

            cv2.imwrite(
                str(out_non / f'{image_path.stem}_non_{created:03d}.png'), crop
            )
            created += 1
            neg_count += 1

    return len(image_paths), pos_count, neg_count, skipped_small


def main():
    parser = argparse.ArgumentParser(
        description='Convert the uploaded YOLO Moon/Mars dataset into lunar crater/non-crater patches.'
    )
    parser.add_argument(
        '--source', default='craters',
        help='Path to the extracted craters/ folder containing train/valid/test.'
    )
    parser.add_argument(
        '--output', default='processed_dataset',
        help='Output folder.'
    )
    args = parser.parse_args()

    source_root = Path(args.source)
    output_root = Path(args.output)

    if output_root.exists():
        shutil.rmtree(output_root)

    rng = random.Random(RANDOM_SEED)

    print('Preparing lunar-only dataset...')
    print('Mars images are ignored automatically.\n')

    total_pos = total_neg = 0
    for split in ['train', 'valid', 'test']:
        n_images, pos, neg, small = prepare_split(
            source_root, output_root, split, rng
        )
        total_pos += pos
        total_neg += neg
        print(
            f'{split:5s}: lunar images={n_images:2d}, '
            f'crater patches={pos:3d}, non-crater patches={neg:3d}, '
            f'skipped tiny boxes={small:2d}'
        )

    print('\nDone.')
    print(f'Total crater patches:     {total_pos}')
    print(f'Total non-crater patches: {total_neg}')
    print(f'Output: {output_root.resolve()}')


if __name__ == '__main__':
    main()
