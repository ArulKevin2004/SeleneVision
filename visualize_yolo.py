from pathlib import Path
import argparse
import cv2


def main(image_path, label_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError('Could not read image')
    h, w = image.shape[:2]

    for line in Path(label_path).read_text().splitlines():
        if not line.strip():
            continue
        _, xc, yc, bw, bh = map(float, line.split())
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    Path('outputs').mkdir(exist_ok=True)
    out = Path('outputs/annotated_sample.jpg')
    cv2.imwrite(str(out), image)
    print(f'Saved: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('label')
    args = parser.parse_args()
    main(Path(args.image), Path(args.label))
