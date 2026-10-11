from pathlib import Path
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract_features


DATA_ROOT = Path('processed_dataset')
VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

EXPERIMENTS = [
    ('HOG only', 'hog', False, False),
    ('CLAHE + HOG', 'hog', True, False),
    ('CLAHE + Blur + HOG', 'hog', True, True),
    ('CLAHE + Blur + HOG+LBP', 'hog_lbp', True, True),
]


def load_split(split, feature_type, clahe, blur):
    X, y = [], []
    for folder, label in [('non_crater', 0), ('crater', 1)]:
        for path in sorted((DATA_ROOT / split / folder).iterdir()):
            if path.suffix.lower() not in VALID_EXT:
                continue
            X.append(extract_features(path, feature_type, clahe, blur))
            y.append(label)
    return np.asarray(X), np.asarray(y)


def evaluate(name, feature_type, clahe, blur):
    X_train, y_train = load_split('train', feature_type, clahe, blur)
    X_test, y_test = load_split('test', feature_type, clahe, blur)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=1.0, class_weight='balanced')),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average='binary', zero_division=0
    )
    return acc, precision, recall, f1


def main():
    print(f"{'Experiment':35s} {'Acc':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s}")
    print('-' * 72)
    for exp in EXPERIMENTS:
        metrics = evaluate(*exp)
        print(f'{exp[0]:35s} ' + ' '.join(f'{m:8.4f}' for m in metrics))


if __name__ == '__main__':
    main()
