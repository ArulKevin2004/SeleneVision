from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract_features


DATA_ROOT = Path('processed_dataset')
FEATURE_TYPE = 'hog'
USE_CLAHE = True
USE_BLUR = True
MODEL_PATH = Path('models/lunar_crater_svm.joblib')
CM_PATH = Path('outputs/confusion_matrix.png')
VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_split(split):
    X, y = [], []
    for folder, label in [('non_crater', 0), ('crater', 1)]:
        class_dir = DATA_ROOT / split / folder
        if not class_dir.exists():
            raise FileNotFoundError(
                f'Missing {class_dir}. Run prepare_dataset.py first.'
            )
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in VALID_EXT:
                continue
            X.append(extract_features(
                path,
                feature_type=FEATURE_TYPE,
                use_clahe=USE_CLAHE,
                use_blur=USE_BLUR,
            ))
            y.append(label)
    return np.asarray(X), np.asarray(y)


def main():
    print('Extracting train features...')
    X_train, y_train = load_split('train')
    print('Extracting validation features...')
    X_val, y_val = load_split('valid')
    print('Extracting test features...')
    X_test, y_test = load_split('test')

    print(f'Train: {len(y_train)} samples')
    print(f'Valid: {len(y_val)} samples')
    print(f'Test : {len(y_test)} samples')

    # Linear SVM keeps the project close to classical HOG+SVM literature.
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            class_weight='balanced',
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    print('\nVALIDATION RESULTS')
    val_pred = model.predict(X_val)
    print(f'Accuracy: {accuracy_score(y_val, val_pred):.4f}')
    print(classification_report(
        y_val, val_pred, target_names=['Non-crater', 'Crater'], digits=4
    ))

    print('\nFINAL TEST RESULTS')
    test_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, test_pred):.4f}')
    print(classification_report(
        y_test, test_pred, target_names=['Non-crater', 'Crater'], digits=4
    ))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    CM_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        'model': model,
        'feature_type': FEATURE_TYPE,
        'use_clahe': USE_CLAHE,
        'use_blur': USE_BLUR,
    }, MODEL_PATH)

    cm = confusion_matrix(y_test, test_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Non-crater', 'Crater'])
    disp.plot(values_format='d')
    plt.title('Lunar Crater Classification — HOG + Linear SVM')
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=200)
    plt.show()

    print(f'\nSaved model: {MODEL_PATH}')
    print(f'Saved confusion matrix: {CM_PATH}')


if __name__ == '__main__':
    main()
