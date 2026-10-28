"""
CivicPulse Master ML Training Orchestrator (train_all.py)
========================================================
Runs the end-to-end training pipeline:
1. Ingests smart_city_civic_complaints.csv & smart_city_civic_dataset.xlsx
2. Trains Department Model (TF-IDF + Linear SVM)
3. Trains Category Model (TF-IDF + Linear SVM)
4. Trains Priority Model (Random Forest Classifier)
5. Trains Duplicate Model (Random Forest Classifier)
6. Exports all joblib artifacts to `ml/models/`
"""

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from generate_dataset import load_and_augment_dataset
from train_department import train_department_model
from train_category import train_category_model
from train_priority import train_priority_model
from train_duplicate import train_duplicate_model


def run_full_training_pipeline(
    data_path: str = "ml/data/complaints.csv",
    models_dir: str = "ml/models",
    force_rebuild_data: bool = True
):
    print("=" * 70)
    print("       CIVICPULSE MASTER AI/ML TRAINING PIPELINE")
    print("=" * 70)
    start_time = time.time()

    # 1. Dataset Generation
    if force_rebuild_data or not os.path.exists(data_path):
        print("\n[STEP 1/5] Ingesting & augmenting dataset from ground truth smart city files...")
        load_and_augment_dataset(
            csv_path="smart_city_civic_complaints.csv",
            xlsx_path="smart_city_civic_dataset.xlsx",
            output_path=data_path
        )
    else:
        print(f"\n[STEP 1/5] Found existing dataset at '{data_path}'.")

    # 2. Department Model
    print("\n[STEP 2/5] Training Department Classifier (TF-IDF + Linear SVM)...")
    _, dept_acc, dept_f1 = train_department_model(data_path, models_dir, evaluate=False)
    print(f"--> Department Model Accuracy: {dept_acc*100:.2f}% | F1: {dept_f1:.4f}")

    # 3. Category Model
    print("\n[STEP 3/5] Training Category Classifier (TF-IDF + Linear SVM)...")
    _, cat_acc, cat_f1 = train_category_model(data_path, models_dir, evaluate=False)
    print(f"--> Category Model Accuracy: {cat_acc*100:.2f}% | F1: {cat_f1:.4f}")

    # 4. Priority Model
    print("\n[STEP 4/5] Training Priority Classifier (Random Forest Classifier)...")
    _, prio_acc, prio_f1 = train_priority_model(data_path, models_dir, evaluate=False)
    print(f"--> Priority Model Accuracy: {prio_acc*100:.2f}% | F1: {prio_f1:.4f}")

    # 5. Duplicate Model
    print("\n[STEP 5/5] Training Duplicate Detection Classifier (Random Forest)...")
    _, dup_acc, dup_f1 = train_duplicate_model(data_path, models_dir, evaluate=False)
    print(f"--> Duplicate Model Accuracy: {dup_acc*100:.2f}% | F1: {dup_f1:.4f}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"   TRAINING COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("=" * 70)
    print("Exported Model Artifacts:")
    for fname in os.listdir(models_dir):
        fpath = os.path.join(models_dir, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024.0
            print(f"  • {fname:30} ({size_kb:.1f} KB)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_full_training_pipeline()
