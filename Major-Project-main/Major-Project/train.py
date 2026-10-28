"""
CivicPulse ML Model Training & Evaluation Pipeline
==================================================
Trains multi-target NLP models for municipal Department and Priority classification,
evaluates precision, recall, and F1 metrics, builds duplicate detection embeddings,
and exports the serialized bundle as `model.pkl`.
"""

import os
import sys
import random
import logging
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score

from model import (
    CivicDepartmentModel,
    CivicPriorityModel,
    CivicDuplicateDetector,
    CivicMLBundle,
    save_model,
    DEPARTMENTS,
    PRIORITIES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("civicpulse.ml.train")

WARDS = [
    "Indiranagar", "Koramangala", "Whitefield", "Jayanagar", "Malleshwaram",
    "HSR Layout", "Yelahanka", "BTM Layout", "Rajajinagar", "Hebbal",
    "Electronic City", "Bellandur", "Marathahalli", "MG Road", "Basavanagudi"
]

# Comprehensive Seed Corpus of Real-World Civic Issues
RAW_COMPLAINT_DATA = [
    # --- Water & Sewage ---
    ("Major drinking water pipeline burst flooding 100 Feet Road, gallons of water getting wasted", "Water & Sewage", "HIGH"),
    ("Sewage manhole overflowing on residential cross road emitting unbearable foul smell and health hazard", "Water & Sewage", "HIGH"),
    ("Brown muddy contaminated water coming out of municipal tap in apartment complex", "Water & Sewage", "CRITICAL"),
    ("Underground drainage pipeline blocked causing dirty sewage backup into ground floor houses", "Water & Sewage", "CRITICAL"),
    ("Slow continuous water leak from public pipe joint near community park entrance", "Water & Sewage", "LOW"),
    ("Water supply valve broken on main street, water spraying continuously onto road", "Water & Sewage", "HIGH"),
    ("Stormwater drain clogged with silt and plastic before monsoon season", "Water & Sewage", "MEDIUM"),
    ("Open manhole on sidewalk posing grave danger of pedestrians falling in during dark", "Water & Sewage", "CRITICAL"),
    ("Low water pressure in municipal pipeline for the past 4 days in block B", "Water & Sewage", "LOW"),
    ("Gutter wall collapsed into street causing drain water to spill onto footpath", "Water & Sewage", "MEDIUM"),
    ("Public drinking water tap leaking non-stop without tap faucet", "Water & Sewage", "LOW"),
    ("Sewage pipeline mixed with drinking water line causing illness in neighborhood", "Water & Sewage", "CRITICAL"),
    ("Waterlogging in underpass after moderate rain due to choked drainage pumps", "Water & Sewage", "HIGH"),
    ("Commercial tanker broken discharge valve dumping dirty waste water on public road", "Water & Sewage", "MEDIUM"),
    ("Borewell motor damaged, no municipal water supply to entire sector 4", "Water & Sewage", "HIGH"),

    # --- Waste Management ---
    ("Huge garbage dump overflowing on roadside attracting stray animals and creating foul odor", "Waste Management", "MEDIUM"),
    ("Daily garbage collection vehicle has not visited 4th block for over a week", "Waste Management", "MEDIUM"),
    ("Illegal dumping of construction debris and plastic sacks on empty vacant plot", "Waste Management", "LOW"),
    ("Burning of toxic plastic waste and dried leaves in open neighborhood causing suffocating smoke", "Waste Management", "HIGH"),
    ("Broken public dustbin overturned with rotten food waste scattered all over pedestrian walkway", "Waste Management", "MEDIUM"),
    ("Dead animal carcass lying uncleaned near school gate for 2 days", "Waste Management", "HIGH"),
    ("Biomedical and medical clinic waste dumped illegally near public lake bund", "Waste Management", "CRITICAL"),
    ("Commercial restaurant dumping greasy food waste into open street corner", "Waste Management", "HIGH"),
    ("Secondary waste transfer bin full and spilling waste onto main road blocking traffic lane", "Waste Management", "HIGH"),
    ("Garbage truck spilling wet leachate liquid all over the road while driving", "Waste Management", "LOW"),
    ("Market area overflowing with vegetable and fruit refuse after weekly bazaar", "Waste Management", "MEDIUM"),
    ("Litter and empty plastic bottles strewn around bus stop shelter", "Waste Management", "LOW"),
    ("Sanitation sweepers dumping collected road dust into open stormwater drain", "Waste Management", "MEDIUM"),
    ("Hazardous chemical containers dumped openly near residential layout boundary", "Waste Management", "CRITICAL"),

    # --- Traffic ---
    ("Traffic signal at major 4-way junction stuck on flashing yellow causing severe vehicle gridlock", "Traffic", "HIGH"),
    ("Illegal commercial parking on both sides of narrow lane completely blocking ambulance access", "Traffic", "HIGH"),
    ("Missing zebra crossing and pedestrian signals near hospital making crossing impossible", "Traffic", "MEDIUM"),
    ("Unscientific speed breaker without reflective paint causing vehicle underbody crashes", "Traffic", "MEDIUM"),
    ("Extreme traffic bottleneck caused by delivery trucks unloading in peak rush hour", "Traffic", "MEDIUM"),
    ("Blind turn corner obstructed by unauthorized parked trucks near school", "Traffic", "HIGH"),
    ("Traffic light timing malfunctioning: green signal only lasting 5 seconds on main arterial road", "Traffic", "HIGH"),
    ("Vehicles regularly driving on wrong side of one-way road due to missing no-entry sign", "Traffic", "HIGH"),
    ("Vehicular collision blocking 2 lanes of express flyover causing 3 km tailback", "Traffic", "CRITICAL"),
    ("Footpath encroached by two-wheeler parking forcing school children to walk on busy road", "Traffic", "MEDIUM"),
    ("Pedestrian subway locked and closed forcing people to jaywalk across 6-lane road", "Traffic", "HIGH"),
    ("Faded road lane markings and broken cat-eyes near accident-prone sharp curve", "Traffic", "LOW"),
    ("Traffic queue sensor broken at tech park exit gate causing massive delay", "Traffic", "LOW"),

    # --- Police / Public Safety ---
    ("Group of unruly individuals harassing pedestrians and eve-teasing after dark near metro station", "Police / Public Safety", "HIGH"),
    ("Suspicious armed brawl outside local bar late night with broken bottles and threats", "Police / Public Safety", "CRITICAL"),
    ("Complete darkness on pedestrian walkway leading to multiple chain snatching attempts", "Police / Public Safety", "HIGH"),
    ("Vandals broke public surveillance CCTV camera and damaged street benches", "Police / Public Safety", "MEDIUM"),
    ("Robbery attempt reported in alleyway with poor surveillance coverage", "Police / Public Safety", "HIGH"),
    ("Unattended locked briefcase left near crowded bus terminal for several hours", "Police / Public Safety", "CRITICAL"),
    ("Reckless midnight racing by modified cars endangering pedestrians on ring road", "Police / Public Safety", "HIGH"),
    ("Unauthorized extortion demands by local hooligans from street vendors", "Police / Public Safety", "MEDIUM"),
    ("Stalking incident reported near ladies hostel with lack of police patrolling", "Police / Public Safety", "CRITICAL"),
    ("Defaced public monument with offensive graffiti", "Police / Public Safety", "LOW"),
    ("Trespassing and illicit activities in abandoned government building near playground", "Police / Public Safety", "MEDIUM"),

    # --- Electricity ---
    ("Live electric wire snapped and hanging dangerously low over flooded water puddle", "Electricity", "CRITICAL"),
    ("Distribution transformer sparking vigorously with loud buzzing noise and smoke coming out", "Electricity", "CRITICAL"),
    ("Complete street light outage across entire 5th main road making area pitch dark", "Electricity", "MEDIUM"),
    ("High voltage power surge damaged household appliances across 30 homes in sector 2", "Electricity", "HIGH"),
    ("Electric junction box cover open with exposed high-voltage busbars near children playground", "Electricity", "CRITICAL"),
    ("Continuous power tripping every 15 minutes due to overloaded phase transformer", "Electricity", "HIGH"),
    ("Streetlight pole leaning at dangerous angle after getting hit by vehicle, about to fall", "Electricity", "HIGH"),
    ("Underground electric cable burst causing localized blackout in commercial market", "Electricity", "HIGH"),
    ("Streetlight on continuously during bright daylight wasting municipal power", "Electricity", "LOW"),
    ("Tangled mess of unauthorized internet and power cables dangling into bus window height", "Electricity", "MEDIUM"),
    ("Flickering street lamp pole giving minor electric shock when touched during rain", "Electricity", "CRITICAL"),
    ("Broken meter box on public utility post with dangling wires", "Electricity", "MEDIUM"),

    # --- Roads & Infrastructure ---
    ("Massive 3-foot deep pothole on main road causing two-wheeler rider falls and severe damage", "Roads & Infrastructure", "HIGH"),
    ("Road surface caved in creating dangerous sinkhole after water pipe burst underneath", "Roads & Infrastructure", "CRITICAL"),
    ("Broken concrete slabs on pedestrian sidewalk with exposed rusty iron rods", "Roads & Infrastructure", "HIGH"),
    ("Damaged concrete road divider missing warning reflectors causing night accidents", "Roads & Infrastructure", "HIGH"),
    ("Flyover expansion joint loose and rattling dangerously as heavy vehicles pass over", "Roads & Infrastructure", "CRITICAL"),
    ("Unpaved muddy gravel road with severe ruts making residential access impossible in rains", "Roads & Infrastructure", "MEDIUM"),
    ("Open utility trench dug for pipeline left un-barricaded and without warning lights", "Roads & Infrastructure", "HIGH"),
    ("Footpath curb stones dislodged and scattered across road shoulder", "Roads & Infrastructure", "LOW"),
    ("Newly laid asphalt peeling off within 2 weeks of road laying due to poor bituminous mix", "Roads & Infrastructure", "MEDIUM"),
    ("Pedestrian bridge stairs slippery with broken handrail over busy railway track", "Roads & Infrastructure", "HIGH"),
    ("Uneven road surface after poor patch work causing severe vehicular vibrations", "Roads & Infrastructure", "LOW"),
    ("Retaining wall near bridge showing deep structural cracks and tilting", "Roads & Infrastructure", "CRITICAL"),

    # --- Parks ---
    ("Heavy dead tree branch snapped and hanging precariously over children play area swing", "Parks", "HIGH"),
    ("Large tree uprooted during storm blocking walking track and damaging park perimeter fence", "Parks", "HIGH"),
    ("Broken metal slide with sharp rusted edge posing cut hazard to small children", "Parks", "HIGH"),
    ("Public park benches vandalized and wooden planks missing", "Parks", "LOW"),
    ("Sprinkler water system broken flooding lawn and creating stagnant mosquito breeding pool", "Parks", "MEDIUM"),
    ("Overgrown thorny bushes and weeds encroaching onto jogging track", "Parks", "LOW"),
    ("Public park gate broken allowing stray cattle and dogs to enter and damage flower beds", "Parks", "LOW"),
    ("Park lighting completely non-functional making park unsafe after 7 PM", "Parks", "MEDIUM"),
    ("Outdoor gym fitness equipment broken with loose bolts and missing pedals", "Parks", "LOW"),
    ("Dried up unwatered lawn and dead ornamental plants in neighborhood civic park", "Parks", "LOW"),
    ("Dry tree about to collapse onto adjacent electrical overhead line", "Parks", "CRITICAL")
]


def build_augmented_dataset(multiplier: int = 15) -> pd.DataFrame:
    """
    Generates a rich, diversified civic complaints dataset with variations,
    location tags, noise, and realistic citizen phrasing.
    """
    templates_prefix = [
        "Urgent: ",
        "Please look into this issue: ",
        "Complaint regarding ",
        "Severe problem of ",
        "Citizen report: ",
        "Issue observed at ",
        "There is an ongoing problem of ",
        "Attention authorities: ",
        "Immediate action needed for ",
        "Notice of ",
        "Repeated complaint: ",
        ""
    ]

    location_suffixes = [
        " near main bus stop",
        " opposite to market complex",
        " in front of government school",
        " at 2nd cross, 4th main",
        " near metro station gate 2",
        " on outer ring road",
        " behind civic hospital",
        " in residential colony lane",
        " near temple junction",
        ""
    ]

    records = []
    random.seed(42)

    for item in RAW_COMPLAINT_DATA:
        base_text, dept, prio = item
        # Add original base
        records.append({
            "text": base_text,
            "department": dept,
            "priority": prio,
            "ward": random.choice(WARDS)
        })

        # Generate augmented variations
        for _ in range(multiplier):
            prefix = random.choice(templates_prefix)
            suffix = random.choice(location_suffixes)
            ward = random.choice(WARDS)

            # Insert ward name inside or at end
            if random.random() > 0.5:
                text_variant = f"{prefix}{base_text}{suffix}, {ward}."
            else:
                text_variant = f"{prefix}{base_text} in {ward}{suffix}."

            records.append({
                "text": text_variant,
                "department": dept,
                "priority": prio,
                "ward": ward
            })

    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def train_and_evaluate(
    output_path: str = "model.pkl",
    evaluate: bool = True,
    test_size: float = 0.20
) -> Tuple[CivicMLBundle, Dict[str, Any]]:
    """
    Executes end-to-end model training, evaluation, and serialization.
    """
    logger.info("Building augmented training dataset for CivicPulse...")
    df = build_augmented_dataset(multiplier=20)
    logger.info("Dataset generated with %d total complaint samples.", len(df))
    logger.info("Department distribution:\n%s", df["department"].value_counts().to_string())
    logger.info("Priority distribution:\n%s", df["priority"].value_counts().to_string())

    X = df["text"].tolist()
    y_dept = df["department"].tolist()
    y_prio = df["priority"].tolist()

    # Split for robust evaluation
    X_train, X_test, y_dept_train, y_dept_test, y_prio_train, y_prio_test = train_test_split(
        X, y_dept, y_prio, test_size=test_size, random_state=42, stratify=y_dept
    )

    logger.info("Training set size: %d | Test set size: %d", len(X_train), len(X_test))

    # 1. Train Department Model
    logger.info("Training Civic Department NLP Model...")
    dept_model = CivicDepartmentModel(max_features=6000, alpha=0.1)
    dept_model.fit(X_train, y_dept_train)
    y_dept_pred = dept_model.predict(X_test)
    dept_acc = accuracy_score(y_dept_test, y_dept_pred)
    dept_f1 = f1_score(y_dept_test, y_dept_pred, average="weighted")
    dept_report = classification_report(y_dept_test, y_dept_pred, digits=4)

    # 2. Train Priority Model
    logger.info("Training Civic Priority NLP Model...")
    prio_model = CivicPriorityModel(max_features=4000, alpha=0.2)
    prio_model.fit(X_train, y_prio_train)
    y_prio_pred = prio_model.predict(X_test)
    prio_acc = accuracy_score(y_prio_test, y_prio_pred)
    prio_f1 = f1_score(y_prio_test, y_prio_pred, average="weighted")
    prio_report = classification_report(y_prio_test, y_prio_pred, digits=4)

    # 3. Fit Duplicate Detection Corpus
    logger.info("Fitting Duplicate Detection vectorizer on corpus...")
    dup_detector = CivicDuplicateDetector(similarity_threshold=0.65)
    dup_detector.fit(X)

    # Collect Metrics Summary
    metrics_summary = {
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "department_accuracy": round(float(dept_acc), 4),
        "department_f1_weighted": round(float(dept_f1), 4),
        "priority_accuracy": round(float(prio_acc), 4),
        "priority_f1_weighted": round(float(prio_f1), 4),
        "departments_count": len(DEPARTMENTS),
        "priorities_count": len(PRIORITIES),
    }

    # Package Bundle
    bundle = CivicMLBundle(
        department_model=dept_model,
        priority_model=prio_model,
        duplicate_detector=dup_detector,
        metrics=metrics_summary
    )

    # Save to disk
    bundle.save(output_path)
    logger.info("Successfully trained and saved model bundle to '%s'!", output_path)

    if evaluate:
        print("\n" + "=" * 70)
        print("  CIVICPULSE ML TRAINING & EVALUATION REPORT")
        print("=" * 70)
        print(f"\n[+] Total Samples Trained: {len(df)}")
        print(f"[+] Department Classification Accuracy: {dept_acc * 100:.2f}% (F1-Weighted: {dept_f1:.4f})")
        print(f"[+] Priority Classification Accuracy:   {prio_acc * 100:.2f}% (F1-Weighted: {prio_f1:.4f})")
        print("\n--- Department Classification Report ---")
        print(dept_report)
        print("\n--- Priority Classification Report ---")
        print(prio_report)
        print("=" * 70 + "\n")

    return bundle, metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Train CivicPulse NLP Models and export model.pkl")
    parser.add_argument("--output", "-o", type=str, default="model.pkl", help="Output path for model.pkl")
    parser.add_argument("--no-eval", action="store_true", help="Skip verbose evaluation output")
    args = parser.parse_args()

    train_and_evaluate(
        output_path=args.output,
        evaluate=(not args.no_eval)
    )


if __name__ == "__main__":
    main()
