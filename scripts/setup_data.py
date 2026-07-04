"""
setup_data.py: Telecharge et prepare les 200 images RSNA pour le projet.

Usage:
    python scripts/setup_data.py

Prerequis:
    pip install kaggle
    Mettre kaggle.json dans ~/.kaggle/ (depuis kaggle.com > Account > API)
    OU definir KAGGLE_USERNAME et KAGGLE_KEY dans .env
"""

import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check_kaggle():
    try:
        import kaggle  # noqa
        return True
    except ImportError:
        print("Installation de kaggle...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "-q"], check=True)
        return True


def download_rsna():
    raw_dir = ROOT / "data" / "rsna" / "raw"
    if (raw_dir / "input" / "images").exists() and len(list((raw_dir / "input" / "images").glob("*.jpg"))) > 100:
        print(f"Images brutes deja presentes ({len(list((raw_dir / 'input' / 'images').glob('*.jpg')))} fichiers).")
        return raw_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    print("Telechargement depuis Kaggle (rsna-pneumonia-detection-2018)...")
    print("Cela peut prendre 5 a 10 minutes...")

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

    subprocess.run([
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", "sovitrath/rsna-pneumonia-detection-2018",
        "-p", str(raw_dir), "--unzip"
    ], check=True)

    print("Telechargement termine.")
    return raw_dir


def run_preprocessing():
    processed_dir = ROOT / "data" / "rsna" / "processed" / "images"
    cases_csv = ROOT / "data" / "rsna" / "cases.csv"

    if processed_dir.exists() and len(list(processed_dir.glob("*.png"))) >= 200 and cases_csv.exists():
        print(f"Images deja preprocessees : {len(list(processed_dir.glob('*.png')))} PNG + cases.csv present.")
        return

    print("Lancement du preprocessing (etape 1)...")
    nb = ROOT / "notebooks" / "01_baseline_vlm.ipynb"
    if nb.exists():
        print(f"Ouvrir {nb} et executer les cellules de preprocessing.")
        print("Ou lancer directement le script de preprocessing :")
        print(f"  python -c \"import sys; sys.path.insert(0, '{ROOT}'); from src.preprocessing import preprocess_image\"")
    else:
        print("Notebook 01_baseline_vlm.ipynb introuvable.")


def main():
    print("=== Setup donnees RSNA ===\n")

    check_kaggle()
    download_rsna()
    run_preprocessing()

    processed = ROOT / "data" / "rsna" / "processed" / "images"
    cases = ROOT / "data" / "rsna" / "cases.csv"
    n_png = len(list(processed.glob("*.png"))) if processed.exists() else 0

    print("\n=== Bilan ===")
    print(f"Images PNG processed : {n_png}/200")
    print(f"cases.csv present    : {cases.exists()}")
    if n_png >= 200 and cases.exists():
        print("\nSetup complet. Tu peux lancer l'app :")
        print("  cd app && python -m streamlit run streamlit_app.py")
    else:
        print("\nSetup incomplet. Lance le notebook 01_baseline_vlm.ipynb pour generer les 200 images.")


if __name__ == "__main__":
    main()
