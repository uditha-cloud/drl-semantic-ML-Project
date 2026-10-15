import subprocess
import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"


# ============================================================
# RUN PYTHON FILE
# ============================================================

def run_script(script_name):

    script_path = SRC_DIR / script_name

    print("\n" + "=" * 70)
    print(f"RUNNING: {script_name}")
    print("=" * 70)

    result = subprocess.run(
        [
            sys.executable,
            str(script_path)
        ],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:

        print(
            f"\nERROR: {script_name} failed."
        )

        sys.exit(
            result.returncode
        )

    print(
        f"\n{script_name} completed successfully."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("DEEP REINFORCEMENT LEARNING-BASED")
    print("ADAPTIVE SEMANTIC COMPRESSION")
    print("FOR ENERGY-CONSTRAINED IoT COMMUNICATION")
    print("=" * 70)

    print("\nStarting complete ML pipeline...")

    # --------------------------------------------------------
    # STEP 1: Load dataset
    # --------------------------------------------------------

    run_script(
        "load_dataset.py"
    )

    # --------------------------------------------------------
    # STEP 2: Preprocess dataset
    # --------------------------------------------------------

    run_script(
        "preprocess.py"
    )

    # --------------------------------------------------------
    # STEP 3: Extract semantic features
    # --------------------------------------------------------

    run_script(
        "semantic_features.py"
    )

    # --------------------------------------------------------
    # STEP 4: Test compression module
    # --------------------------------------------------------

    run_script(
        "compression.py"
    )

    # --------------------------------------------------------
    # STEP 5: Test DRL environment
    # --------------------------------------------------------

    run_script(
        "environment.py"
    )

    # --------------------------------------------------------
    # STEP 6: Train DQN
    # --------------------------------------------------------

    run_script(
        "train_drl.py"
    )

    # --------------------------------------------------------
    # STEP 7: Evaluate trained model
    # --------------------------------------------------------

    run_script(
        "evaluate.py"
    )

    # --------------------------------------------------------
    # STEP 8: Generate graphs
    # --------------------------------------------------------

    run_script(
        "plots.py"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated outputs:")

    print("\nProcessed Dataset:")
    print(
        PROJECT_ROOT /
        "dataset" /
        "processed"
    )

    print("\nTrained Model:")
    print(
        PROJECT_ROOT /
        "models"
    )

    print("\nGraphs:")
    print(
        PROJECT_ROOT /
        "results" /
        "graphs"
    )

    print("\nProject execution completed.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()