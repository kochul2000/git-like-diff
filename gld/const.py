from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Test data paths
test_old_path = BASE_DIR / "data" / "old.xlsx"
test_new_path = BASE_DIR / "data" / "new.xlsx"
test_result_path = BASE_DIR / "data" / "diff.xlsx"