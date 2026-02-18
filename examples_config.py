from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "static"
pressure_examples = [str(EXAMPLES_DIR / f"pressure_{i}.jpg") for i in range(1, 4)]
no_pressure_examples = [str(EXAMPLES_DIR / f"no_pressure_{i}.jpg") for i in range(1, 4)]
examples = pressure_examples + no_pressure_examples