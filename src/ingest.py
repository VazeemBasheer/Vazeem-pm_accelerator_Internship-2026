from pathlib import Path
import shutil
import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASET_HANDLE = "nelgiriyewithana/global-weather-repository"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TARGET_CSV = RAW_DIR / "Global Weather Repository.csv"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = Path(kagglehub.dataset_download(DATASET_HANDLE, force_download=True))
    csv_files = list(cache_path.rglob("*.csv"))

    if csv_files:
        shutil.copy2(csv_files[0], TARGET_CSV)
    else:
        # Fallback: load directly via dataset_load if archive extraction yielded no local CSV
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            DATASET_HANDLE,
            "GlobalWeatherRepository.csv",
        )
        df.to_csv(TARGET_CSV, index=False)

    print(f"Exported raw dataset to {TARGET_CSV}")


if __name__ == "__main__":
    main()