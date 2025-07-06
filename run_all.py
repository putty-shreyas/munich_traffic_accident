from src.data_preprocessing import processor
from src.train_model import trainer
from pathlib import Path

ROOT = Path(__file__).parents[0].__str__()

def main():
    print("\nStep 1: Running data preprocessing...")
    processor(ROOT)

    print("\nStep 2: Training model...")
    trainer(ROOT)

if __name__ == main():
    main(ROOT)