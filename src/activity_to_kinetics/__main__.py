# src/activity_to_kinetics/__main__.py
from .cli import parse_args
from .main import main as run  # din eksisterende main(args)

def main_entry():
    args = parse_args()
    run(args)

if __name__ == "__main__":
    main_entry()