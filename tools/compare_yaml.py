import argparse
from pathlib import Path

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", type=Path)
    parser.add_argument("file2", type=Path)
    args = parser.parse_args()

    with args.file1.open() as f1, args.file2.open() as f2:
        yaml1 = yaml.load(f1, Loader=yaml.FullLoader)
        yaml2 = yaml.load(f2, Loader=yaml.FullLoader)
    if yaml1 == yaml2:
        print("Two files are the same")
    else:
        print("Not equal!!!")
