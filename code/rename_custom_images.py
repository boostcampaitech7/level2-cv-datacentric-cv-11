import os
from pathlib import Path

def rename_files(directory, prefix='image', insert='cu'):
    """
    지정된 디렉토리 내의 모든 'image_{number}.jpg' 파일명을 'image.cu.{number}.jpg'로 변경합니다.

    Args:
        directory (str): 파일명을 변경할 대상 디렉토리 경로.
        prefix (str, optional): 파일명의 접두사. 기본값은 'image'.
        insert (str, optional): 파일명에 삽입할 문자열. 기본값은 'cu'.
    """
    p = Path(directory)
    if not p.exists():
        print(f"Directory does not exist: {directory}")
        return

    for file in p.iterdir():
        if file.is_file() and file.suffix.lower() == '.jpg':
            # 파일명 분리: 'image_{number}.jpg'
            parts = file.stem.split('_')
            if len(parts) != 2 or parts[0] != prefix:
                print(f"Skipping file with unexpected name format: {file.name}")
                continue
            base, number = parts
            new_name = f"{base}.{insert}.{number}{file.suffix}"
            new_path = p / new_name

            # 파일명이 이미 변경된 경우 건너뜀
            if new_path.exists():
                print(f"Target file already exists, skipping: {new_name}")
                continue

            print(f"Renaming '{file.name}' to '{new_name}'")
            try:
                file.rename(new_path)
            except Exception as e:
                print(f"Failed to rename '{file.name}' to '{new_name}': {e}")

def main():
    # 기본 디렉토리 경로 설정
    base_dir = '/home/naring/workplace/code/data/custom_receipt/img'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    # Train 디렉토리 파일명 변경
    print("Renaming files in 'train' directory...")
    rename_files(train_dir)

    # Val 디렉토리 파일명 변경
    print("\nRenaming files in 'val' directory...")
    rename_files(val_dir)

    print("\n파일명 변경 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
