#!/usr/bin/env python3
"""
Deploy FootyPredict V4 Training to HuggingFace Spaces
Uses Python API for deployment
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import shutil
import tempfile

# Configuration
SPACE_NAME = "nananie143/footypredict-v4-training"
LOCAL_DIR = Path(__file__).parent
FILES_TO_UPLOAD = [
    "app.py",
    "requirements.txt",
    "enhanced_engineering.py",
    "enhanced_training.py",
    "README.md",
    "data/merged_training_data.parquet",
]

def main():
    api = HfApi()
    user_info = api.whoami()
    print(f"✅ Logged in as: {user_info['name']}")
    
    # Create the Space if it doesn't exist
    print(f"\n📦 Creating/updating Space: {SPACE_NAME}")
    try:
        create_repo(
            repo_id=SPACE_NAME,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False
        )
        print(f"✅ Space created/verified: {SPACE_NAME}")
    except Exception as e:
        print(f"Space creation: {e}")
    
    # Upload files
    print("\n📤 Uploading files...")
    for file_path in FILES_TO_UPLOAD:
        full_path = LOCAL_DIR / file_path
        if full_path.exists():
            print(f"  Uploading: {file_path}")
            api.upload_file(
                path_or_fileobj=str(full_path),
                path_in_repo=file_path,
                repo_id=SPACE_NAME,
                repo_type="space"
            )
            print(f"  ✅ Uploaded: {file_path}")
        else:
            print(f"  ⚠️ File not found: {file_path}")
    
    print(f"\n" + "="*60)
    print(f"✅ Deployment complete!")
    print(f"")
    print(f"🌐 View Space: https://huggingface.co/spaces/{SPACE_NAME}")
    print(f"")
    print(f"📝 Next steps:")
    print(f"   1. Go to Settings > Hardware on the Space page")
    print(f"   2. Select T4 GPU (free tier)")
    print(f"   3. Click 'Restart Space'")
    print(f"="*60)


if __name__ == "__main__":
    main()
