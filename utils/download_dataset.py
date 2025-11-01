import os
import requests
import zipfile
from tqdm import tqdm
import gdown

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def download_iam_from_google_drive(config):
    """
    Downloads IAM dataset from Google Drive backup
    Alternative source since official site is down
    """
    print("=" * 80)
    print("📥 IAM DATASET DOWNLOADER")
    print("=" * 80)
    print("⚠️  Note: Official IAM website is currently unavailable")
    print("📦 Downloading from alternative source (Google Drive backup)...")
    print()
    
    # Google Drive file IDs (backup hosted by community)
    # These are publicly available IAM dataset backups
    WORDS_IMAGES_ID = "1QzUOn2V7FNriHV6PeREEFxr_A5TWgsn0"  # words.tgz
    WORDS_TXT_ID = "1bV_gPLxjxGZ_ylVmTy7gXZYdcYR3Hxqj"      # words.txt
    
    words_tgz_path = os.path.join(config.DATA_ROOT, 'words.tgz')
    words_txt_path = config.WORDS_TXT
    
    try:
        # Download words.txt
        if not os.path.exists(words_txt_path):
            print("📄 Downloading words.txt...")
            gdown.download(
                f"https://drive.google.com/uc?id={WORDS_TXT_ID}",
                words_txt_path,
                quiet=False
            )
            print("✅ words.txt downloaded successfully")
        else:
            print("✅ words.txt already exists")
        
        # Download words.tgz
        if not os.path.exists(words_tgz_path):
            print("\n📦 Downloading words.tgz (this may take a while ~500MB)...")
            gdown.download(
                f"https://drive.google.com/uc?id={WORDS_IMAGES_ID}",
                words_tgz_path,
                quiet=False
            )
            print("✅ words.tgz downloaded successfully")
        else:
            print("✅ words.tgz already exists")
        
        # Extract words.tgz
        if not os.path.exists(config.WORDS_DIR):
            print("\n📂 Extracting words.tgz...")
            import tarfile
            with tarfile.open(words_tgz_path, 'r:gz') as tar:
                tar.extractall(path=config.DATA_ROOT)
            print("✅ Extraction complete")
        else:
            print("✅ words directory already exists")
        
        print("\n" + "=" * 80)
        print("✅ IAM DATASET READY!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\n📌 MANUAL DOWNLOAD INSTRUCTIONS:")
        print("   1. Visit: https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database")
        print("   2. Download the dataset")
        print(f"   3. Extract to: {config.DATA_ROOT}")
        print("   4. Ensure structure: data/iam/words/ and data/iam/words.txt")
        return False

def download_from_kaggle(config):
    """
    Alternative: Download from Kaggle using kaggle API
    Requires: pip install kaggle
    And kaggle.json in ~/.kaggle/
    """
    print("📥 Attempting to download from Kaggle...")
    try:
        import kaggle
        kaggle.api.dataset_download_files(
            'nibinv23/iam-handwriting-word-database',
            path=config.DATA_ROOT,
            unzip=True
        )
        print("✅ Downloaded from Kaggle successfully")
        return True
    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        return False

def setup_dataset(config):
    """Main function to setup IAM dataset"""
    # Check if dataset already exists
    if os.path.exists(config.WORDS_TXT) and os.path.exists(config.WORDS_DIR):
        print("✅ IAM dataset already exists")
        return True
    
    print("\n🔍 IAM dataset not found. Setting up...")
    
    # Try Google Drive first
    if download_iam_from_google_drive(config):
        return True
    
    # Try Kaggle as fallback
    print("\n📌 Trying alternative source (Kaggle)...")
    if download_from_kaggle(config):
        return True
    
    # If all fails, show manual instructions
    print("\n" + "=" * 80)
    print("⚠️  AUTOMATIC DOWNLOAD FAILED")
    print("=" * 80)
    print("Please download manually:")
    print("1. Option A - Kaggle:")
    print("   https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database")
    print("\n2. Option B - GitHub:")
    print("   https://github.com/arthurflor23/handwritten-text-recognition")
    print(f"\n3. Extract to: {config.DATA_ROOT}")
    print("=" * 80)
    
    return False
