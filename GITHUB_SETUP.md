# 📤 How to Publish ModernHTR on GitHub

## Step 1: Prepare Repository

### 1.1 Clean up sensitive data

```bash
cd "/Users/glebishchenko/Documents/codes/HTR cw"

# Remove large files that shouldn't be on GitHub
rm -rf data/iam/*  # Keep structure but remove dataset
rm -rf outputs/models/*.pth  # Remove trained models
rm -rf __pycache__  # Remove Python cache
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete
```

### 1.2 Keep only necessary files

✅ Keep:
- All `.py` files
- `requirements.txt`
- `README.md`
- `LICENSE`
- `.gitignore`
- `docs/` folder
- Empty `data/` structure

❌ Remove:
- Trained models (`.pth` files)
- Downloaded dataset
- Generated visualizations (can be regenerated)
- Virtual environment (`venv/`)

## Step 2: Initialize Git

```bash
# Navigate to project
cd "/Users/glebishchenko/Documents/codes/HTR cw"

# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: ModernHTR - Modern Handwritten Text Recognition"
```

## Step 3: Create GitHub Repository

### Option A: Using GitHub Website

1. Go to https://github.com/new
2. Repository name: `ModernHTR`
3. Description: `Modern Handwritten Text Recognition with PyTorch - CNN+BiLSTM+CTC`
4. Select **Public**
5. ⚠️ **Don't** initialize with README (we have one)
6. Click **Create repository**

### Option B: Using GitHub CLI

```bash
# Install GitHub CLI (if not installed)
brew install gh  # On Mac
# or visit: https://cli.github.com/

# Login
gh auth login

# Create repository
gh repo create ModernHTR --public --source=. --remote=origin --push
```

## Step 4: Connect & Push

### If created via website:

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/ModernHTR.git

# Check remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

### If used GitHub CLI:

Already done! Skip to Step 5.

## Step 5: Verify Upload

1. Visit: `https://github.com/YOUR_USERNAME/ModernHTR`
2. Check all files are there
3. Verify README displays correctly

## Step 6: Add Topics & Details

On GitHub repository page:

1. Click **⚙️ Settings**
2. Under **Topics**, add:
   - `pytorch`
   - `deep-learning`
   - `ocr`
   - `handwriting-recognition`
   - `computer-vision`
   - `ctc-loss`
   - `apple-silicon`
   - `python`

3. Update **About** section:
   - Website: (if you have one)
   - Description: "Modern HTR system with automatic dataset download and Apple Silicon optimization"

## Step 7: Create Release (Optional)

```bash
# Tag first version
git tag -a v1.0.0 -m "ModernHTR v1.0.0 - Initial Release"
git push origin v1.0.0
```

On GitHub:
1. Go to **Releases**
2. Click **Draft a new release**
3. Tag version: `v1.0.0`
4. Release title: `ModernHTR v1.0.0 - Initial Release`
5. Description:
```markdown
## 🎉 ModernHTR v1.0.0

First stable release of ModernHTR!

### Features
✅ CNN + BiLSTM + CTC architecture
✅ Automatic IAM dataset download
✅ Apple Silicon (M1/M2/M3) optimization
✅ 15+ comprehensive visualizations
✅ Production-ready code

### Performance
- CER: 14.60%
- WER: 35.09%
- Accuracy: 64.91%

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/ModernHTR.git
cd ModernHTR
pip install -r requirements.txt
python main.py
```

See [README.md](README.md) for full documentation.
```

6. Click **Publish release**

## Step 8: Add Shields/Badges

Badges already in README.md! They will show:
- Python version
- PyTorch version
- License
- Apple Silicon support

## Step 9: Enable GitHub Features

### Enable Issues
1. Go to **Settings** → **General**
2. Under **Features**, check ✅ **Issues**

### Enable Discussions (Optional)
1. Go to **Settings** → **General**
2. Under **Features**, check ✅ **Discussions**

### Add Description
In main repo page, click **Edit** (pencil icon) and add:
