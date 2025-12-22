# GitHub Setup Instructions

## Step 1: Initialize Git Repository

Open PowerShell in your project directory and run:

```powershell
# Navigate to project directory (if not already there)
cd C:\Users\alisa\Downloads\CrowdDetecto\CrowdDetecto

# Initialize git repository
git init
```

## Step 2: Add All Files

```powershell
# Add all files (respects .gitignore)
git add .

# Check what will be committed
git status
```

## Step 3: Create Initial Commit

```powershell
git commit -m "Initial commit: Smart Crowd Detector

Features:
- Real-time YOLOv8 people detection with GPU acceleration
- Adaptive heatmap visualization with smart kernel sizing
- Support for camera and video file sources
- Video looping with seamless restart
- Web dashboard with real-time statistics
- Configurable alert thresholds (warning/critical)
- Performance optimizations: frame caching, deque, thread-safe state
- Bug fixes: 30+ issues resolved including memory leaks, race conditions, IOU edge cases"
```

## Step 4: Create GitHub Repository

1. Go to https://github.com/nowayitsme-eng
2. Click "+" → "New repository"
3. Repository name: `Smart_Crowd_Detector`
4. Description: `AI-powered real-time crowd detection using YOLOv8 with adaptive heatmap`
5. Choose: **Public** (or Private if preferred)
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

## Step 5: Connect to GitHub

```powershell
# Add your GitHub repository as remote
git remote add origin https://github.com/nowayitsme-eng/Smart_Crowd_Detector.git

# Verify remote was added
git remote -v
```

## Step 6: Push to GitHub

```powershell
# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Troubleshooting

### If push fails with authentication error:

**Option 1: Use GitHub CLI (Recommended)**
```powershell
# Install GitHub CLI: https://cli.github.com/
# Then authenticate
gh auth login

# Push again
git push -u origin main
```

**Option 2: Use Personal Access Token**
1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scopes: `repo` (full control)
4. Copy the token
5. When prompted for password, paste the token

### If files are too large:

```powershell
# Check for large files
git ls-files | xargs ls -lh | sort -k5 -h -r | head -20

# If model files are too large, ensure they're in .gitignore
echo "*.pt" >> .gitignore
echo "*.engine" >> .gitignore
git rm --cached *.pt *.engine
git commit -m "Remove large model files"
```

## What Gets Uploaded ✅

- ✅ Source code (src/, app.py)
- ✅ Frontend (static/, templates/)
- ✅ Configuration (config.yaml)
- ✅ Documentation (README.md, LICENSE)
- ✅ Dependencies (requirements.txt)
- ✅ Tests (test_fixes.py, test_resize.py)

## What Gets Ignored ❌

- ❌ videos/ (user uploads)
- ❌ models/ (*.pt, *.engine files)
- ❌ __pycache__/ (Python bytecode)
- ❌ .vscode/ (IDE settings)
- ❌ logs/ (log files)
- ❌ venv/ (virtual environment)

## After Upload

Your repository will be available at:
**https://github.com/nowayitsme-eng/Smart_Crowd_Detector**

Share it, star it, and keep building! 🚀

---

**Need help?** Check GitHub docs: https://docs.github.com/en/get-started
