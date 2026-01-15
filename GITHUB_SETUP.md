# GitHub Setup & Deployment Guide

## 🚀 Publishing to GitHub

### Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the **"+"** icon → **"New repository"**
3. Fill in the details:
   - **Repository name**: `spinal-injury-detection`
   - **Description**: `AI-powered web app for detecting spinal lesions from DICOM X-ray images`
   - **Visibility**: ✅ **Public** (for public access)
   - ❌ Do NOT initialize with README (we already have one)
4. Click **"Create repository"**

### Step 2: Connect Local Folder to GitHub

Open PowerShell in your project directory and run:

```powershell
# Navigate to project directory
cd "d:\Languages\Deploy-Model"

# Initialize git (if not already done)
git init

# Add all files to staging
git add .

# Create first commit
git commit -m "Initial commit: Spinal Injury Detection System"

# Add your GitHub repository as remote
# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/spinal-injury-detection.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Verify Upload

1. Refresh your GitHub repository page
2. You should see all your files uploaded
3. Check that README.md is displayed on the main page

---

## 🌐 Deployment Options

### Option 1: Render.com (Recommended - Free Tier Available)

**Pros:** Free tier, easy setup, persistent storage

1. Go to [Render.com](https://render.com) and sign up
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `spinal-injury-detection`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free (or paid for better performance)
5. Click **"Create Web Service"**
6. Wait 5-10 minutes for deployment
7. Access your app at: `https://spinal-injury-detection.onrender.com`

**Note:** Free tier sleeps after inactivity. First request may be slow.

---

### Option 2: Heroku (Paid Plans)

**Pros:** Reliable, good documentation

```powershell
# Install Heroku CLI first: https://devcenter.heroku.com/articles/heroku-cli

# Login to Heroku
heroku login

# Create new app
heroku create spinal-injury-detection

# Add Python buildpack
heroku buildpacks:set heroku/python

# Push to Heroku
git push heroku main

# Open your app
heroku open

# View logs
heroku logs --tail
```

Your app will be at: `https://spinal-injury-detection.herokuapp.com`

**Important:** Heroku no longer has a free tier. Basic plans start at $7/month.

---

### Option 3: Railway.app (Easy & Affordable)

**Pros:** $5 free credit, simple deployment

1. Go to [Railway.app](https://railway.app) and sign up with GitHub
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your repository
4. Railway auto-detects Python and deploys
5. Add environment variables if needed
6. Get your public URL

---

### Option 4: Google Cloud Run (Serverless)

**Pros:** Pay-per-use, scales to zero, generous free tier

```powershell
# Install Google Cloud SDK first

# Login
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud run deploy spinal-injury-detection `
  --source . `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 2

# Your app URL will be provided after deployment
```

---

### Option 5: Docker + Any Cloud Platform

**Build Docker Image:**
```powershell
# Build the image
docker build -t spinal-injury-detection .

# Test locally
docker run -p 5000:5000 spinal-injury-detection

# Tag for deployment (example for Docker Hub)
docker tag spinal-injury-detection YOUR_DOCKERHUB_USERNAME/spinal-injury-detection

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/spinal-injury-detection
```

Then deploy to:
- **AWS ECS/Fargate**
- **Azure Container Instances**
- **Google Cloud Run**
- **DigitalOcean App Platform**

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure:

- ✅ All model files are in correct directories
- ✅ `requirements.txt` is complete
- ✅ `.gitignore` excludes unnecessary files
- ✅ `uploads/` directory exists
- ✅ No hardcoded secrets or API keys
- ✅ README.md has clear instructions
- ✅ Medical disclaimer is visible

---

## 🔧 Configuration for Production

### Environment Variables (Optional)

Create `.env` file for local development:
```env
FLASK_ENV=production
MAX_CONTENT_LENGTH=16777216
SECRET_KEY=your-secret-key-here
```

Add to `.gitignore`:
```
.env
```

### Update app.py for production

For production deployments, modify the last lines in `app.py`:

```python
if __name__ == '__main__':
    load_models()
    # Use environment variables for production
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
```

---

## 🎯 Making Your Repository Discoverable

### Add Topics to GitHub

On your GitHub repository page:
1. Click **"⚙️ Settings"** (repository settings, not account)
2. Scroll to **"Topics"**
3. Add relevant topics:
   - `medical-imaging`
   - `deep-learning`
   - `healthcare-ai`
   - `spine-detection`
   - `dicom`
   - `pytorch`
   - `flask`
   - `computer-vision`
   - `medical-ai`
   - `object-detection`

### Create GitHub Pages (Optional)

For project documentation:
1. Go to repository **Settings** → **Pages**
2. Source: **Deploy from branch**
3. Branch: **main** → folder: **/ (root)**
4. Save
5. Your docs will be at: `https://YOUR_USERNAME.github.io/spinal-injury-detection`

### Add Badges to README

Add these at the top of your README.md:

```markdown
![Python](https://img.shields.io/badge/python-v3.9-blue)
![Flask](https://img.shields.io/badge/flask-3.0.0-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.1.0-red)
![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-active-success)
```

---

## 📊 Expected Repository Structure

```
spinal-injury-detection/
├── 📄 README.md                    # Main documentation
├── 📄 DEPLOYMENT.md                # Deployment instructions
├── 📄 GITHUB_SETUP.md             # This file
├── 🐍 app.py                      # Flask application
├── 📄 requirements.txt            # Python dependencies
├── 🐳 Dockerfile                  # Docker configuration
├── 📄 Procfile                    # Heroku configuration
├── 📄 runtime.txt                 # Python version
├── 📄 .gitignore                  # Git ignore rules
├── 📁 templates/
│   └── index.html                 # Web interface
├── 📁 static/                     # CSS/JS (if any)
├── 📁 uploads/                    # Temporary upload folder
│   └── .gitkeep                   # Keep directory in git
├── 📁 ensemble output/
│   ├── densenet121_balanced/
│   │   └── model_best.pth         # 80 MB
│   ├── resnet50_optimized/
│   │   └── model_best.pth         # 26 MB
│   └── tf_efficientnetv2_s_optimized/
│       └── model_best.pth         # 23 MB
├── 📁 detection output/
│   └── yolo11/
│       └── weights/
│           └── best.pt            # 48 MB
└── 📓 vindr-spinexr-dataset-analysis.ipynb
```

**Total Size:** ~180 MB (within GitHub's 100 MB file limit per file ✅)

---

## ⚠️ Important Notes

### Large File Handling

Your model files are under 100 MB each, so they'll work fine with regular git. If you had larger files, you'd need Git LFS:

```powershell
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "*.pt"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"
```

### Security Best Practices

1. **Never commit:**
   - API keys
   - Database credentials
   - Secret tokens
   - Private patient data

2. **Always use `.env` for sensitive data**

3. **Add security headers** in production

### License

Consider adding a LICENSE file:
- **MIT License**: Permissive, allows commercial use
- **GPL-3.0**: Open source, derivative works must be open
- **Apache 2.0**: Permissive with patent grant

For medical software, consult with legal experts.

---

## 🎉 After Deployment

### Share Your Project

1. **GitHub:** Add link to live demo in README
2. **LinkedIn:** Share your project
3. **Dev.to/Medium:** Write a blog post
4. **Twitter/X:** Tweet about it with hashtags
5. **Reddit:** Post in relevant subreddits (r/MachineLearning, r/Python)

### Monitor Your App

- Check logs regularly
- Monitor error rates
- Track user uploads (anonymously)
- Set up alerts for downtime

### Continuous Improvement

- Add user feedback form
- Implement analytics
- Add more lesion types
- Improve detection accuracy
- Add batch processing

---

## 🆘 Troubleshooting

### Push Rejected

```powershell
# If push is rejected, pull first
git pull origin main --rebase
git push origin main
```

### Large File Error

```
remote: error: File is XXX MB; this exceeds GitHub's file size limit of 100 MB
```

**Solution:** Your files are under 100MB, but if you get this:
1. Remove file from history: `git filter-branch`
2. Use Git LFS (see above)
3. Store models externally (S3, Google Drive)

### Deployment Timeout

If deployment times out during model loading:
- Increase memory allocation
- Use prebuilt Docker image
- Lazy load models on first request

---

## 📞 Support

- **GitHub Issues:** Use for bug reports
- **Discussions:** Use for questions
- **Pull Requests:** Welcome contributions!

---

**Ready to deploy? Follow Step 1-3 above!** 🚀
