# Migration Guide: New Project Structure

This guide explains the recent restructuring and how to adapt to the new professional layout.

## What Changed?

The project has been reorganized from a flat structure to a modular, industry-standard layout that improves maintainability, testability, and scalability.

## Directory Changes

| Old Location | New Location | Description |
|-------------|--------------|-------------|
| `app.py` | `src/app.py` | Main Flask application |
| `gradcam_generator.py` | `src/explainability/gradcam.py` | Grad-CAM module |
| `lime_explainer.py` | `src/explainability/lime_explainer.py` | LIME module |
| `segmentation_visualizer.py` | `src/explainability/segmentation.py` | Segmentation module |
| `download_as_jpeg.py` | `scripts/download_as_jpeg.py` | Utility script |
| `generate_*.py` | `scripts/generate_*.py` | Generation scripts |
| `test_*.py` | `tests/test_*.py` | Test files |
| `test_*.png` | `tests_output/*.png` | Test outputs |
| `DEPLOYMENT.md` | `docs/DEPLOYMENT.md` | Documentation |
| `CONTRIBUTING.md` | `docs/CONTRIBUTING.md` | Documentation |
| `MICCAI_Methodology.md` | `docs/MICCAI_Methodology.md` | Documentation |
| `ensemble output/` | `models/ensemble/` | Classification models |
| `detection output/` | `models/detection/` | Detection models |
| `Testing/` | `data/samples/` | Test DICOM files |
| `uploads/` | `data/uploads/` | Runtime uploads |
| `runs/` | `tests_output/runs/` | Test runs |
| `gunicorn_config.py` | `config/gunicorn_config.py` | Server config |
| `.env.example` | `config/.env.example` | Env template |

## Running the Application

### Before (Old Structure)
```bash
python app.py
# or
gunicorn --config gunicorn_config.py app:app
```

### After (New Structure)
```bash
# Development
python run.py

# Production
gunicorn --config config/gunicorn_config.py src.app:app
```

## Import Changes

### Before (Old Structure)
```python
import app
from lime_explainer import generate_lime
from gradcam_generator import generate_gradcam
```

### After (New Structure)
```python
from src import app
from src.explainability.lime_explainer import generate_lime
from src.explainability.gradcam import generate_gradcam
```

## Running Tests

### Before
```bash
pytest test_lime.py
pytest test_ensemble_lime.py
```

### After
```bash
pytest tests/
pytest tests/test_lime.py
pytest tests/test_ensemble_lime.py
```

## Running Scripts

### Before
```bash
python download_as_jpeg.py sample.dicom
python generate_preview.py sample.dicom
```

### After
```bash
python scripts/download_as_jpeg.py data/samples/sample.dicom
python scripts/generate_preview.py data/samples/sample.dicom
```

## Docker Changes

The Dockerfile has been updated to use the new structure. Rebuild your image:

```bash
docker build -t spinal-lesion-detection .
docker run -p 5000:5000 spinal-lesion-detection
```

## Configuration Changes

### Environment Variables
Move your `.env` file to `config/.env` or recreate from template:
```bash
cp config/.env.example config/.env
```

Then edit with your values.

### Gunicorn Config
The gunicorn configuration has moved to `config/gunicorn_config.py`. Update your deployment scripts:

```bash
# Old
gunicorn --config gunicorn_config.py app:app

# New
gunicorn --config config/gunicorn_config.py src.app:app
```

## Path References in Code

All file paths have been updated internally. The application now uses:
- `models/ensemble/` for classification models
- `models/detection/` for detection models
- `data/uploads/` for file uploads
- `data/samples/` for test data

Project root is automatically detected, so the app works whether run from root or src/ directory.

## Git Considerations

### Update Your Working Tree
```bash
# If you have uncommitted changes, stash them first
git stash

# Pull the restructured code
git pull

# Apply your stashed changes
git stash pop
```

### .gitignore Updates
The `.gitignore` has been updated to reflect new paths:
- `data/uploads/*` (instead of `uploads/*`)
- `tests_output/*` (instead of `runs/*`)
- `models/` paths (instead of `*output/` paths)

## Benefits of New Structure

### For Developers
- **Clear Separation**: Source code, tests, scripts, and documentation are logically separated
- **Python Package**: Proper `__init__.py` files enable package imports
- **IDE Support**: Better autocomplete and refactoring support
- **Scalability**: Easy to add new modules within organized directories

### For Reviewers
- **Professional Layout**: Follows industry best practices for ML/AI projects
- **Easy Navigation**: Clear directory structure makes code review easier
- **Documentation**: README files in subdirectories explain each component
- **Testability**: Dedicated tests/ directory with clear organization

### For Deployment
- **Clean Root**: Fewer files in project root, cleaner repository
- **Configuration**: Centralized configuration in config/ directory
- **Docker Friendly**: Logical structure maps well to container layers
- **CI/CD Ready**: Clear separation enables better pipeline design

## Backwards Compatibility

The new structure maintains **functional compatibility**—all features work as before. However, direct imports and scripts need path updates as shown above.

## Need Help?

- See `STRUCTURE.md` for detailed directory documentation
- Check `README.md` for updated setup and usage instructions
- Review `docs/DEPLOYMENT.md` for production deployment
- Read directory-specific README files for module details

## Rollback (If Needed)

If you need the old structure temporarily:
```bash
# Go back to the commit before restructuring
git log --oneline  # Find the commit hash
git checkout <hash-before-restructure>
```

However, we recommend adapting to the new structure as it provides significant long-term benefits.

---
**Migration Date**: February 26, 2026  
**Structure Version**: 2.0
