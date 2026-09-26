# EV Fault Detection

Securing electric vehicle performance with machine-learning-driven fault detection and classification.

## Project Overview

This project is a Python-based web application for detecting and classifying faults in electric vehicles. It includes a dataset, trained model artifacts, and a web interface for users and administrators.

## Contents

- `App.py` - Main Flask application logic
- `Dataset/` - Original dataset CSV files
- `Model/` - Stored model and related assets
- `Static/` - Frontend assets: CSS, JavaScript, images
- `Templates/` - HTML templates for the web UI
- `uploads/` - Uploaded datasets

## Quick Start

1. Create a Python virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies. If a `requirements.txt` is available, use:

```powershell
pip install -r requirements.txt
```

If no requirements file exists, install Flask and any required ML packages manually.

3. Run the app:

```powershell
python App.py
```

4. Open the browser and visit:

```text
http://127.0.0.1:5000
```

## GitHub Push Instructions

If you have not yet added a GitHub remote, run:

```powershell
git remote add origin https://github.com/<your-username>/EV_Fault_Detection.git
git push -u origin Main
```

## Notes

- Keep large datasets and model files out of Git if possible.
- Add or update `requirements.txt` to document Python dependencies.
