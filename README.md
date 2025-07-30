# 🚀 Dev Deployment Environment

This directory is the **working development environment** for model deployment.

## 📌 Purpose

- Serve as the **active branch** for building and refining deployment pipelines
- Collaborate on production-ready deployment code
- Provide a structured area for development and staging before final merging

## ✅ Guidelines for Contributors

- Push **only dev-related deployment files** (e.g., model serving, FastAPI, Flask, CI/CD configs)
- Keep code clean, modular, and documented
- Regularly pull the latest changes and coordinate with teammates to avoid conflicts

## 📂 Suggested Structure

dev/
├── app/
│ ├── main.py
│ ├── requirements.txt
├── Docker/
├── deployment_notes.md
├── README.md


## 🧱 Branch Policy

- This is a shared development space. Please **do not push broken or untested code**.
- Finalized versions will be merged into the professor's `main` or `dev` branch via PRs.

## 📎 Notes

- All previous files are backed up in `backup_all_before_cleaning`.
