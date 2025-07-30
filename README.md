# 🚧 Test Deployment Environment

This directory is the **clean environment** for testing model deployment workflows.

## 📌 Purpose

- Isolate **deployment tests** from other development activity
- Ensure a minimal and controlled workspace
- Provide a place to test deployment-related scripts, containers, APIs, etc.

## ✅ Guidelines for Contributors

- Only commit files related to testing deployment (e.g., Dockerfiles, test endpoints, CI setup)
- Avoid adding unrelated experimental or production code
- Structure test scripts and configs inside this folder

## 📂 Suggested Structure

test/
├── Docker/
├── API_tests/
├── README.md


## 📎 Notes

- This branch is **not** meant for production deployment.
- All original files are backed up in `backup_all_before_cleaning` branch.
