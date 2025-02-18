# Deployment Guide 🚀

## 📦 Building and Publishing the Package
### 1️⃣ Build the Package
```bash
poetry build
```

### 2️⃣ Publish to PyPI
```bash
poetry publish --build
```

Ensure you are authenticated with:
```bash
poetry config pypi-token.pypi YOUR_API_TOKEN
```