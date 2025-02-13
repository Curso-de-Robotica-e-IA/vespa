# Configuration Guide ⚙️

After installing, you may need to configure some settings for specific use cases.

## 1️⃣ Environment Variables (Optional)
If your project requires custom configurations, set environment variables:

```ini
VESPA_CONFIG_PATH=/path/to/config.json
VESPA_LOG_LEVEL=INFO
```

## 2️⃣ Check Available Modules
To list available modules in Vespa, run:
```bash
python -c "import vespa; print(vespa.available_modules())"
```