## 📌 API Integration

Vespa can be used programmatically in larger applications.

### Example: Using Vespa in a Web API
```python
from flask import Flask, request, jsonify
import vespa

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    vespa.set_task(data["task"])
    vespa.set_model(data["model"])
    result = vespa.predict(data["image_path"])
    return jsonify(result)

if __name__ == "__main__":
    app.run()
```

For more details, check the official [documentation](../index.md).