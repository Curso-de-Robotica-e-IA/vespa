## 📌 Examples
### Training a Model
```python
result = vespa.train("path/to/dataset")
print(result)
```

### Running Inference
```python
output = vespa.predict("path/to/image.jpg")
print(output)
```

### Batch Processing
```python
images = ["img1.jpg", "img2.jpg"]
results = vespa.batch_predict(image_paths=images)
print(results)
```