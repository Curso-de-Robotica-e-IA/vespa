# Pascal VOC Dataset Format 📌

The **Pascal VOC format** is an XML-based annotation format used in object detection.

## Pascal VOC File Structure
- **JPEGImages/** - Folder with images.
- **Annotations/** - XML files with bounding boxes.

Example Pascal VOC annotation (`.xml`):
```xml
<annotation>
  <folder>VOC2007</folder>
  <filename>image1.jpg</filename>
  <object>
    <name>cat</name>
    <bndbox>
      <xmin>50</xmin>
      <ymin>50</ymin>
      <xmax>200</xmax>
      <ymax>200</ymax>
    </bndbox>
  </object>
</annotation>
```

## Loading Pascal VOC Data in Vespa
```python
from vespa.datasets import load_voc

dataset = load_voc("path/to/voc/dataset")
print(dataset)
```

For additional dataset configurations, check the API documentation.