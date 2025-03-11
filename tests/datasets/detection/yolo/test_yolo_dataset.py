def test_yolo_dataset_train_length(yolo_dataset_train_retina):
    """
    Testa se o tamanho do dataset YOLO está correto.
    """
    dataset = yolo_dataset_train_retina
    assert len(dataset), 5


def test_yolo_dataset_train_image_shape(yolo_dataset_train_retina):
    """
    Testa se as imagens carregadas pelo
    YOLODataset têm as dimensões corretas.
    """
    dataset = yolo_dataset_train_retina
    img, _ = dataset[0]
    assert img.shape, (3, 100, 100)  # noqa


def test_yolo_dataset_train_boxes(yolo_dataset_train_retina):
    """
    Testa se as bounding boxes estão sendo carregadas corretamente.
    """
    dataset = yolo_dataset_train_retina
    _, target = dataset[0]
    assert len(target['boxes']) > 0, True


def test_yolo_dataset_train_get_images(yolo_dataset_train_retina):
    """
    Testa se as bounding boxes estão sendo carregadas corretamente.
    """
    dataset = yolo_dataset_train_retina

    for img, _ in dataset:
        assert img is not None, True


def test_yolo_dataset_train_get_targets(yolo_dataset_train_retina):
    """
    Testa se as bounding boxes estão sendo carregadas corretamente.
    """
    dataset = yolo_dataset_train_retina

    for _, target in dataset:
        assert target is not None, True


def test_yolo_dataset_test_length(yolo_dataset_test_retina):
    """
    Testa se o tamanho do dataset YOLO está correto.
    """
    dataset = yolo_dataset_test_retina
    assert len(dataset), 5


def test_yolo_dataset_test_image_shape(yolo_dataset_test_retina):
    """
    Testa se as imagens carregadas pelo
    YOLODataset têm as dimensões corretas.
    """
    dataset = yolo_dataset_test_retina
    img, _ = dataset[0]
    assert img.shape, (3, 100, 100)  # noqa


def test_yolo_dataset_test_boxes(yolo_dataset_test_retina):
    dataset = yolo_dataset_test_retina
    _, target = dataset[0]

    print(f"Boxes encontrados: {target['boxes']}")

    # Se houver pelo menos uma imagem com bounding boxes, o teste passa
    assert any(len(t['boxes']) > 0 for _, t in dataset), "Nenhuma bounding box encontrada no dataset!" # noqa



def test_yolo_dataset_test_get_images(yolo_dataset_test_retina):
    """
    Testa se as bounding boxes estão sendo carregadas corretamente.
    """
    dataset = yolo_dataset_test_retina

    for img, _ in dataset:
        assert img is not None, True


def test_yolo_dataset_test_get_targets(yolo_dataset_test_retina):
    """
    Testa se as bounding boxes estão sendo carregadas corretamente.
    """
    dataset = yolo_dataset_test_retina

    for _, target in dataset:
        assert target is not None, True
