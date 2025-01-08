from vespa.datasets.yolo.yolo_dataset import YOLODataset
from vespa.datasets.coco.coco_dataset import COCODataset
from vespa.datasets.pascalVOC.pascal_voc_dataset import PascalVOCDataset
from vespa.datasets.kitti.kitti_dataset import KITTIDataset


class DatasetFactory:
    """
    Fábrica para criar instâncias de datasets com base no tipo especificado.
    """

    @staticmethod
    def create_dataset(dataset_type, **kwargs):
        """
        Cria um dataset com base no tipo especificado.

        Args:
            dataset_type (str): Tipo do dataset (ex.: 'yolo', 'coco', 'pascal_voc', 'kitti').
            kwargs (dict): Argumentos necessários para inicializar o dataset.

        Returns:
            BaseDataset: Uma instância do dataset correspondente.

        Raises:
            ValueError: Caso o tipo do dataset não seja suportado.
        """
        if dataset_type == "yolo":
            return YOLODataset(
                root_dir=kwargs["root_dir"],
                txt_file=kwargs["txt_file"],
                image_size=kwargs["image_size"],
                transforms=kwargs.get("transforms", None),
            )
        elif dataset_type == "coco":
            return COCODataset(
                root_dir=kwargs["root_dir"],
                ann_file=kwargs["ann_file"],
                transforms=kwargs.get("transforms", None),
            )
        elif dataset_type == "pascal_voc":
            return PascalVOCDataset(
                root_dir=kwargs["root_dir"],
                transforms=kwargs.get("transforms", None),
            )
        elif dataset_type == "kitti":
            return KITTIDataset(
                root_dir=kwargs["root_dir"],
                label_dir=kwargs["label_dir"],
                transforms=kwargs.get("transforms", None),
            )
        else:
            raise ValueError(f"Dataset type '{dataset_type}' não suportado.")
