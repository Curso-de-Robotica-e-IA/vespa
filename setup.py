from setuptools import setup, find_packages

setup(
    name="vespa",
    version="1.0.0",
    description="A Python library for classification, regression, and object detection.",
    author=["Matheus_Hopper", "Jefferson_Norberto"],
    author_email=["mhjc@softex.cin.ufpe.br", "jmn@softex.cin.ufpe.br"],
    url="https://github.com/Curso-de-Robotica-e-IA/vespa",
    packages=find_packages(where="vespa"),
    package_dir={"": "vespa"},
    install_requires=[
        "numpy==1.26.4",
        "opencv-python==4.10.0.84",
        "scikit-learn==1.6.0",
        "albumentations==1.4.24",
        "tqdm==4.67.1",
        "matplotlib==3.10.0",
        "onnxruntime==1.20.1",
        "pycocotools==2.0.8",
        "ruff==0.9.0",
        "pytest==8.3.4",
        "taskipy==1.14.1",
        "lark==1.2.2",
        "setuptools==75.8.0",
        "torch",
        "torchvision",
        "torchaudio"
    ],
    extras_require={
        "cuda": [
            "https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp312-cp312-win_amd64.whl; platform_system=='Windows'",
            "https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp312-cp312-linux_x86_64.whl; platform_system=='Linux'"
        ]
    },
    entry_points={
        "console_scripts": [
            "vespa=vespa.vespa.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)
