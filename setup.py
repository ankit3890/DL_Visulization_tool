from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nn_live",
    version="0.1.0",
    author="Ankit Kumar Singh",
    description="Real-Time Neural Network Visualizer for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ankit3890/nn_live",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=1.9.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "websockets>=10.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.7",
)
