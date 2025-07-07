from setuptools import setup, find_packages

setup(
    name="sidnet",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "torch", "matplotlib"],
    author="Ruihuan Zhang",
    description="Synergistic Information Decomposition for microbial interaction networks",
    url="https://github.com/zrhuan0318/SID-Net",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
