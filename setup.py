import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fcpy",
    version="0.0.1",
    author="Kyle Chen",
    author_email="kchen513@sjtu.edu.cn",
    description="Python-based functional connectivity analyzer for ECoG neural data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeoNeuron/Monkey_ECoG",
    project_urls={
        "Bug Tracker": "https://github.com/NeoNeuron/Monkey_ECoG/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)