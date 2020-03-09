import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spellpy",
    version="0.0.1",
    author="nailo2c",
    author_email="nailo2c@gmail.com",
    description="Auto parser for system raw log without human guidance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nailo2c/pyspell",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
