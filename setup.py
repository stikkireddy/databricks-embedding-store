from setuptools import setup, find_packages

setup(
    name="delta-embeddings",
    author="Sri Tikkireddy",
    author_email="sri.tikkireddy@databricks.com",
    description="Databricks Embeddings Store In Delta",
    long_description="Store embeddings in delta tables in databricks and search",
    long_description_content_type="text/markdown",
    url="",
    license="",
    packages=find_packages(include=["*"], exclude=['tests', 'tests.*', ]),
    use_scm_version={
        "local_scheme": "dirty-tag"
    },
    setup_requires=['setuptools_scm'],
    install_requires=[
        'setuptools>=45',
        'sentence_transformers',
        'langchain',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
