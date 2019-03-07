from setuptools import setup, find_packages

setup(
    name='yolov3',
    version='0.1.0',
    author='hysts',
    url='https://github.com/hysts/pytorch_yolov3',
    install_requires=[
        'matplotlib',
        'numpy',
        'opencv-python',
        'pycocotools',
        'tensorboardX>=1.6',
        'torch>=1.0.1',
        'tqdm',
        'yacs>=0.1.6',
    ],
    packages=find_packages(exclude=('tests', )))
