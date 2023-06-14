from setuptools import setup, find_namespace_packages

setup(name='mullet',
      packages=find_namespace_packages(include=["mullet", "mullet.*"]),
      version='1.0.1',
      description='Multi-phase Liver Lesion Segmentation. Framework for multi-phase liver lesions segmentation.',
      url='https://github.com/shenhai1895/Multi-phase-Liver-Lesion-Segmentation',
      author='Lei Wu',
      author_email='shenhai1895@zju.edu.cn',
      license='MIT License',
      python_requires=">=3.9",
      install_requires=[
          "torch~=2.0.1",
          "numpy~=1.24.3",
          "argparse~=1.4.0",
          "SimpleITK~=2.2.1",
          "tqdm~=4.65.0",
          "segmentation_models_pytorch~=0.3.2",
          "scikit-image~=0.19.3"
      ],
      entry_points={
          'console_scripts': [
              'mullet_predict = mullet.inference.predict:predict_entry_point',  # api available
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'mullet']
      )