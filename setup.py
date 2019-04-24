from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='example to run keras on gcloud ml-engine',
      author='Lee Alessandrini',
      author_email='alessandrinil@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py'
      ],
      zip_safe=False)