from setuptools import setup

setup(name='allegro',
      version='0.0.1',
      description='Joyful data science toolkit',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering'
      ],
      keywords='machine-learning python statistics data-science data-analysis',
      url='https://github.com/thoriuchi0531/allegro',
      author='thoriuchi0531',
      author_email='thoriuchi0531@gmail.com',
      license='MIT',
      packages=['allegro'],
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'scikit-learn',
          'xgboost',
          'lightgbm',
          'catboost',
          'matplotlib',
          'hyperopt',
          'tqdm'
      ],
      zip_safe=False,
      include_package_data=True,
      python_requires='>=3.5',
      test_suite='nose.collector',
      tests_require=['nose'])
