from distutils.core import setup
import setuptools

setup(name='spellpie',
      version='0.0.1',
      description='Basic information extraction tool.',
      url='https://bitbucket.org/dcronkite/pytakes',
      author='dcronkite',
      author_email='dcronkite-gmail',
      license='MIT',
      classifiers=[  # from https://pypi.python.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Text Processing :: Linguistic',
      ],
      keywords='nlp information extraction',
      entry_points={
          'console_scripts':
              [
              ]
      },
      install_requires=['nltk', 'regex'],
      package_dir={'': 'src'},
      packages=setuptools.find_packages('src'),
      package_data={'spellpie': ['data/*.db']},
      zip_safe=False
      )