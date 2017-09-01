from setuptools import setup, find_packages

setup(
    name='qtree',
    version="0.1",
    description="A project for experimenting with particle quadtrees",
    url="https://github.com/ngoldbaum/qtree",
    author="Nathan Goldbaum",
    author_email="ngoldbau@illinois.edu",
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords="quadtree particle",
    packages=find_packages(),
    install_requires=[],
)
