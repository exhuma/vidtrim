from setuptools import setup, find_packages

version = open('vidtrim/version.txt').read().strip()

setup(
    name="vidtrim",
    version=version,
    packages=find_packages(),
    install_requires=[
        'raspicam',
        'opencv-python',
    ],
    entry_points={
        'console_scripts': [
            'vidtrim=vidtrim.main:main'
        ]
    },
    provides=['vidtrim'],
    include_package_data=True,
    author="Michel Albert",
    author_email="michel@albert.lu",
    description="Simple Python motion detection thing",
    license="MIT",
    url="https://github.com/exhuma/vidtrim",
)
