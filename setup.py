#!/usr/bin/env python

from setuptools import setup, find_packages
import os

# Read the requirements from requirements.txt, excluding development dependencies
def read_requirements():
    requirements = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines, comments, and development dependencies
                if line and not line.startswith('#') and 'pytest' not in line and 'black' not in line and 'flake8' not in line:
                    requirements.append(line)
    return requirements

# Read the README file for long description
def read_readme():
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="llmproxy",
    version="0.1.0",
    author="LLMProxy Contributors",
    author_email="",
    description="An intelligent load balancer and proxy for LLM APIs",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llmproxy",  # Update this with your GitHub URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "llmproxy=llmproxy.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.yaml.example'],
    },
    zip_safe=False,
) 