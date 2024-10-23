from setuptools import setup, find_packages

setup(
    name='async_llm_handler',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'openai',
        'anthropic',
        'aiolimiter',
        'tiktoken',
        'pydantic',
        'python-dotenv',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='An asynchronous handler for LLM API interactions.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/async_llm_handler',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
