from setuptools import setup, find_packages

setup(
    name="lazy_GP",  # Replace with your project name
    version="1.2.0",
    packages=find_packages(),  # Automatically finds all packages in your Python directory
    install_requires=[],  # List dependencies here, e.g., ["numpy", "requests"]
    entry_points={
        "console_scripts": [
            # Define any command-line scripts here if needed
            # Example: "my_command = my_package.module:main_function"
        ],
    },
)