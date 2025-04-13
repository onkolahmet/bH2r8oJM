import sys
import subprocess
import os
import platform
import time
from pathlib import Path

# Set environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

print(f"Running setup on platform: {platform.platform()}")

# Install pip if not available
def ensure_pip_installed():
    print("Checking if pip is installed...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, stdout=subprocess.PIPE)
        print("pip is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("pip is not installed. Installing pip...")
        try:
            subprocess.run(["curl", "https://bootstrap.pypa.io/get-pip.py", "-o", "get-pip.py"], check=True)
            subprocess.run([sys.executable, "get-pip.py"], check=True)
            os.remove("get-pip.py")
            print("pip installed successfully")
            return True
        except Exception as e:
            print(f"Failed to install pip: {e}")
            print("Please install pip manually and try again.")
            return False


# Create requirements.txt file
def create_requirements_file():
    requirements = [
        "transformers>=4.37.0",
        "datasets>=2.14.0",
        "trl>=0.7.4",
        "gradio>=3.50.0",
        "torch>=2.1.0",
        "evaluate>=0.4.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "huggingface_hub>=0.19.0",
        "sqlparse>=0.4.4",
        "bitsandbytes>=0.41.1",
        "peft>=0.6.0",
        "accelerate>=0.25.0",
        "scipy",
        "scikit-learn",
        "tensorboard",
        "tqdm>=4.66.0",
        "nltk>=3.8.1",
        "gradio>=3.50.0"
        ]
        

    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))

    print("Created requirements.txt file")

# Install dependencies from requirements.txt
def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All Python dependencies installed successfully!")
    except Exception as e:
        print(f"Warning: Failed to install some dependencies: {e}")
        print("Continuing anyway...")

# === Run setup steps ===
if __name__ == "__main__":
    if ensure_pip_installed():
        create_requirements_file()
        install_requirements()
