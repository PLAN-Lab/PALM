# Installation
To set up PALM, we will create an isolated environment called palm. This environment is designed to support pre-training, fine-tuning, and inference workflows.

## palm env
**(1) Env**
```python
conda create -n palm python=3.10 -y
conda activate palm
```
**(2) Third Party Packages**
```python
cd ${YOUR_PATH_TO_PALM}
pip install -r requirements.txt
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```
