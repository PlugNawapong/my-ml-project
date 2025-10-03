# Cloud Training Setup Guide

This guide helps you train your hyperspectral material classification model on cloud platforms.

## Step 1: Push Code to GitHub

```bash
cd /Users/nawapong/Projects/dl-plastics

# Initialize git (if not already done)
git init

# Add your GitHub repository as remote
git remote add origin https://github.com/PlugNawapong/my-ml-project.git

# Add all files (excluding those in .gitignore)
git add .

# Commit
git commit -m "Add hyperspectral material classification pipeline"

# Push to GitHub
git push -u origin main
# (or 'master' if your default branch is master)
```

## Step 2: Upload Data to Cloud Storage

Since your data folders are large and excluded from git, you need to upload them separately:

### Option A: Google Drive (for Google Colab)
1. Upload `data/`, `inference_data_set1/`, `inference_data_set2/` to Google Drive
2. Note the folder IDs from the URLs

### Option B: AWS S3 (for AWS)
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Upload data
aws s3 cp data/ s3://your-bucket-name/dl-plastics/data/ --recursive
aws s3 cp inference_data_set1/ s3://your-bucket-name/dl-plastics/inference_data_set1/ --recursive
aws s3 cp inference_data_set2/ s3://your-bucket-name/dl-plastics/inference_data_set2/ --recursive
```

### Option C: Kaggle Datasets
1. Create dataset on Kaggle: https://www.kaggle.com/datasets
2. Upload data folders
3. Note the dataset name

## Step 3: Choose Cloud Platform

### **Option 1: Google Colab (FREE - Recommended for Beginners)**

**Pros:**
- Free GPU (Tesla T4)
- No setup required
- Easy to use
- 12 hours per session

**Cons:**
- Session timeout
- Limited storage
- May disconnect

**Setup:**
1. Go to: https://colab.research.google.com
2. Click "New Notebook"
3. Use the provided `colab_training.ipynb` notebook (see below)
4. Runtime → Change runtime type → GPU

---

### **Option 2: Kaggle Notebooks (FREE)**

**Pros:**
- Free GPU (Tesla P100/T4)
- 30 hours/week GPU quota
- Integrated with Kaggle datasets
- No disconnections

**Cons:**
- 30 hours/week limit
- Requires Kaggle account

**Setup:**
1. Go to: https://www.kaggle.com
2. Create new notebook
3. Settings → Accelerator → GPU T4 x2
4. Add dataset to notebook

---

### **Option 3: AWS SageMaker**

**Pros:**
- Powerful GPUs
- Scalable
- Professional

**Cons:**
- Costs money
- More complex setup

**Setup:**
1. Create AWS account
2. Go to SageMaker console
3. Create notebook instance (ml.p3.2xlarge for GPU)
4. Clone your GitHub repo

---

### **Option 4: Paperspace Gradient (PAID - $0.45/hr for V100)**

**Pros:**
- Affordable
- Good GPUs
- Simple interface
- No session timeout

**Cons:**
- Requires payment

**Setup:**
1. Sign up: https://www.paperspace.com/gradient
2. Create new notebook
3. Select GPU (V100 recommended)
4. Clone GitHub repo

---

### **Option 5: Lambda Labs (PAID - Best GPU Performance)**

**Pros:**
- Cheapest GPU rates ($0.50-$1.10/hr)
- Powerful GPUs (A100, H100)
- Best for serious training

**Cons:**
- GPU availability varies
- Requires some Linux knowledge

---

## Step 4: Training in the Cloud

### Basic Cloud Training Script:

```python
# Clone repository
!git clone https://github.com/PlugNawapong/my-ml-project.git
%cd my-ml-project

# Install dependencies
!pip install -r requirements.txt

# Download data (adjust based on your storage method)
# For Google Drive in Colab:
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/dl-plastics-data/data ./
!cp -r /content/drive/MyDrive/dl-plastics-data/inference_data_set1 ./
!cp -r /content/drive/MyDrive/dl-plastics-data/inference_data_set2 ./

# Train model
!python train.py --model spectral_cnn_1d --epochs 50 --batch_size 2048 \
    --max_samples_per_class 10000 --dropout 0.5 --lr 0.001

# Or train 2D model with patches
!python train.py --model hybrid --use_patches --patch_size 3 \
    --epochs 50 --batch_size 512 --max_samples_per_class 5000 \
    --dropout 0.5 --augment --bin_factor 2

# Run inference
!python inference.py \
    --checkpoint outputs/spectral_cnn_1d_*/best_model.pth \
    --model spectral_cnn_1d \
    --data_dir inference_data_set2

# Download results
# For Colab:
from google.colab import files
!zip -r results.zip outputs/ predictions/
files.download('results.zip')
```

## Step 5: Monitor Training

### Using TensorBoard (optional):
```python
# Install
!pip install tensorboard

# In training script, add:
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# Log metrics
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Accuracy/train', train_acc, epoch)

# View in notebook
%load_ext tensorboard
%tensorboard --logdir runs
```

### Using Weights & Biases (optional):
```python
# Install
!pip install wandb

# Login
!wandb login

# In training script:
import wandb
wandb.init(project='dl-plastics', name='experiment_1')
wandb.log({'train_loss': train_loss, 'train_acc': train_acc})
```

## Recommended Workflow

1. **Start with Google Colab** (free, easy)
2. Test your training script with small epochs (5-10)
3. If it works, run full training (50-100 epochs)
4. If you need more GPU time, switch to Kaggle
5. For production/serious work, consider paid options

## Cost Comparison

| Platform | GPU | Cost | Free Tier |
|----------|-----|------|-----------|
| Google Colab | T4 | Free | 12 hrs/session |
| Kaggle | P100/T4 | Free | 30 hrs/week |
| Paperspace | V100 | $0.45/hr | $10 credit |
| Lambda Labs | A100 | $1.10/hr | No |
| AWS SageMaker | V100 | $3.06/hr | Free trial |

## Tips for Cloud Training

1. **Save checkpoints frequently** - Cloud sessions can disconnect
2. **Use smaller batch sizes** if GPU runs out of memory
3. **Enable mixed precision training** for faster training:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```
4. **Monitor GPU usage**: `!nvidia-smi`
5. **Download models immediately** after training
6. **Use wandb or tensorboard** to track experiments

## Troubleshooting

### Out of Memory Error:
- Reduce `--batch_size`
- Increase `--bin_factor`
- Reduce `--max_samples_per_class`

### Session Timeout:
- Save checkpoints every N epochs
- Use Kaggle instead of Colab (longer sessions)
- Consider paid platforms for uninterrupted training

### Slow Data Loading:
- Upload data to cloud storage first
- Use `--num_workers 2` (not too high on cloud)
- Cache data in memory if enough RAM

## Next Steps

Choose a platform from Step 3 and follow the setup instructions. Google Colab is recommended for getting started quickly!
