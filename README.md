# Skin Lesion Detection using Deep Learning

## Project Overview

This project aims to develop a robust application for detecting and classifying skin lesions using deep learning techniques. The model is trained to classify various types of skin lesions, leveraging a deep convolutional neural network architecture.

## Dataset

The dataset used for this project is the HAM10000 dataset, which contains 10,045 images of different skin lesions categorized into seven classes:
- **nv**: Melanocytic nevi
- **mel**: Melanoma
- **bkl**: Benign keratosis-like lesions
- **bcc**: Basal cell carcinoma
- **akiec**: Actinic keratoses
- **vasc**: Vascular lesions
- **df**: Dermatofibroma

## Installation

To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

Ensure you have the following libraries installed:
- `torch`
- `torchvision`
- `tqdm`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

## Usage

1. **Data Preparation:**
   - Place all images in the specified directory.
   - Update the `base_dir` variable to point to your image directory.
   - Ensure the metadata file `HAM10000_metadata.csv` is in the same directory.

   ```python
   base_dir = '/path/to/your/image/directory'
   all_image_path = glob(os.path.join(base_dir, '*.jpg'))

   if not all_image_path:
       raise FileNotFoundError("No images found in the specified directory.")
   else:
       print(f"Number of images found: {len(all_image_path)}")

   metadata_path = os.path.join(base_dir, 'HAM10000_metadata.csv')
   assert os.path.exists(metadata_path), "Metadata file not found!"

   df_original = pd.read_csv(metadata_path)
   df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
   df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
   df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
   ```

2. **Training the Model:**
   - The model uses the ResNeXt architecture for training.
   - Perform data augmentation and class balancing to enhance accuracy.

   ```python
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torch.autograd import Variable
   import torchvision
   from torchvision import transforms, models
   from torch.utils.data import DataLoader, Dataset
   ```

3. **Evaluation:**
   - Evaluate the model using metrics such as confusion matrix and classification report from `scikit-learn`.

   ```python
   from sklearn.metrics import confusion_matrix
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import classification_report
   ```

## Results

The trained model achieves high accuracy in classifying the different types of skin lesions. Detailed results and performance metrics will be provided after training.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

We thank the providers of the HAM10000 dataset and the open-source community for their invaluable resources and support.
