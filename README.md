
# Monet Style Transfer with CycleGAN

This project implements a CycleGAN to perform unpaired image-to-image translation, transforming regular photos into Monet-style paintings. The implementation is based on TensorFlow and utilizes a ResNet-based generator architecture.

## Dataset
The dataset used for training is provided by the Kaggle competition [GANs Getting Started](https://www.kaggle.com/competitions/gan-getting-started). It contains:
- `monet_jpg`: 300 Monet paintings.
- `photo_jpg`: 7028 photos.
Both datasets consist of 256x256 pixel images.

## Model
The CycleGAN model includes:
- Two generators: Photo-to-Monet and Monet-to-Photo.
- Two discriminators: Monet discriminator and Photo discriminator.

### Generator
The generator uses a ResNet-based architecture with configurable residual blocks. It includes:
- Downsampling layers.
- Residual blocks for feature extraction.
- Upsampling layers for image reconstruction.

### Discriminator
The discriminator uses convolutional layers to classify real vs. generated images.

## Training
The model is trained using:
- **Cycle Consistency Loss**: Ensures the translated image can be reconstructed back to its original domain.
- **Adversarial Loss**: Guides the generator to produce images indistinguishable from real images.
- **Identity Loss**: Preserves content by ensuring generated images resemble the original.

### Hyperparameter Tuning
Hyperparameters explored include:
- Learning rate
- Cycle loss weight
- Identity loss weight
- Number of residual blocks
- Batch size

## Results
The model generates 7,000 to 10,000 Monet-style images for submission, evaluated using Fréchet Inception Distance (FID).

## Usage
1. **Training the Model:**
   Run the provided Jupyter Notebook to train the CycleGAN model.

2. **Generating Monet-Style Images:**
   The trained generator can transform new photos into Monet-style paintings.

3. **Submission:**
   The output images are saved in a zip file (`images.zip`) for submission or further use.

## File Structure
- `generator_g.h5`: Trained Photo-to-Monet generator.
- `generator_f.h5`: Trained Monet-to-Photo generator.
- `images.zip`: Zip file containing generated Monet-style images.

## References
- Zhu, Jun-Yan, et al. *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.* ICCV 2017.
- [TensorFlow CycleGAN Tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan)
- [Kaggle GANs Getting Started Competition](https://www.kaggle.com/competitions/gan-getting-started)

## Requirements
- Python 3.8+
- TensorFlow 2.6+
- NumPy
- Matplotlib
- tqdm

## How to Run
1. Clone this repository.
2. Install the dependencies: `pip install -r requirements.txt`.
3. Train the model or use the pre-trained model to generate Monet-style images.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
