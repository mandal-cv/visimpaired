# visimpaired
This repository contains the dataset and models proposed in the paper - ***'Helping Visually Impaired People Take Better Quality Pictures'***. If this work helps in you research, please cite us - https://arxiv.org/abs/2305.08066

## Dataset
The 'Dataset' folder has two separate datasets -
1. **VizWiz** - The images for the Vizwiz dataset can be downloaded at: https://vizwiz.org/tasks-and-datasets/image-quality-issues/
  The dataset contains annotations for training, validation, and test sets. The csv contains annotations for quality and distortions (Blurry, Shaky, Bright, Dark, Grainy, None, and Other) for both the full image as well as the patch extracted from it. The coordinates for the extracted patch and the kind of patch extracted (Random or Salient) is also provided.
2. **ORBIT** - The images in ORBIT dataset has been sampled from ORBIT videos: https://orbit.city.ac.uk/
   We have sampled images from the dataset and labeled them similarly to VizWiz. The sampled images can be found at: https://utexas.box.com/s/hu1ktilsernkcbtf01uc7g94f9qr8y7r

## Models 
To test the models, refer to notebooks *base_model.ipynb* and *local_feedback_models.ipynb*. Some pre-trained models can be found at: https://utexas.box.com/s/gzfbybfm9elepccclaaldk4hjzr8jdg5

## Demo and Mobile App
A short demo of the mobile application built on our proposed models can be found at: https://vimeo.com/702709509
