# A Semi-Supervised Approach to Clustering using SimSiam 

This work is a component of the Yale Social Robotics Lab robot recycling project led by PhD student Debsmita Ghose: [Project Page](https://sites.google.com/view/corl22-contrastive-recycling/home), [Public Project Repo](https://github.com/ScazLab/HumanSupContrastiveClustering), [Paper](https://drive.google.com/file/d/1LOa3ugXvbT_Gd0myeg8thzOMIRyFgBIr/view).

Our problem is framed as a recycling robot learning to remove correct recyclables from a recycling stream based on a few examples provided by a human. The model in this repo leverages the contrastive learning framework [SimSiam](https://github.com/facebookresearch/simsiam), which learns meaningful representations of images from unlabeled datasets. Using SimSiam on its own does not take full advantage of the small set of human examples during training. Therefore, we add a human supervised head to pull the human-selected images close together in the latent space, creating a representation that is tailored to the human's needs. The final project opted for the [Contrastive Clustering](https://arxiv.org/abs/2009.09687) learning framework rather than SimSiam since it provided better results.

## Training and Evaluating the Model

The model was implemented in PyTorch. The train_runner and avg_eval_runner scripts are provided for training and evaluating the model on new data. Instructions can be found within the scripts. The model options and data paths can be set in the file config/config.yaml. For usage on new data, the CropsDataSet in data/crops.py class should be modified to load in data from the desired dataset. 

