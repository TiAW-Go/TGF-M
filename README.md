<h1 style="border-bottom: 2px solid lightgray;">TGF-M: Topology-Augmented Geometric Features Enhance Molecular Property Prediction</h1>

__Abstract:__ Accurate prediction of molecular properties is a key component of Artificial Intelligence-driven Drug Design (AIDD). Despite significant progress in improving these predictive models, balancing accuracy with computational complexity remains a challenge. 
Molecular topological and geometric features provide rich spatial information, crucial for improving prediction accuracy, but their extraction typically increases model complexity. To address this, we propose TGF-M (Topology-augmented Geometric Features for Molecular Property Prediction), a novel predictive model that optimizes feature extraction to enhance information capture and improve model accuracy, and reduces model complexity to lower computational cost. This approach enhances the model’s ability to leverage both topological and geometric features without unnecessary complexity.  On the re-segmented PCQM4Mv2 dataset, TGF-M performs remarkably, achieving a low mean absolute error (MAE) of 0.0647 in the HOMO-LUMO gap prediction task with only 6.4M parameters. Compared to two recent state-of-the-art models evaluated within a unified validation framework, TGF-M demonstrates comparable performance with less than one-tenth of the parameters. We conducted an in-depth analysis of TGF-M's chemical interpretability. The results further validate the method’s effectiveness in leveraging complex molecular topology and geometry during model learning, underscoring its potential and advantages.  

![](overall.png)

## Getting Started

### Installation

Set up conda environment and clone the github repo

```
cd ./TGF-M
conda env create -f requirement.yaml
conda activate TGF-M
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch_geometric==1.6.3
pip install torch_scatter==2.0.7
pip install torch_sparse==0.6.9
pip install azureml-defaults
pip install rdkit-pypi cython
python setup.py build_ext --inplace
python setup_cython.py build_ext --inplace
pip install -e .
pip install --upgrade protobuf==3.20.1
pip install --upgrade tensorboard==2.9.1
pip install --upgrade tensorboardX==2.5.1
```


### Training

To train the TGF-M
```
$ cd TGF-M
$ python train.py
```





