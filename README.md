<h1 style="border-bottom: 2px solid lightgray;">TGF-M: Topology-Augmented Geometric Features Enhance Molecular Property Prediction</h1>

__Abstract:__ Accurate prediction of molecular properties is a key component of Artificial Intelligence-driven Drug Design (AIDD). Despite significant progress in improving these predictive models, balancing accuracy with computational complexity remains a challenge. Molecular topological and geometric features encapsulate rich spatial information essential for enhancing prediction accuracy, but their extraction increases model complexity. Therefore, effectively leveraging these features is pivotal to overcoming this challenge. We propose a novel predictive model named TGF-M (Topology-augmented Geometric Features for Molecular Property Prediction), which significantly enhances the model’s ability to capture both topological and geometric features. On the re-segmented PCQM4Mv2 dataset, TGF-M performs remarkably, achieving a low mean absolute error (MAE) of 0.0647 in the HOMO-LUMO gap prediction task with only 6.4M parameters. Compared to two recent state-of-the-art models evaluated within a unified validation framework, TGF-M demonstrates comparable performance with less than one-tenth of the parameters. We conducted an in-depth analysis of TGF-M's chemical interpretability. The results further validate the method’s effectiveness in leveraging complex molecular topology and geometry during model learning, underscoring its potential and advantages. 

![](Fig.1.png)

## Getting Started

### Installation

Set up conda environment and clone the github repo

```
# create a new environment
$ conda create --name DIG-Mol python=3.7
$ conda activate TGF-M

# install requirements
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torch-geometric==1.6.3 torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
$ pip install PyYAML
$ conda install -c conda-forge rdkit=2020.09.1.0
$ conda install -c conda-forge tensorboard
$ conda install -c conda-forge nvidia-apex # optional

# clone the source code of DIG-Mol
$ git clone https://github.com/ZeXingZ/DIG-Mol.git
$ cd DIG-Mol
```

### Dataset

You can download the pre-training data and benchmarks used in the paper [here](https://drive.google.com/file/d/1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE/view?usp=sharing) and extract the zip file under `./data` folder. The data for pre-training can be found in `pubchem-10m-clean.txt`. All the databases for fine-tuning are saved in the folder under the benchmark name. You can also find the benchmarks from [MoleculeNet](https://moleculenet.org/).

### Data preprocessing
To convert SMILES strings into molecular graphs using RDKit, refer to the data processing code available in `dataset/dataset.py`.

RDKit link[https://github.com/rdkit/rdkit]

### Pre-training

To train the DIG-Mol, where the configurations and detailed explaination for each variable can be found in `config.yaml`
```
$ python DIG-Mol.py
```

### Fine-tuning 

To fine-tune the DIG-Mol pre-trained model on downstream molecular benchmarks, where the configurations and detailed explaination for each variable can be found in `config_finetune.yaml`
```
$ python DIG-Mol_finetune.py
```

### Pre-trained models

We also provide pre-trained DIGNN models, which can be found in `model.pth` and `model_50.pth` for different pretraining epoches respectively. 

