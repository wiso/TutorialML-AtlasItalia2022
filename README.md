# TutorialML-AtlasItalia2022
Material for the ML tutorial held during [ATLAS Italia 2022](https://agenda.infn.it/event/29726/).

## Notebooks
You can follow the tutorial in several way. Some way are interactive (you can run the code), some are static. On github and on colab you won't find the precomputed output.

   * on [cernbox](https://cernbox.cern.ch/index.php/s/Rs3cZOmooVbwO03) (static)
   * using [Swan](https://swan.cern.ch/) (interactive) (or https://swan-k8s.cern.ch/ if you have GPU access) -> Share -> Projects shared with me -> TutorialML-AtlasItalia2022).) You will find the project only if you are registered to this tutorial (and I haven't forget you)
   * using colab: see links below (interactive)
   * on github: see the lins below (static)
   * on your laptop (instruction below)


### [0.1-IntroML](notebooks/0.1-IntroML.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/main/notebooks/0.1-IntroML.ipynb)
Just some fun material about the present (2022) status of ML around the world

### [0.2-IntroKeras](notebooks/0.2-IntroKeras.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/main/notebooks/0.2-IntroKeras.ipynb)
A super quick introduction to neural networks, tensorflow and keras

### [1.0-Image classification](notebooks/1.0-ImageClassification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/1.0-ImageClassification.ipynb)
Images classification (fashion-mnist dataset) with a plain neural network

### [1.1-ImageClassification](notebooks/1.1-ImageClassification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/1.1-ImageClassification.ipynb)
An improved version of the previous example using a convolutional neural network

### [2.0-EnergyRegression](notebooks/2.0-EnergyRegression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/2.0-EnergyRegression.ipynb)
Build an energy calibration for electron in ATLAS using a neural network: introduction to the problem and the dataset

### [2.1-EnergyRegression](notebooks/2.1-EnergyRegression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/2.1-EnergyRegression.ipynb)
The real neural network for the energy calibration

### [2.2-EnergyRegression](notebooks/2.2-EnergyRegression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/2.2-EnergyRegression.ipynb)
Some more complicated: build a regression of the distribution of the energy response of the detector to electrons

### [3.0-Autoencoder](notebooks/3.0-AutoEncoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/3.0-AutoEncoder.ipynb)
Use an autoencoder to generate fashion images

### [3.1-Autoencoder_denoise](notebooks/3.1-AutoEncoder_denoise.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/3.1-AutoEncoder_denoise.ipynb)
Use an autoencoder to remove noise from fashion images

### [3.2-VariationalAutoEncoder](notebooks/3.2-VariationalAutoEncoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/3.2-VariationalAutoEncoder.ipynb)
Improve the generation of fashion images with a variational autoencoder


### [3.3-VariationalAutoEncoderConditional](notebooks/3.3-VariationalAutoEncoderConditional.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/3.3-VariationalAutoEncoderConditional.ipynb)
And make it conditional: generate your favourite fashion item

### [4.0-MLReweighting](notebooks/4.0-MLReweighting.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/4.0-MLReweighting.ipynb)
Reweight the MC to match the data using a NN correcting for several variables together.

### [5.0-Generative adversarial network](notebooks/5.0-GAN.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/5.0-GAN.ipynb)


## Run on your laptop
Download the repository:

```
git clone git@github.com:wiso/TutorialML-AtlasItalia2022.git
```

you need a recent version of python (tested with python 3.10.4) and the possibility to install packages (here using a `virtualenv`, you can use [conda](https://docs.conda.io/en/latest/miniconda.html) instead)

```
cd TutorialML-AtlasItalia2022/
python -m virtualenv myenv --system-site-packages  # the last option if you have ROOT already installed
source myenv/bin/activate
pip install -r requirements.txt
```

Open the first notebook:
```
cd notebooks
jupyter notebook 0.1-IntroML.ipynb
```

