# TutorialML-AtlasItalia2022
Material for the ML tutorial held during [ATLAS Italia 2022](https://agenda.infn.it/event/29726/).

## Notebooks
All the notebooks with their outputs can be found on [cernbox](https://cernbox.cern.ch/index.php/s/oiIGWYvFjC7QFYQ) or on swan. You can run them interactively with colab (see links below) or with swan (follow the link you have received, or go to https://swan.cern.ch/ (or https://swan-k8s.cern.ch/ if you have GPU access) -> Share -> Projects shared with me -> TutorialML-AtlasItalia2022). You can also run on your laptop.

[0.1-IntroML](notebooks/0.1-IntroML.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/main/notebooks/0.1-IntroML.ipynb)

[0.2-IntroKeras](notebooks/0.2-IntroKeras.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/main/notebooks/0.2-IntroKeras.ipynb)

[1.0-Image classification](notebooks/1.0-ImageClassification.ipynb): Images classification (fashion-mnist dataset) with a plain neural network [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/1.0-ImageClassification.ipynb)


[1.1-ImageClassification](notebooks/1.1-ImageClassification.ipynb): Images classification (fashion-mnist dataset) with a convolutional neural network [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/1.1-ImageClassification.ipynb)

[2.0-EnergyRegression](notebooks/2.0-EnergyRegression.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/2.0-EnergyRegression.ipynb)

[2.1-EnergyRegression](notebooks/2.1-EnergyRegression.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/2.1-EnergyRegression.ipynb)

[2.2-EnergyRegression](notebooks/2.2-EnergyRegression.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/2.2-EnergyRegression.ipynb)

[3.0-Autoencoder](notebooks/3.0-AutoEncoder.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/3.0-AutoEncoder.ipynb)

[3.1-Autoencoder_denoise](notebooks/3.1-AutoEncoder_denoise.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/3.1-AutoEncoder_denoise.ipynb)

[3.2-VariationalAutoEncoder](notebooks/3.2-VariationalAutoEncoder.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/3.2-VariationalAutoEncoder.ipynb)

[3.3-VariationalAutoEncoderConditional](notebooks/3.3-VariationalAutoEncoderConditional.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiso/TutorialML-AtlasItalia2022/blob/master/notebooks/3.3-VariationalAutoEncoderConditional.ipynb)


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
pip install requirements.txt
```

Open the first notebook:
```
jupyter notebook notebooks/0.1-IntroML.ipynb
```

