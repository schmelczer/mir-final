# MIR final assignment

## Install

```sh
python3 -m pip install -r requirements.txt
```

## Run

- [histogram.ipynb](histogram.ipynb)
  > Shows the distribution of the training data
- [compile_data.ipynb](compile_data.ipynb)
  > Preprocesses the dataset and generates lists of [Captioned images](captioned_image.py)
- [embed.py](embed.py)
  > Finetunes VGG, ResNet, Inception V3, BERT, and creates the dictionary for tf-idf
  > Calculates the embeddings of each caption and image
- [match.ipnyb](match.ipnyb)
  > Learns the mapping between the embeddings, shows visualisations, and evaluates the models

## Dataset sources

### Birds

- [CUB-200](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images?resource=download)
- [Birds](https://drive.google.com/file/d/1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ/view)

### Flowers

- [Oxford flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [Flowers](https://drive.google.com/uc?export=download&confirm=l7Ld&id=0B0ywwgffWnLLcms2WWJQRFNSWXM)
- [Flower names](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset)
