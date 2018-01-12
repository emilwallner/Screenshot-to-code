# Turning design mockups into code with deep learning
[![Cloud GPU](https://img.shields.io/badge/Run%20experiment%20on-FloydHub-blue.svg)](https://www.floydhub.com/emilwallner/projects/picturetocode)
[![MIT](https://img.shields.io/cocoapods/l/AFNetworking.svg)](https://github.com/emilwallner/Screenshot-to-code-in-Keras/blob/master/LICENSE)

This is the code for the article ['Turning design mockups into code with deep learning'](https://blog.floydhub.com/Turning-design-mockups-into-code-with-deep-learning/) on FloydHub's blog. 

Within three years deep learning will change front-end development. It will increase prototyping speed and lower the barrier for building software.

The field took off last year when Tony Beltramelli introduced the [pix2code paper](https://arxiv.org/abs/1705.07962) and Airbnb launched [sketching interfaces](https://airbnb.design/sketching-interfaces/). 

Currently, the largest barrier to automating front-end development is computing power. However, we can use current deep learning algorithms, along with synthesized training data, to start exploring artificial front-end automation right now.

In the provided models, we’ll teach a neural network how to code a basic a HTML and CSS website based on a picture of a design mockup.

We’ll build the neural network in three iterations. Starting with a Hello World version, followed by the main neural network layers, and ending by training it to generalize. 

A quick overview of the process: 

### 1) Give a design image to the trained neural network

![Insert image](https://i.imgur.com/LDmoLLV.png)

### 2) The neural network converts the image into HTML markup 

<img src="/README_images/html_display.gif?raw=true" width="800px">

### 3) Rendered output

![Screenshot](https://i.imgur.com/tEAfyZ8.png)


## Installation

### FloydHub
FloydHub is hands down the best option to run models on cloud GPUs: [floydhub.com](https://www.floydhub.com/)
``` bash
pip install floyd-cli
floyd login
git clone https://github.com/emilwallner/Screenshot-to-code-in-Keras
cd Screenshot-to-code-in-Keras
floyd init projectname
floyd run --gpu --env tensorflow-1.4 --data emilwallner/datasets/imagetocode/1:data --mode jupyter
```
### Local
``` bash
pip install keras
pip install tensorflow
pip install pillow
pip install h5py
pip install jupyter
```
```
git clone https://github.com/emilwallner/Screenshot-to-code-in-Keras
cd Screenshot-to-code-in-Keras/local
jupyter notebook
```
Go do the desired notebook, files that end with '.ipynb'. To run the model, go to the menu then click on Cell > Run all

The final version, the Bootstrap version, is prepared with a small set to test run the model. If you want to try it with all the data, you need to download the data here: https://www.floydhub.com/emilwallner/datasets/imagetocode, and specify the correct ```dir_name```.

## Folder structure

``` bash
  |-floydhub                               #Folder to run the project on Floyhub
  |  |-Bootstrap                           #The Bootstrap version
  |  |  |-compiler                         #A compiler to turn the tokens to HTML/CSS (by pix2code)
  |  |-Hello_world                         #The Hello World version
  |  |-HTML                                #The HTML version
  |  |  |-resources									
  |  |  |  |-Resources_for_index_file      #CSS and images to test index.html file
  |  |  |  |-html                          #HTML files to train it on
  |  |  |  |-images                        #Screenshots for training
  |-local                                  #Local setup
  |  |-Bootstrap                           #The Bootstrap version
  |  |  |-compiler                         #A compiler to turn the tokens to HTML/CSS (by pix2code)
  |  |  |-resources											
  |  |  |  |-eval_light                    #10 test images and markup
  |  |-Hello_world                         #The Hello World version
  |  |-HTML                                #The HTML version
  |  |  |-Resources_for_index_file         #CSS,images and scripts to test index.html file
  |  |  |-html                             #HTML files to train it on
  |  |  |-images                           #Screenshots for training
  |-readme_images                          #Images for the readme page
```


## Hello World
<p align="center"><img src="/README_images/Hello_world_model.png?raw=true" width="400px"></p>


## HTML
<p align="center"><img src="/README_images/HTML_model.png?raw=true" width="400px"></p>


## Bootstrap
<p align="center"><img src="/README_images/Bootstrap_model.png?raw=true" width="400px"></p>

## Model weights
- [Bootstrap](https://www.floydhub.com/emilwallner/datasets/imagetocode)
- [HTML](https://www.floydhub.com/emilwallner/datasets/html_models)

## Acknowledgments
- The code is largly influenced by Tony Beltramelli's pix2code paper. [Code](https://github.com/tonybeltramelli/pix2code) [Paper](https://arxiv.org/abs/1705.07962)
- The structure and some of the functions are from Jason Brownlee's [excellent tutorial](https://machinelearningmastery.com/develop-a-caption-generation-model-in-keras/)
