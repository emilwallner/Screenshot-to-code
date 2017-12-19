# Turning design mockups into code with deep learning

## Getting started

## Installation

### Floydhub
``` bash
pip install floyd-cli
floyd login
git clone https://github.com/emilwallner/Screenshot-to-code-in-Keras
cd Screenshot-to-code-in-Keras
floyd init projectname
floyd init projectname
floyd run --gpu --env tensorflow-1.4 --data emilwallner/datasets/s2c_val/1:data --mode jupyter
```
### Local
``` bash
pip install keras
pip install tensorflow
pip install pillow
pip install h5py
```

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
![Hello World](https://i.imgur.com/FVVnDeJ.gif "Hello World")
<img src="/README_images/Hello_world_model.png?raw=true" width="400px">


## HTML
<img src="/README_images/html.gif?raw=true" width="800px">
<img src="/README_images/HTML_model.png?raw=true" width="400px">


## Bootstrap
<img src="/README_images/bootstrap.gif?raw=true" width="800px">
<img src="/README_images/bootstrap_model?raw=true" width="400px">

## Acknowledgments
