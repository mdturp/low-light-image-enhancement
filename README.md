# Low Light Image Enhancement

A simple VUE.js application that enhances low light images privately locally in the browser. 

<img width="482" alt="image" src="https://github.com/mdturp/low-light-image-enhancement/assets/26228055/00770ca9-71b0-4fc3-8d00-a1a9a61f5ad0">

This repository contains the code to convert the [Pytorch implementation](https://github.com/vis-opt-group/SCI/) of the Self-Calibrated Illumination (SCI) Learning Framework to the ONNX format and the code to run the model in the browser inside a Vue.js application.

If you want to know more about the processes and steps involved in how to build the vue.js application and how to run the model locally in the browser have a look at my [blog post](https://blog.mdturp.ch/posts/2024-01-24-low-light-image-enhancement.html).


## Installation and Usage

### Project Setup for the model conversion

#### Install python dependencies

```sh
cd model
pip install -r requirements.txt
```

#### Convert the model to ONNX

```sh
python transform_to_onnx_and_test.py
```


### Project Setup for the frontend

```sh
cd frontend
npm install
```

#### Compile and Hot-Reload for Development

```sh
npm run dev
```

#### Compile and Minify for Production

```sh
npm run build
```

#### Lint with [ESLint](https://eslint.org/)

```sh
npm run lint
```




https://github.com/mdturp/low-light-image-enhancement/assets/26228055/96b65056-aded-4d62-8857-ea9373d4574c



## Acknowledgements

The model code and weights are based on the [Pytorch implementation](https://github.com/vis-opt-group/SCI/) of the following [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.html).
