# Restoration of images of old handwritten documents

## Dependency
* Pytorch
* PIL/Pillow
* Numpy
* Scikit-learn (for GMM)

## Run
`method1.py` and `method2.py` contains the main code. The main difference between the two methods is how the background is restored. In `method1.py` the background is restored using GMM, whereas in `method2.py` the background is restored using a trained CNN. In both the method same network is utilized for extracting the text from the images.

Note: The inference_batch_size parameter may be changed based on available RAM in the system. 

### Syntax
```
python <code.py> input_image [output_location]"
```

## Files
```
.
├── data_gen.py					# data generator for training
├── data					# Sample input data
│   ├── 16.jpg
│   └── 21.jpg
├── gen_bg_gmm.py				# generate backgrounds for training
├── lib.py					# Some helper functions
├── method1.py					# main method: the background is done using GMM 
├── method2.py					# main method2: the background is done using CNN
├── model
│   ├── bg_model.pt				# weights og background restoration model
│   └── upto2017_model_ourdata.pt		# weights of text extractor model
├── model.py					# model architectures
├── out						# sample output
│   ├── 16_out_cnn_bg.jpg			# method2
│   ├── 16_out_gmm_bg.jpg			# method1
│   ├── 21_out_cnn_bg.jpg
│   └── 21_out_gmm_bg.jpg
├── pytorch_ssim				# pytorch implementation of SSIM[1]: utilized for training
│   └── __init__.py
├── README.md
├── test_keras_wt_load.py			# helper code to convert the keras weights to pytorch
├── train_bgnet.py				# trains background restorer CNN
└── trainTextExtractorNet.py			# trains text extractor CNN
```


## Publications
1. Mayank Wadhwani, Debapriya Kundu, and Bhabatosh Chanda. "Text Extraction and Restoration of Old Handwritten Documents." 2nd Workshop on Digital Heritage (WDH '18), Satellite workshop of ICVGIP 2018
2. Mayank Wadhwani, Debapriya Kundu, Deepayan Chakraborty, and Bhabatosh Chanda. "Text Extraction and Restoration of Old Handwritten Documents." arXiv preprint arXiv:2001.08742 (2020).

## Reference
[1] https://github.com/Po-Hsun-Su/pytorch-ssim