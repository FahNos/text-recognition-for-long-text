# SVTRv2_1 for long text

This repository was build on [OpenOCR](https://github.com/Topdu/OpenOCR/tree/main) frame work

I had the change in encoder part and use VNese dataset to training

and the results as below:

## Results
I was implement test follow dataset of OpenOCR, and result as below:

|  Model   |  LTB  | IC13<br/>857 | SVT  | IIIT5k<br/>3000 | IC15<br/>1811 | SVTP | CUTE80 |  Avg  |                             Config&Model&Log                              |
| :------: | :----------------------------------------------------------: | :----------: | :--: | :-------------: | :-----------: | :--: | :----: | :---: | :-----------------------------------------------------------------------: |
| SVTRv2-T | 47.83 |     98.6     | 96.6 |      98.0       |     88.4      | 90.5 |  96.5  | 94.78 | [Google drive](https://drive.google.com/drive/folders/12ZUGkCS7tEhFhWa2RKKtyB0tPjhH4d9s?usp=drive_link) |
| SVTRv2-S | 47.57 |     99.0     | 98.3 |      98.5       |     89.5      | 92.9 |  98.6  | 96.13 | [Google drive](https://drive.google.com/drive/folders/1mOG3EUAOsmD16B-VIelVDYf_O64q0G3M?usp=drive_link) |
| SVTRv2-B | 50.23 |     99.2     | 98.0 |      98.7       |     91.1      | 93.5 |  99.0  | 96.57 | [Google drive](https://drive.google.com/drive/folders/11u11ptDzQ4BF9RRsOYdZnXl6ell2h4jN?usp=drive_link) |
| SVTRv2-1 | **57.67** |     97.5     | 96.4 |      98.5       |     88.8      | 90.9 |  95.8  | 94.7  | [Hugging face](https://huggingface.co/FahNos/text_regconition_svtr2_1_long_text/tree/main/svtrv2_smtr_gtc_rctc_small) |

- The results show that model SVTR2-1 have best results for long text prediction, even better than [SMTR](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/smtr) model, which have reach 51.0 for long text prediction

### Dataset
- I use the dataset with the combinition between Union14M-L  and VNese dataset, and the dataset distribution of ratio between width / heigh as table below:

| **Ratio  **    | 1       | 2         | 3      | 4      | 5      | 6      | 7      | 8     | 9     | 10    | 11    | 12    | 13    | 14    | 15    | 16    | 17    | 18    | 19    | 20    | 21    | 22    | 23   | 24   | 25    |
|------------|---------|-----------|--------|--------|--------|--------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|------|------|-------|
| **Qty **       | 180287  | 1102443   | 616312 | 388852 | 281372 | 183585 | 155346 | 55600 | 41072 | 35863 | 16112 | 30108 | 28328 | 26244 | 22112 | 21253 | 18782 | 17818 | 14994 | 15683 | 11812 | 10361 | 3704 | 7885 | 53701 |
| **Percentage** | 27.74%  | 26.02%    | 15.70% | 9.37%  | 6.92%  | 2.93%  | 2.07%  | 1.34% | 1.07% | 1.01% | 0.92% | 0.84% | 0.72% | 0.67% | 0.58% | 0.56% | 0.48% | 0.44% | 0.38% | 0.35% | 0.30% | 0.26% | 0.12% | 0.19% | 1.32% |

### Usage
- You can run this model follow this [demo](https://github.com/FahNos/svtr2_1_long_text/blob/main/demo_long_text.ipynb)
