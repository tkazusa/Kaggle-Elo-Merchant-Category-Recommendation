# Kaggle Elo Merchant Category Recommendation
A brief solution for [Kaggle Elo Merchant Category Recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation)

## Requirements


## Getting started
### Build docker image 
```
docker build -t <image name> .
docker run -it -p 8888:8888 --name <container name> <image name>
```

### Setup kaggle api credential
Download kaggle.json and place in the location: ~/.kaggle/kaggle.json.

See details: https://github.com/Kaggle/kaggle-api


### Download and unzip datasets from competition page
Data donwload from the kaggle competition page with kaggle api command.
```
mkdir $HOME/input
cd ./input
kaggle competitions download -c elo-merchant-category-recommendation
unzip '*.zip'
```

### Run jupyter lab
```
jupyter lab --ip 0.0.0.0 --allow-root
```

## What you learn from this kernel
TBD

## References
- https://www.kaggle.com/chauhuynh/my-first-kernel-3-699
- https://amalog.hateblo.jp/entry/elo-merchant-category-recommendation
- https://qiita.com/makotu1208/items/f10855aec2e67b4a44d1
