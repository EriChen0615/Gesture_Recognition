# Guide to generate image demo

## How to configure demo.py

- Test images should be put in a sub-folder of the `Testing_Demo_Data` folder, configure `TestImage_subfolder = YourSubfolderName`

  `Image_postfix = YourImagePostfix` (Postfix without '.')

- Model weights should be put at the path `Model/{Net}/` and should contain four files:

  e.g.

  - Model/PNet/PNet-500.meta
  - Model/PNet/PNet-500.index
  - Model/PNet/PNet-500.meta
  - Model/PNet/checkpoint

  Configure `model_path`

e.g.

```python
# The model path, should be the same in the checkpoint file
model_path = ['Model/PNet/PNet-500', 'Model/RNet/RNet-500', 'Model/ONet/ONet-116']

# The sub-folder in the folder Testing_Demo_Data
TestImage_subfolder = "Train"

# Test image postfix
Image_postfix = 'jpg'
```

## How to run demo.py

```bash
python3 demo.py test_mode
```

e.g.

```bash
python3 demo.py PNet
python3 demo.py RNet
python3 demo.py ONet
```
For other input argument, it will raise AssertionError.

## Results

The results will be saved in the MTCNN_demo folder, Depending on test_mode

- MTCNN_demo/PNet/{Your sub-folder name}/ResultImage
- MTCNN_demo/RNet/{Your sub-folder name}/ResultImage
- MTCNN_demo/ONet/{Your sub-folder name}/ResultImage

For RNet and ONet, in the ResultImage folder, a sub-folder`prediction` is generated to store image for each bounding box and its classification.