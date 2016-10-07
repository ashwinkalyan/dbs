# Diverse Beam Search

This code implements diverse beam search - an approximate inference algorithm that generates diverse decodings. This repository demos the method for image-captioning using [neuraltalk2][1]

## Requirements
You will need to install [torch](http://torch.ch/) and the packages 
- `nn`
- `nngraph`
- `image`
- `loadcaffe` 
- `hdf5` (optional, depending on how you want to input data)
You might want to install torch using [this](https://github.com/torch/distro) repository. It installs a bunch of the requirements. 
Additionally, if you are using a GPU you will need to install `cutorch` and `cunn`. If the image-captioning checkpoint was trained using `cudnn`, you will need to download `cudnn`. First, you will need to download it from NVIDIA's [website](https://developer.nvidia.com/cudnn) and add it to your `LD_LIBRARY_PATH`. 
Any of the checkpoints distributed by Andrej Karpathy along with the [neuraltalk2][1] repository can be used with this code. Additionally, you could also train your own model using [neuraltalk2][1] and use this code to sample diverse sentences. 

## Generating Diverse Sequences
After installing all the dependencies, you should be able to obtain diverse captions by:
```
$ th -model /path/to/model.t7 -num_images 1 -image_folder eval_images -gpuid -1
```
To run a beam search of size 10 with 5 diverse groups and a diversity strength of 0.5 on the same image you would do:
```
$ th -model /path/to/model.t7 -B 10 -M 5 -lambda 0.5 -num_images 1 -image_folder eval_images -gpuid -1
```

## References
[1]: https://github.com/karpathy/neuraltalk2

