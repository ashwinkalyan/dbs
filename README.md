# Diverse Beam Search
This code implements Diverse Beam Search (DBS) - a replacement for beam search that generates diverse sequences from sequence models like LSTMs. This repository lets you generate diverse image-captions for models trained using the popular [neuraltalk2][1] repository. A demo of our implementation on captioning is available at [dbs.cloudcv.org](http://dbs.cloudcv.org/)

![Alt Text](https://s22.postimg.org/hoor3ricx/db_cover_2_1.png)
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
After installing the dependencies, you should be able to obtain diverse captions by: 
```
$ th eval.lua -model /path/to/model.t7 -num_images 1 -image_folder eval_images -gpuid -1
```
To run a beam search of size 10 with 5 diverse groups and a diversity strength of 0.5 on the same image you would do: 
```
$ th eval.lua -model /path/to/model.t7 -B 10 -M 5 -lambda 0.5 -num_images 1 -image_folder eval_images -gpuid -1
```
The output of the code will be written to a `json` file that contains all the generated captions and their scores for each image.

## Using DBS for other tasks
The core of our method is in `dbs/beam_utils.lua`. It contains two functions that you will need to replicate:
- `beam_step` - Performs one expansion of the beams held at any given time. 
- `beam_search` - Modifies the log-probabilities of the sequences and calls `beam_step` at every time step. This handles both division of the beam budget into groups and augmenting scores with diversity.

## Li15
- Checkout branch li15. You can download the language model from [here](https://filebox.ece.vt.edu/~ram21/dbs/language_model.t7) 

You should be able to run this by:
```
sh run_msr_eval.sh
```

## 3rd party
- [neuraltalk2][1]

[1]: https://github.com/karpathy/neuraltalk2
