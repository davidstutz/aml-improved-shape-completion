# Weakly Supervised Shape Completion

This repository contains the code for the weakly-supervised shape completion
method, called amortized maximum likelihood (AML), described in:

    @article{Stutz2018ARXIV,
        author    = {David Stutz and Andreas Geiger},
        title     = {Learning 3D Shape Completion under Weak Supervision},
        journal   = {CoRR},
        volume    = {abs/1805.07290},
        year      = {2018},
        url       = {http://arxiv.org/abs/1805.07290},
    }

If you use this code for your research, please cite the paper.

**This work is an extension of [1] and
[davidstutz/daml-shape-completion](https://github.com/davidstutz/daml-shape-completion).**
The extension improves visual quality of the completed shapes and
increases their variety. Additionally, the experiments are based on
improved benchmarks, see [Data](#data).

    [1] David Stutz, Andreas Geiger.
        Learning 3D Shape Completion from Laser Scan Data with Weak Supervision.
        IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

![Illustration of the proposed approach.](screenshot.jpg?raw=true "Illustration of the proposed approach.")

## Overview

The paper proposes a weakly-supervised approach to shape completion.
In particular, a denoising variational auto-encoder (VAE) is
trained to learn a shape prior - on a set of synthetic shapes from
[ShapeNet](https://www.shapenet.org/) or [ModelNet](http://modelnet.cs.princeton.edu/).
The generative model (the decoder) is then fixed and a new recognition model
(encoder) is trained to embed observations in the same latent shape space.
The encoder can be trained in an unsupervised fashion
- as we know the object category, the approach can be described
as weakly-supervised. In particular, the encoder
predicts Gaussian distributions that match the prior on the latent space
(a unit Gaussian) and simultaneously minimizes the loss between generated
shape and observations. The overall approach can be described as
amortized maximum likelihood (AML), as the encoder is trained
to minimize a maximum likelihood loss. As shape representation, occupancy
grids and signed distance functions are used.

In this repository we provide our implementation of the
amortized maximum likelihood approach, two supervised baselines
(including [5]), and two data-driven baselines (including [6]):

    [5] Angela Dai, Charles Ruizhongtai Qi, Matthias Nießner.
        Shape Completion using 3D-Encoder-Predictor CNNs and Shape Synthesis.
        CoRR abs/1612.00101 (2016).
    [6] Francis Engelmann, Jörg Stückler, Bastian Leibe:
        Joint Object Pose Estimation and Shape Reconstruction in Urban Street Scenes Using 3D Shape Priors. GCPR 2016: 219-230

### Difference to CVPR'18 Approach

This paragraph briefly summarizes the difference of this code to
the code corresponding to [1] which can be found in
[davidstutz/daml-shape-completion](https://github.com/davidstutz/daml-shape-completion).
First of all, the code has been adapted to the improvsed datasets
as described in the paper and downloadable in [Data](#data).
This is mainly due to an improved data pipeline allowing to obtain higher-quality
watertight meshes through TSDF fusion (the corresponding implementation
can be found at [davidstutz/mesh-fusion](https://github.com/davidstutz/mesh-fusion)).
Additionally, datasets for higher resolutions are provided.
Regarding the proposed approach, we improve noise handling and additionally
try to enforce more variety and details in the completed shapes.
To this end, the encoder trained for shape inference does not predict
a deterministic maximum likelihood solution anymore, but predicts a
Gaussian distribution (similar to the VAE shape prior). In the loss,
the quadratic regularizer is then replaced by a Kullback-Leibler divergence
which ensures that the encoder predicts a variety of different codes
while still fitting the shapes to the observations. Finally,
more comprehensive experiments on ModelNet have been conducted and
additional baselines have been considered.

**Why do we provide both repositories separately?**

Our training procedure as well as the architectures and the training data
changed significantly. In order to provide reproducible results,
we provide the code and data separately, although the underlying
code base might have some overlap.

## Installation

LUA/Torch requirements:

* Torch ([torch/distro](https://github.com/torch/distro) recommended);
* [deepmind/torch-hdf5](https://github.com/deepmind/torch-hdf5);
* [harningt/luajson](https://github.com/harningt/luajson);
* [luafilesystem](http://keplerproject.github.io/luafilesystem);
* [clementfarabet/lua---nnx](https://github.com/clementfarabet/lua---nnx) (already included in [torch/distro](https://github.com/torch/distro));
* [nicholas-leonard/cunnx](https://github.com/nicholas-leonard/cunnx);
* [davidstutz/torch-volumetric-nnup](https://github.com/davidstutz/torch-volumetric-nnup) (see the installation instructions in the repository).

Installing [deepmind/torch-hdf5](https://github.com/deepmind/torch-hdf5)
might be tricky. After building orch-hdf5,

    git clone https://github.com/deepmind/torch-hdf5
    cd torch-hdf5
    luarocks make hdf5-0-0.rockspec
    cd ..

it might be necessary to adapt the configuration in case you installed
HDF5 locally. For example, when installing hdf5 locally in `DB_PATH`,
`torch/install/share/lua/5.1/hdf5/config.lua` might need
to be adapted as follows:

    require('os')
  
    db_path = os.getenv("DB_PATH")
    hdf5._config = {
        HDF5_INCLUDE_PATH = db_path .. "/hdf5/hdf5/include/",
        HDF5_LIBRARIES = db_path .. "/hdf5/hdf5/lib/libhdf5_cpp.so;" .. db_path .. "/hdf5/hdf5/lib/libhdf5.so;/usr/lib/x86_64-linux-gnu/libpthread.so;/usr/lib/x86_64-linux-gnu/libz.so;/usr/lib/x86_64-linux-gnu/libdl.so;/usr/lib/x86_64-linux-gnu/libm.so"
    }

Make sure that `nnx`, `cunnx` and the volumetric nearest neighbor
upsampling layer works by following the instructions in
[davidstutz/torch-volumetric-nnup](https://github.com/davidstutz/torch-volumetric-nnup).

The remaining packages can easily be installed using luarocks. You can run

    th check_requirements.lua

to check the packages listed above.

Pyton requirements:

* NumPy;
* h5py;
* [PyMCubes](https://github.com/davidstutz/PyMCubes) (**make sure to use the `voxel_center` branch**).

For installing PyMCubes, follow the instructions [here](https://github.com/davidstutz/PyMCubes);
NumPy and h5py can be installed using `pip` and might themselves
have dependencies.

We also include an implementation of the method by Engelmann et al. [1].

    [1] Francis Engelmann, Jörg Stückler, Bastian Leibe:
        Joint Object Pose Estimation and Shape Reconstruction in Urban Street Scenes Using 3D Shape Priors. GCPR 2016: 219-230

First, make sure to install:

* [OpenCV 2.4.13.x](https://opencv.org/);
* [VTK 7.1.x](https://www.vtk.org/download/);
* [Ceres](http://ceres-solver.org/installation.html):
  * Eigen3;
  * [GLog](https://github.com/google/glog);
  * [GFlags](https://github.com/gflags/gflags);
  * ([Suitesparse](http://faculty.cse.tamu.edu/davis/suitesparse.html));

The installation of Ceres might be a bit annoying; so we provide our
installation scripts for some of the dependencies in `rw/dependencies` for further details.
These still need to be adapted (e.g. they assume that dependencies as well
as Ceres are installed in `$WORK/dev-box` where `$WORK` is a defined base directory).
Note that SuiteSparse is optional but significantly reduces runtime
(by roughly factor 2-4).

When all dependencies are installed, make sure to adapt the corresponding
CMake files in `rw/cmake_modules`. This means removing `NO_CMAKE_SYSTEM_PATH`
if necessary and inserting the correct paths to the installations.

Then:

    cd rw/external/viz
    mkdir build
    cd build
    cmake ..
    make
    # make sure VIZ is built correctly
    cd ../../
    mkdir build
    cd build
    cmake ..
    make
    # make sure KittiShapePrior and ShapeNetShapePrior are built correctly

Also make sure to download the pre-trained PCA shape prior
from [VisualComputingInstitute/ShapePriors_GCPR16](https://github.com/VisualComputingInstitute/ShapePriors_GCPR16).

For the C++ implementation of the evaluation tool (mesh-to-mesh and point-to-mesh distances),
follow the instructions here: [davidstutz/mesh-evaluation](https://github.com/davidstutz/mesh-evaluation);
essentially, the tool requires:

* CMake;
* Boost;
* Eigen;
* OpenMP;
* C++11.

Make sure to adapt the corresponding CMake modules, then run:

    cd mesh-evaluation
    mkdir build
    cd build
    cmake ..
    make

For building the ICP baseline, Eigen3, HDF5, Boost and OpenMP are required.
It might be necessary to adapt (or even add) the corresponding CMake
modules in `icp/cmake/`. Then:

    cd icp/
    mkdir build
    cd build
    cmake ..
    make

## Data

The data is derived from [ShapeNet](https://www.shapenet.org/terms) [1],
[KITTI](http://www.cvlibs.net/datasets/kitti/) [2],
[ModelNet](http://modelnet.cs.princeton.edu/) [3]
and [Kinect](https://github.com/Yang7879/3D-RecGAN-extended) [4].
For ShapeNet, two datasets, in the paper referred to as SN-clean and SN-noisy,
were created. We also provide the raw, watertight models for ShapeNet
and ModelNet.

Download links:

* [SN-clean (2.0GB)](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_clean.zip)
* [SN-noisy (1.5GB)](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_noisy.zip)
* [KITTI (5.6GB)](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_kitti.zip)
* [bathtubs (3.1GB)](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_bathtub.zip)
* [chairs (3.8GB)](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_chair.zip)
* [desks (3.8GB)](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_desk.zip)
* [tables (3.6GB)](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_table.zip)
* [ModelNet10 (6.1GB)](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_modelnet10.zip)
* [ShapeNet Models (241.5MB)](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_shapenet_models.zip)
* [ModelNet10 (3.5GB)](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_modelnet10_models.zip)

**Note that SN-clean and SN-noisy are different from the corresponding
datasets in our CVPR'18 paper!**

**See [Data](data/README.md) for more details.**

Make sure to cite [3] and [4] in addition to this paper when using the data.

    [3] Andreas Geiger, Philip Lenz, Raquel Urtasun:
        Are we ready for autonomous driving? The KITTI vision benchmark suite. CVPR 2012: 3354-3361
    [4] Angel X. Chang, Thomas A. Funkhouser, Leonidas J. Guibas, Pat Hanrahan, Qi-Xing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, Fisher Yu:
        ShapeNet: An Information-Rich 3D Model Repository. CoRR abs/1512.03012 (2015)

## Models

Models can be downloaded here:

* [AML Models](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_models_aml.zip).
* [Dai et al. Models](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_models_dai.zip).
* [Sup (Supervised Baseline) Models](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_models_sup.zip).
* [DVAE (Shape Prior) Models](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_models_vae.zip).

The downloaded ZIP-archive contains models for the approach by Dai et al.,
our supervised baseline, our shape prior and the AML approach. The
directory structure is named according to the configuration files. For example,
check `vae/config/`.

The models have been saved using `data/tools/lua/compress_model_dat.lua`
in order to reduce their size.

For running a model, it is sufficient to extract the corresponding
`.dat` file and place it in the correct `base_directory` according
to the configuration file. For example, for `vae/config/clean.low`,
create the directory `vae/config/clean.low` and place the model file inside.
Then:

    th 4_run.lua config/clean.low.json

More details are given for the training procedures, see below.

## Experiments

Make sure that the data has been downloaded and
all requirements are met, for example run
`check_requirements.lua` and `check_requirements.py`.

_Note that not all cases have been tested and some configuration files
have been copy-and-pasted from our experiments to ensure reproducibility._

### Shape Prior

The shape prior is trained using `vae/1_train.lua`. However, `vae/0_generate_codes.py` should
be run first to generate a set of random codes - this was mainly used for
reproducible experiments. For example:

    python 0_generate_codes.py --code_size=10 --number=100

Then, the configuration files, for example `vae/config/clean.low.json` should
be adapted. In particular,  the `data_directory` key needs to be set according
to the downloaded data. Then:

    th 1_train.lua config/clean.low.json

By default, only view iterations are done for illustration. However, training
parameters can be adjusted in the corresponding configuration file. The
options include:

* `noise_level`: probability of Bernoulli noise and 2*standard deviation of Gaussian noise;
  if `0` is used, a standard variational auto-encoder (not denoising) will be trained.
* `centering`: if set to `true`, the data will be centered.
* `optimizer`: the optimizer to be used; ADAM is recommended.
* `learning_rate`: the initial learning rate.
* `momentum`: the initial momentum; for ADAM this does not matter.
* `weight_decay`: weight decay for training.
* `batch_size`: batch sized for training.
* `epochs`: number of epoch for training; one epoch includes `N/batch_size` iterations.
* `weight_initialization`: weight initialization method to use, see `lib/th/WeightInitialization.lua`.
* `decay_iterations`: in which steps to decay learning rate and momentum.
* `decay_learning_rate`: factor for learning rate decay.
* `decay_momentum`: factor for momentum decay.

Additional options are fixed in `vae/1_train.lua`.

After training, `vae/2_a_marching_cubes.py` can be used to convert the
SDFs predicted on the test set to triangular meshes. This tool requires
the PyMCubes installation from the `voxel_center` branch in
[davidstutz/PyMCubes](https://github.com/davidstutz/PyMCubes); options
are:

    usage: Read LTSDF file and run marching cubes. [-h] [--input INPUT]
                                               [--output OUTPUT]
    optional arguments:
      -h, --help       show this help message and exit
      --input INPUT    Input HDF5 file.
      --output OUTPUT  Output directory.

Similarly, `vae/2_b_test.py` and `vae/3_sanity_check.py` can be used
to evaluate and generate simple visualizations of the predictions.
Available options are:

    usage: Evaluate predicted occupancy grids and LTSDFs. [-h]
                                                      [--predictions PREDICTIONS]
                                                      [--targets_occ TARGETS_OCC]
                                                      [--targets_sdf TARGETS_SDF]
                                                      [--results_file RESULTS_FILE]
    optional arguments:
      -h, --help            show this help message and exit
      --predictions PREDICTIONS
                            Predictions HDF5 file.
      --targets_occ TARGETS_OCC
                            Ground truth occupancy grids as HDF5 file.
      --targets_sdf TARGETS_SDF
                            Ground truth LTSDF as HDF5 file.
      --results_file RESULTS_FILE
                            Results txt file.
    usage: Visualize predictions. [-h] [--predictions PREDICTIONS]
                              [--targets_occ TARGETS_OCC]
                              [--targets_sdf TARGETS_SDF] [--randoms RANDOMS]
                              [--directory DIRECTORY]
                              [--n_observations N_OBSERVATIONS]
    optional arguments:
      -h, --help            show this help message and exit
      --predictions PREDICTIONS
                            Predictions HDF5 file.
      --targets_occ TARGETS_OCC
                            Ground truth occupancy grids HDF5 file.
      --targets_sdf TARGETS_SDF
                            Ground truth SDF HDF5 file.
      --randoms RANDOMS     Random predictions HDF5 file.
      --directory DIRECTORY
                            Output directory.
      --n_observations N_OBSERVATIONS

### Shape Inference

Shape inference using AML requires a pre-trained shape model. Therefore,
a model (usually `prior_model.dat`) from the previous step is required.
This model needs to be copied manually into the corresponding
base directory. For training on the clean ShapeNet dataset (corresponding to
`aml/config/clean.low.json`), the correct path would be
`aml/clean.low/prior_model.dat`.

Afterwards, the shape inference model can be trained using

    th 1_a_train.lua config/clean.low.json

The training parameters in the configuration files are similar to
the ones described above, except for:

* `weights`: determines the relative weights applied to the
  occupancy point loss, the occupancy free space loss, the SDF point loss
  and the SDF free space loss in this order.
* `weighted`: determines whether (for noisy cases), the free space
  statistics in `training_statistics` should be used.
* `reinitialize_encoder`: whether the loaded encoder from the shape
  prior should be reinitialized or fine-tuned.

As above, the remaining Python tools are used for visualization and evaluation.
For KITTI, `aml/1_b_traing.lua` needs to be used as ground truth shapes
are not available to monitor training.

### Dai et al. Baseline

The supervised baseline by Dai et al. [5] is included in `dai/`.
Note that the model introduced in [5] needed to be adapted slightly for
higher resolutions as well as on ShapeNet and KITTI due to the non-cubic resolution
of `24 x 54 x 24`. The models can be found in `dai/0_model.lua`.

For training, the configuration files in `dai/config/` provide
the corresponding hyper parameters, which are very similar
to the parameters described above for the shape prior. Then,
training is started using

    th 1_train.lua config/clean.low.json

Evaluation is done using the remaining tools in `dai/` as also
illustrated above.

### Engelmann et al. Baseline

First, make sure that the work by Engelmann et al. [1] can be compiled
as outlined in [Installation](#installation).

Then, two command line tools are provided:

* `KittiShapePrior` for running the approach on KITTI; arguments are
  the input directory with point clouds as `.txt` files, a `.txt` file containing
  the correpsonding bounding boxes, and the output directory.
* `ShapeNetShapePrior` for running the approach on ShapeNet ("clean" and "noisy");
  arguments are the input directory containing the point clouds as `.txt` files,
  and the output directory.

Running the approach on ShapeNet might look as follows:

    ./ShapeNetShapePrior /path/to/test_off_gt_5_48x64_24x54x24_clean output_directory

Subsequently, `rw/tools/shapenet_marching_cubes.py` can be used to obtain
meshes from the predicted signed distance functions:

    python shapenet_marching_cubes.py output_directory off_directory

For KITTI, the approach is similar; however, in addition to the input points,
the bounding boxes are required:

    ./KittiShapePrior /path/to/bounding_boxes_txt_validation_gt_padding_corrected_1_24x54x24/ /path/to/bounding_boxes_validation_gt_padding_corrected_1_24x54x24.txt output_directory

Similarly, `rw/tools/kitti_marching_cubes.py` expects the bounding boxes
as second argument as well.

### ICP Baseline

The ICP baseline might be quite slow; therefore a simple example is inclided in
`icp/test`. The tool is split into sampling the reference meshes and then
performing point-to-point icp given a partial point cloud:

    # From within the icp/build directory:
    ../bin/sample ../test/off/ ../test/points.h5
    ../bin/icp ../test/txt/0.txt ../test/off/ ../test/points.h5 ../test/output/ ../test/out.log

So, the ICP baseline can be called for each point cloud in TXT format
individually.

## Visualization

The original meshes included in the data downloads
as well as the predicted meshes (of both the shape prior and shape inference)
can be visualized using [MeshLab](http://www.meshlab.net/).

Additional tools for visualization using [Blender](https://www.blender.org/)
are provided in
[davidstutz/bpy-visualization-utils](https://github.com/davidstutz/davidstutz/bpy-visualization-utils).

Evaluation can be done using the tool included in
[davidstutz/mesh-evaluation](https://github.com/davidstutz/mesh-evaluation).

## License

Note that the data is based on [ShapeNet](https://www.shapenet.org/terms) [1],
[KITTI](http://www.cvlibs.net/datasets/kitti/) [2],
[ModelNet](http://modelnet.cs.princeton.edu/) [3]
and [Kinect](https://github.com/Yang7879/3D-RecGAN-extended) [4].
Check the corresponding websites for licenses.
The derived benchmarks are licensed as
[CC BY-NC-SA 3.0](Attribution-https://creativecommons.org/licenses/by-nc-sa/3.0/).

The code includes snippets from the following repositories:

* [pyrender](https://github.com/griegler/pyrender)
* [pyfusion](https://github.com/griegler/pyfusion)
* [Tomas_Akenine-Möller](http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/)
* [Box-Ray Intersection](http://www.cs.utah.edu/~awilliam/box/)
* [Tomas Akenine-Möller Code](http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/)
* [griegler/pyrender](https://github.com/griegler/pyrender)
* [christopherbatty/SDFGen](https://github.com/christopherbatty/SDFGen)
* [High-Resolution Timer](http://www.songho.ca/misc/timer/timer.html)
* [Tronic/cmake-modules](https://github.com/Tronic/cmake-modules)
* [dimatura/binvox-rw-py](https://github.com/dimatura/binvox-rw-py)
* [alextsui05/blender-off-addon](https://github.com/alextsui05/blender-off-addon)

The remaining code is licensed as follows:

Copyright (c) 2018 David Stutz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.