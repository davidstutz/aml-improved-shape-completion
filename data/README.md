# Shape Completion Benchmark

This directory contains the data for

    @article{Stutz2018ARXIV,
        author    = {David Stutz and Andreas Geiger},
        title     = {Learning 3D Shape Completion under Weak Supervision},
        journal   = {CoRR},
        volume    = {abs/1805.07290},
        year      = {2018},
        url       = {http://arxiv.org/abs/1805.07290},
    }

![Examples of the data.](screenshot.png?raw=true "Examples of the data.")

## Data

The data is derived from [ShapeNet](https://www.shapenet.org/terms),
[KITTI](http://www.cvlibs.net/datasets/kitti/) and
[ModelNet](http://modelnet.cs.princeton.edu/). For ShapeNet,
two datasets, a clean dataset and a noisy dataset, were created.
On ModelNet, individual datasets for bathtubs, chairs, tables and desks
where created as well as a datset for ModelNet10. In each case
several resolutions may be available and are offered as download
separately. We also provide the watertight (not simplified) as well as
the simplified models.

Download links:

* SN-clean (2.0GB): [Amazon AWS](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_clean.zip) [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/ijcv2018-shape-completion/ijcv2018_shape_completion_clean.zip)
* SN-noisy (1.5GB): [Amazon AWS](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_noisy.zip) [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/ijcv2018-shape-completion/ijcv2018_shape_completion_noisy.zip)
* KITTI (5.6GB): [Amazon AWS](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_kitti.zip) [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/ijcv2018-shape-completion/ijcv2018_shape_completion_kitti.zip)
* bathtubs (3.1GB): [Amazon AWS](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_bathtub.zip) [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/ijcv2018-shape-completion/ijcv2018_shape_completion_bathtub.zip)
* chairs (3.8GB): [Amazon AWS](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_chair.zip) [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/ijcv2018-shape-completion/ijcv2018_shape_completion_chair.zip)
* desks (3.8GB): [Amazon AWS](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_desk.zip) [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/ijcv2018-shape-completion/ijcv2018_shape_completion_desk.zip)
* tables (3.6GB): [Amazon AWS](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_table.zip) [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/ijcv2018-shape-completion/ijcv2018_shape_completion_table.zip)
* ModelNet10 (6.1GB): [Amazon AWS](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_modelnet10.zip) [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/ijcv2018-shape-completion/ijcv2018_shape_completion_modelnet10.zip)
* ShapeNet Models (241.5MB): [Amazon AWS](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_shapenet_models.zip) [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/ijcv2018-shape-completion/ijcv2018_shape_completion_shapenet_models.zip)
* ModelNet10 Models (3.5GB): [Amazon AWS](https://s3.eu-central-1.amazonaws.com/avg-shape-completion/arxiv2018_shape_completion_modelnet10_models.zip) [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/ijcv2018-shape-completion/ijcv2018_shape_completion_modelnet10_models.zip)

**If these links do not work anymore, please let us know!**

Contained files, for example for resolution `24 x 54 x 24` and SN-clean:

| File | Description |
| --- | --- |
| test_off_gt_5_48x64_24x54x24_clean | Test meshes as OFF files, scaled to the corresponding resolutions. |
| test_txt_gt_5_48x64_24x54x24_clean | Test point clouds as TXT files, scaled. |
| training_inference_off_gt_5_48x64_24x54x24_clean | Training meshes as OFF files, scaled. |
| training_inference_txt_gt_5_48x64_24x54x24_clean | Training point clouds as TXT files, scaled. |
| training_prior_off_gt_5_48x64_24x54x24_clean | Meshes as OFF files for training a possible shape prior (if applicable), scaled. |
| test_filled_5_48x64_24x54x24_clean.h5 | Test shapes as occupancy grids, in a tensor of size `N x 1 x H x D x W`. |
| test_inputs_5_48x64_24x54x24_clean.h5 | Test point clouds as occupancy grids, same size. |
| test_inputs_ltsdf_5_48x64_24x54x24_clean.h5 | Test point clouds as log-truncated distance functions, same size. |
| test_ltsdf_5_48x64_24x54x24_clean.h5 | Test shapes as log-truncated signed distance functions, same size. |
| test_space_5_48x64_24x54x24_clean.h5 | Test free space as occupancy grids, same size. |
| training_inference_filled_5_48x64_24x54x24_clean.h5 | Training shapes as occupancy grids, same size. |
| training_inference_inputs_5_48x64_24x54x24_clean.h5 | Training point clouds as occupancy grids, same size. |
| training_inference_inputs_ltsdf_5_48x64_24x54x24_clean.h5 | Training point clouds as log-truncated distance functions, same size. |
| training_inference_ltsdf_5_48x64_24x54x24_clean.h5 | Training shapes as log-truncated signed distance functions, same size. |
| training_inference_space_5_48x64_24x54x24_clean.h5 | Training free space as occupancy grids, same size. |
| training_prior_filled_5_48x64_24x54x24_clean.h5 | Shapes as occupancy grids for training a possible shape prior, same size. |
| training_prior_ltsdf_5_48x64_24x54x24_clean.h5 | Shapes as log-truncated signed distance functions for training a possible shape prior, same size. |

The same files are provided for the ModelNet datasets.
For KITTI, we provide (for the same resolution):

| File | Description |
| --- | --- |
| bounding_boxes_txt_training_padding_corrected_1_24x54x24 | Training point clouds as TXT files, scaled to `[0,24] x [0,54] x [0,24]`. |
| bounding_boxes_txt_validation_gt_padding_corrected_1_24x54x24 | Validation point clouds as TXT files, same scale. |
| velodyne_gt_txt_training_padding_corrected_1_24x54x24 | Training ground truth point clouds as TXT files, same scale. |
| velodyne_gt_txt_validation_gt_padding_corrected_1_24x54x24 | Validation ground truth point clouds as TXT files, same scale. |
| bounding_boxes_training_padding_corrected_1_24x54x24.txt | List of training bounding boxes as TXT files, each bounding box being described by `(width_x, height_y, depth_z, translation_x, translation_y, translation_z, rotation_x, rotation_y, rotation_z)`, not scaled. | 
| bounding_boxes_validation_gt_padding_corrected_1_24x54x24.txt | List of validation bounding boxes as above. |
| input_training_padding_corrected_1_24x54x24_f.h5 | Training point clouds as occupancy grids, as `24 x 54 x 24` resulting in a `N x 1 x 24 x 54 x 24` tensor. |
| input_validation_gt_padding_corrected_1_24x54x24_f.h5 | Validation point clouds as occupancy grids, same size. |
| input_lsdf_training_padding_corrected_1_24x54x24_f.h5 | Training point clouds as log distance functions, same size. |
| input_lsdf_validation_gt_padding_corrected_1_24x54x24_f.h5 | Validation point clouds as log distance functions, same size. |
| part_space_training_padding_corrected_1_24x54x24_f.h5 | Training free space as occupancy grids, same size. |
| part_space_validation_gt_padding_corrected_1_24x54x24_f.h5 | Validation free space as occupancy grids, same size. |
| real_space_statistics_training_prior.h5 | Occupancy probabilities computed on the shape prior training set; indicates the probability of a voxel being occupied. |

Note that for higher resolutions, the point clouds may not be provided in TXT format.
However, these can easily be obtained by manually scaling the low resolution ones.

## Tools

Some tools for I/O can be found in `tools/`. Tools for visualization can be
found in [davidstutz/bpy-visualization-utils](https://github.com/bpy-visualization-utils).

The tools are mostly self-explanatory, but include:

* Python:
    * Conversion between [OFF](http://shape.cs.princeton.edu/benchmark/documentation/off_format.html)
      and [OBJ](http://paulbourke.net/dataformats/obj/) formats.
    * Conversion between our TXT point cloud format and [PLY](http://paulbourke.net/dataformats/ply/)
      format.
    * I/O code for KITTI's bounding boxes (note that the format is not the
      same as used for KITTI).
* LUA:
    * Example for HDF% I/O.
* C++:
    * Examples for reading OFF, HDF5 and TXT point clouds files.

## Date Generation

The code provided in `code/` is an untested version of the data generation
code.

The code should be self-explanatory, but may be tricky to understand in the beginning.
The C++ tools can be compiled with CMake, given that Boost, Eigen3, HDF5 and JSONCpp are installed.
For rendering [griegler/pyrender](https://github.com/griegler/pyrender) is required.
The easiest way to get started is to look into the configuration files outlining
the individual steps of the data pipeline and then look into the corresponding python scripts.

**Note that the data provided above only represents the final results of the
full pipeline.** You need to download the original datasets yourself.

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
