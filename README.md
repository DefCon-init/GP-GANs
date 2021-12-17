# Introduction

The aim of this project to implement the research paper on [GP-GAN: Towards Realistic High-Resolution Image Blending.](https://arxiv.org/pdf/1703.07195.pdf) The paper  uses the concept of Gaussian Poisson Blending to attempt to solve the problem of high resolution image blending. Given a source,destination and a mask image, our goal is to generate a realistic composite belnded image. We have used TensorFlow to implement this paper.

The notebook is a google colab file which contains the steps to run the python files and the link to get download the dataset.

# Requirements

 - tensorflow
 - scikit-image
 - opencv-python

# Table Of Contents

-  [Implementation Flow](#implementation-flow)
-  [Graphs and Images](#graphs-and-Images)
-  [Pitfalls and Challenges](#pitfalls-and-challenges)
-  [Future Work](#future-work)

# Implementation Flow

Here is how you can use this algorithm to create your own blended images from a source and destination image

In `crop_aligned_images.py` file, we have created a function which crops the images so that the source destination and masked images have the same size. The input to this file is our aligned image dataset. The cropped files will be saved in the cropped_images folder

```python
  
  for data_root in images:
          mask = imread(data_root)
          cropped_mask = mask[sx:ex, sy:ey]

          mask_name = os.path.basename(data_root)
          imsave(os.path.join(args.result_folder, name, mask_name), cropped_mask)
```

In the `write_tf_records.py` file, the task of image blending is performed by a copy-paste function. It takes source image and destination images from the cropped_image folder and gives the output in a serialized format.

```python
def create_copy_pastes(root, folders, max_images, load_size, crop_size, crop_size_ratio, writer):
    """
    This function creates copy paste images from the object and background cropped images
    """
        obj = cv2.cvtColor(cv2.imread(obj_path), cv2.COLOR_BGR2RGB)  # source
        bg = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)  # background
        w, h, _ = obj.shape
        min_size = min(w, h)
        ratio = load_size / min_size
        rw, rh = int(np.ceil(w * ratio)), int(np.ceil(h * ratio))
        sx, sy = np.random.randint(0, rw - crop_size+1), np.random.randint(0, rh - crop_size+1)

        obj_croped = _crop(obj, rw, rh, sx, sy, crop_size)
        bg_croped = _crop(bg, rw, rh, sx, sy, crop_size)

        copy_paste = bg_croped.copy()
        copy_paste[sx_cp:sx_cp + size, sx_cp:sx_cp + size, :] = obj_croped[ sx_cp:sx_cp + size,
                                                                            sx_cp:sx_cp + size, :]

        tf_example = create_tf_example(copy_paste, bg_croped)
        writer.write(tf_example.SerializeToString())
```

In `train_blending_gan.py` file, we have defined function to initialize a blending GAN and set training parameters. In order to increase the accuracy of the GAN, the training parameters could be changed to increase the number of iterations and snapshot_interval. After running this file, checkpoints are created which is the weight of the model 

```python
'''
Code to initialize generator
'''
generator = EncoderDecoder(encoder_filters=args.nef, encoded_dims=args.nBottleneck, output_channels=args.nc,
                               decoder_filters=args.ngf, is_training=True, image_size=args.image_size, skip=False, scope_name='generator') #, conv_init=init_conv,

generator_val = EncoderDecoder(encoder_filters=args.nef, encoded_dims=args.nBottleneck, output_channels=args.nc,
                               decoder_filters=args.ngf, is_training=False, image_size=args.image_size, skip=False, scope_name='generator')

```
```python
'''
Code to initialize discriminator
'''
discriminator = DCGAN_D(image_size=args.image_size, encoded_dims=1, filters=args.ndf, is_training=True, scope_name='discriminator') #, conv_init=init_conv, bn_init=init_bn)  # D

discriminator_val = DCGAN_D(image_size=args.image_size, encoded_dims=1, filters=args.ndf, is_training=False,
                           scope_name='discriminator')

```
In the `run_gp_gan.py`file, we take the source image, destination image, masked image and the weights from the trained model, to perform blending using the GP-GAN. The result image will be stored as a result.jpg

```python

from gp_gan import gp_gan
blended_im = gp_gan(obj, bg, mask, gan_im_tens, inputdata, sess, args.image_size, color_weight=args.color_weight,
                            sigma=args.sigma,
                            gradient_kernel=args.gradient_kernel, smooth_sigma=args.smooth_sigma)
```
# Graphs and Images

 Tensorflow graphs
 
 <img src="https://github.com/DefCon-init/GP-GANs/blob/master/DataBase/example_results/merge_from_ofoct.jpg" width="1500" height="1500">
 
 Under images you will find image samples from the training process: 
 
 | <img src="https://github.com/DefCon-init/GP-GANs/blob/master/DataBase/example_results/composed_image_1.PNG" width="250" height="250"> | <img src="https://github.com/DefCon-init/GP-GANs/blob/master/DataBase/example_results/composed_image_2.PNG" width="250" height="250"> | <img src="https://github.com/DefCon-init/GP-GANs/blob/master/DataBase/example_results/real_image_1.PNG" width="250" height="250"> | <img src="https://github.com/DefCon-init/GP-GANs/blob/master/DataBase/example_results/real_image_2.PNG" width="250" height="250">

# Pitfalls and Challenges

We encountered compatibility issues while running the code due to mismatch between the tensorflow versions. Due to hardware constraints, this version of the code does not create images with very good resolution. This algorithm failes to generate realistic image when the composite images are far from the distribution of the training dataset

# Future Work

Inorder to create blended images with greater resolution, the code can be tweaked to increase the number of steps and cycles to a greater value.
Currently the scope of this implementation is limited to using supervised learning. As a future scope, this can be implemented using unsupervised learning techniques. 
   
