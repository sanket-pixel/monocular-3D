# Literature Review

## 1. Monocular 3D Detection with Geometric Constraints Embedding and Semi-supervised Training Embedding and Semi-Supervised Training [[Link]](https://arxiv.org/abs/2009.00764)

Object detection in 3D essentially concerns with finding all the instances of a set of classes in a given image, represented by a 3D bounding box. A 3D bounding box consists of 8 corners. In this work, they also consider the center of the bounding box, and refer to the projection of these 9 3D points onto the 2D image plane ( 8 corners + 1 centre ) as **keypoints**. Now, the goal is to predict these keypoints for every object in a given an RGB image from a single camera. 

Question : In how many ways can we make this happen? 

As it turns out, there are more than one ways to obtain these keypoints :
1. Let us first discuss the most obvious approach. Given a set of training images with keypoints (8 corners and 1 centre ) as labels, we simply train a network to predict all the 9 values. During inference, just with the given input image, the network gives out the 9 keypoints. 
2. A more involved approach is to use perspective geometry. In perspective geometry, if the 3D position ( of the centre of the 3D bounding box in global frame) , dimensions ( height, width, length) and orientation ( yaw, pitch, roll ) is given,  one can obtain the keypoints using projective transformations. But how do we obtain the dimension, orientation and position? The dimension and orientation can be predicted using CNN. And how do we find the position, given the predicted orientation and dimension.

Here is where the essensce of approach proposed by the paper comes into play. Since we have two different ways to obtain the same keypoints, we can minimize the difference between these two sets of keypoints by optimizing over the 3D position. The approach leverages the best parts from both CNN and perspective geometry in an intuitive and end-to-end differentiable ( hence trainable ) network to formulate a unified approach for 3D object detection from monocular images. 


### Method
The architecture consists of two main modules. The first module is a fully connected CNN, which extracts features from the input image. 
These features are further used across multiple heads for predicting appearance-related properties of an object namely dimension, orientation and ordered list of 2D perspective keypoints. The second module is for geometric reasoning which applies point-to-point geometric constraints for 3D position prediction. Now let us understand at each component in some more detail :


### Detection Heads 
1. Main Centre 
  * Predicts the 2D bounding box center of every object. One point per instance.
  * The prediction is a heatmap around the centers, with one channel for every category. 
2. Keypoints 
  * Projections of 3D corners and centers onto the 2D image plane. ( 8 corners + 1 center = 9 points )
  * 18 dimensional because each of the 9 points ( x, y ) is 2 dimensional.
3. Dimensions
  * Height, Width and Length
  * Regression of residual values instead of absolute values
  * Scaled by constants H = 1.63, W = 1.53, L = 3.88



## 2.  MonoCon : Learning Auxiliary Monocular Contexts Helps Monocular 3D Object Detection

The method trains a common CNN based feature backbone for a primary task and a set of auxillary tasks. The primary task is trained to predict the 2D bounding box center, offset vector ( from 2D center to 3D center ), depth, shape and observation angle ( angle of object from camera ). During inference, the 2D bbox cente, along with depth is used to extract the 3D bbox center using Intrinsic Camera Matrix (K). Auxillary tasks are defined with the goal of providing monocular context to the model. In principal, the auxillary tasks are not used in inference, but are only used during training to assist the primary tasks in getting monocular context. Specifically, the auxillary tasks include prediction of projected keypoints, offset vector ( of 2D center to 9 keypoints ), 2D bounding box shape, 2D bounding box center quantization residual, projected keypoints quantization residual. Five losses are used 
























## Homeruns

## Notes
