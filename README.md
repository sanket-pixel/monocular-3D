# Monocular 3D Detection with Geometric Constraints Embedding and Semi-supervised Training Embedding and Semi-Supervised Training

## Intuition
Object detection in 3D essentially concerns with finding all the instances of a set of classes in a given image, represented by a 3D bounding box. A 3D bounding box consists of 8 corners. In this work, they also consider the center of the bounding box, and refer to the projection of these 9 3D points onto the 2D image plane ( 8 corners + 1 centre ) as **keypoints**. Now, the goal is to predict these keypoints for every object in a given an RGB image from a single camera. 

Question : In how many ways can we make this happen? 

As it turns out, there are more than one ways to obtain these keypoints :
1. Let us first discuss the most obvious approach. Given a set of training images with keypoints (8 corners and 1 centre ) as labels, we simply train a network to predict all the 9 values. During inference, just with the given input image, the network gives out the 9 keypoints. 
2. A more involved approach is to use perspective geometry. In perspective geometry, if the 3D position ( of the centre of the 3D bounding box in global frame) , dimensions ( height, width, length) and orientation ( yaw, pitch, roll ) is given,  one can obtain the keypoints using projective transformations. But how do we obtain the dimension, orientation and position? The dimension and orientation can be predicted using CNN. And how do we find the position, given the predicted orientation and dimension.

Here is where the essensce of approach proposed by the paper comes into play. Since we have two different ways to obtain the same keypoints, we can minimize the difference between these two sets of keypoints by optimizing over the 3D position. The approach leverages the best parts from both CNN and perspective geometry in an intuitive and end-to-end differentiable ( hence trainable ) network to formulate a unified approach for 3D object detection from monocular images. 
