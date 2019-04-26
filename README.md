Caffe for MONet 
==================
IMPORTANT: Edit train.prototxt to use your selected dataset and 
make sure the correct parts of the network are enabled by setting/adding
loss weights and blob learning rates. 

NOTE: The training templates include augmentation, during which an affine 
transformation is applied to a crop from the input immages. For training we 
use different batch sizes for each resolution:  

FlyingChairs: 		448 x 320 (batch size 8)
ChairsSDHom:		448 x 320 (batch size 8)
FlyingThings3D:		768 x 384 (batch size 4) 



