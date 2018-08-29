# Latent Space Mapping for Generation of Object Elements with Corresponding Data Annotation

This is the code for the paper "Latent Space Mapping for Generation of Object Elements with Corresponding Data Annotation"
The BEGAN model is trained on a subset of CelebA database. The subset is extracted using the face detection code in
https://github.com/Heumi/BEGAN-tensorflow/tree/master/Data/celeba

\
The Facial Anootations are calculated using the method explained in 
A. Asthana, S. Zafeiriou, S. Cheng, M. Pantic, Incremental Face Alignment in the Wild, in: 2014 IEEE Conf. Comput. Vis. Pattern Recognit., 2014: pp. 1859–1866. doi:10.1109/CVPR.2014.240
The code is available at ("Chehra Matlab Fitting Code")
https://sites.google.com/site/chehrahome/

\
The python file "BEGANattempt3.py" is the BEGAN implementation in Lasagne on top of Theano.

\
The file "PointGen1.py" is the python implementation of inverse of generator in Lasagne on top of Theano.

\
The python file "Param2Point.py" is the tarining script for the landmark genertor.

