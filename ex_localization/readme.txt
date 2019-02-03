This is an example to localize sources with the method and neural networks presented in 
[1] L. Perotin, R. Serizel, E. Vincent, and A. Guérin, “CRNN-based joint azimuth and elevation localization with the Ambisonics intensity vector,” presented at the IWAENC 2018 - 16th International Workshop on Acoustic Signal Enhancement, 2018.

[2] L. Perotin, R. Serizel, E. Vincent, and A. Guérin, “CRNN-based multiple DoA estimation using acoustic intensity features for Ambisonics recordings,” submitted in IEEE JOURNAL OF SELECTED TOPICS IN SIGNAL PROCESSING in 2018.



The network trained on two sources ('foadoa_2src_0611.h5') uses a custom bias constraint (non_pos). 
In order to load it, you need to replace the Keras function 'constraint.py' by the provided one.
The Keras functions are usually found in your Python installation folder, in site-packages/keras/ .

You can then run the 'ex_localization.py' file.