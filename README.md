
Here, we assume that there are only NUM_LABELS training samples, and large number of unlabelled images such that
UNLABELLED_SAMPLES >> LABELLED_SAMPLES.

This is quite common in real-world where annotation is expensive but unlabelled data are relatively available in abundance.

We work through code example (FixMatch Algorithm, in this case) to leverage unlabelled data for better representation learning.

The basic idea here is to 

i) train the model using limited labelled data
ii) as the model is getting trained, use the trained model to obtain (pseudo)labels for which the model is relatively certain (P_CUTOFF)
iii) and then obtain a aggressively modified version of the image and train the model to predict the above mentioned (in ii) 
In actual implementation 

