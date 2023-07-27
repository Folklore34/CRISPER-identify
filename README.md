Anti-CRISPER Protein identification
===
Dataset collections
-------
Train data
--
For positive samples, 480 experimental verified RGENs were collected from literature.
And by CD-HIT (setting 70% threshold), finally 389 RGENs samples were filtered.        
For negative samples, samples were collected based on the following criteria:
>* Must not be known or putative Class II effector proteins.
>* Must be isolated from phage or bacterial MGEs (known or putative MGEs).
>* must have < 40% sequence similarity to each other and the positive samples.

The composition of positive samples and negative samples are listed below.

* Positive samples 

`389 RNA guided endonuclease, RGENs (Cas9/Cas12/IscB/TnpB/Cas13)` 

* Negative samples

`719 Class II CRISPR-Cas accessory proteins / Class I CRIPSR-Case `

Test data
----
Totally,102 positive samples were collected and the composition of negative samples are listed below.
>* BEST (18)  
>* Proteins to be validated (7)  
>* TnpB (59)  
>* Fanzor (11)  
>* Cas12n (7)  

Totally,330 negative samples were collected and the composition of positive samples are listed below.  

`Class II CRISPR-Cas accessory proteins / Class I CRIPSR-Cas`


Usages
-----
First, you need to download and prepare the data that you require form train data set. 
After preparing the features in training dataset, download the code folder. Before utilizing the python scripts, you need to input the corresponding positive samples file, negative samples file and feature name. For using the ESM feature, if you want to decide the columns of feature selections in advance, you also need to input in `mrmrK`.  

The output includes:

Validation performances of 5-fold cross validation. (training set)  
Test performance.  

The ROC image of the training dataset.

The ROC image and test predict score of the test dataset.





