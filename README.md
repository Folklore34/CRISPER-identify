Anti-CRISPER Protein identification
===
Dataset collections
-------
## Train data

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

## Independent test data

Totally,102 positive samples were collected and the composition of negative samples are listed below.
>* BEST (18)  
>* Proteins to be validated (7)  
>* TnpB (59)  
>* Fanzor (11)  
>* Cas12n (7)  

Totally,330 negative samples were collected and the composition of negative samples are listed below.  

`Class II CRISPR-Cas accessory proteins / Class I CRIPSR-Cas`


Usages
-----
First, you need to download and prepare the data that you require form train data set. 
After preparing the features in training dataset, download the code folder.  
Before utilizing the python scripts, you need to input the corresponding positive samples file, negative samples file and feature name. For using the ESM feature, if you want to decide the columns of feature selections in advance, you also need to input in `mrmrK`.  

The output will contain:

* A matrix demonstrates validation performances of 5-fold cross validation. 

* The ROC curve of the training dataset based on the result of cross validation.

```
ROC_5_fold(y_pred_valid_all,y_verified_valid_all,feature_name+'_ROC_5_fold.jpg')
```

![ESM_ROC_5_fold.jpg](figure/ESM_ROC_5_fold.jpg)

* A matrix contains the result of the test performance.

* The ROC curve of the test dataset.


![ESM_Test_ROC.jpg](figure/ESM_Test_ROC.jpg)

* The precision/recall curve.


![ESM_Test_PR_curve.jpg](figure/ESM_Test_PR_curve.jpg)

* The CSV file of predicted score.





## Independent test set
You need to input the corresponding test data file, feature name and the direction path of the training model.

The output contains:

* The ROC image of the test dataset.


![ESM_test_ROC.jpg](figure/ESM_test_ROC.jpg)
* The precision/recall curve.



![ESM_PR_curve.jpg](figure/ESM_PR_curve.jpg)
* The csv file of predicted score.


# Reference
Our work is based on the following literature.  
PreAcrs: a machine learning framework for identifying anti-CRISPR proteins. [https://doi.org/10.1186/s12859-022-04986-3]







