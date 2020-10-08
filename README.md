# REBECA
_Developing a neural network architecture to predict RBS strength_

The concentration of different proteins in bacteria varies significantly, spanning over five orders of magnitude. This huge variation is regulated by transcriptional and translational processes, with translation playing a dominant role. In many cases, translation initiation is the rate limiting step making its prediction a good indicator of overall protein synthesis rate. <br /> This approximation has been used to predict protein expression using thermodynamic features of the 5' untranslated region (5’ UTR). Our learning algorithm called REBECA was trained using data from Massively Parallel Report Assays (MPRAs) to predict protein expression rate from the 5’ UTR sequence. REBECA uses a combination of recurrent and convolutional neural networks, to exploit both the sequential and contextual structure of a DNA sequence during prediction. 



## Key results
The final architecture had a 55% accuracy, jumping to 83% when allowing the predictions to be +/- one bin


| Metric| RBS calculator|  Evfratov | Rebeca  |
| :---         |     :---:      |  :---:        |     :---:       |
| R squared   | 0.78   | 0.77  |   -   |
| Accuracy    | -    |21.3%     | __56.2%__       |


<br>

![REBECA](https://user-images.githubusercontent.com/66125433/95323591-563eb980-0896-11eb-81e5-497eac279826.jpg)


## Files

REBECA_tutorial.ipynb walks you through some steps of data pre-processing/ splitting and neural network deployment. 
Dataset folder contains both the Dataset used for training and testing (Evfratov_dataset.csv-Stratified split described in REBECA_tutorial.ipynb) and the weights and biases of REBECA ( Rebeca_v1.2.h5)
