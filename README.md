# REBECA
_Developing a neural network architecture to predict RBS strength_
## Biological context
The concentration of different proteins in bacteria varies significantly, spanning over five orders of magnitude. This huge variation is regulated by transcriptional and translational processes, with translation playing a dominant role. In many cases, translation initiation is the rate limiting step making its prediction a good indicator of overall protein synthesis rate. <br /> This approximation has been used to predict protein expression using thermodynamic features of the 5' untranslated region (5’ UTR). Our learning algorithm called REBECA was trained using data from Massively Parallel Report Assays (MPRAs) to predict protein expression rate from the 5’ UTR sequence. REBECA uses a combination of recurrent and convolutional neural networks, to exploit both the sequential and contextual structure of a DNA sequence during prediction. 

## Modelisation
Labels: DNA read count data was processed to recover the bin where most samples from one construct fall into, ranging from bin no.0 to bin no.7
Input: 5'UTR of each mRNA sequence was extracted by defining the cut-off at the start codon AUG.
We modelised this problem as a classification task.


## Key results
The final architecture had a 55% accuracy, jumping to 83% when allowing the predictions to be +/- one bin


| Metric| RBS calculator|  Evfratov | Rebeca  |
| :---         |     :---:      |  :---:        |     :---:       |
| R squared   | 0.78   | 0.77  |   -   |
| Accuracy    | -    |21.3%     | __56.2%__       |


<br>

![REBECA](https://user-images.githubusercontent.com/66125433/95323591-563eb980-0896-11eb-81e5-497eac279826.jpg)

## Architectures considered and data augmentation
Sequences were one-hot encoded and padded on the right side with an array of 0 to account for the different lengths of 5'UTR sequences, resulting in a 30 ∗ 4 array for each 5'UTR. We reduced overfitting by artificially enlarging the training dataset using a label-preserving transformation consisting of generating image translations. The dataset was augmented by applying a shift of 0 to 10 nucleotide for sequences zero-padded on the left side. Using this trick, the number of training examples jumped from 25,389 to 130,299. <br>


We used custom scripts in addition to the Scikit modules RandomSearchCV and RandomForestRegressor. All of our models were trained with the Stochastic gradient descent optimizer, with the following hyperparameters: (learning rate=0.02, momentum=0.4, nesterov=False, clipvalue=1.0, batch size=32). We used ReLu for all activation functions. Early stopping was used to prevent overfitting to the training data.<br>

Cross-validation was performed to choose the model architecture. We tested combinations of the following hyperparameters: convolutional filter width: [9, 13, 17, 25], number of convolutional filters per layer: [32, 64, 128, 256], number of convolutional layers: [2, 3], number of dense layers: [1, 2,3], dropout probability in convolutionallayers: [0, 0.15],lstm hidden state dimension=[50, 70]  The best combination of hyperparameters resulted in the following model architecture, trained for 18 epochs (5): Conv1D: 128 filters, kernel size=4, dropout=0;Conv1D: 64 filters, kernel size=8, dropout=0; MaxPool1D, poolsize=2, BiderectionalLSTM, 50 units, dropout=0.1, 8 unit dense layer with softmax activation.



## Files

REBECA_tutorial.ipynb walks you through some steps of data pre-processing/ splitting and neural network deployment. 
Dataset folder contains both the Dataset used for training and testing (Evfratov_dataset.csv-Stratified split described in REBECA_tutorial.ipynb) and the weights and biases of REBECA ( Rebeca_v1.2.h5)
