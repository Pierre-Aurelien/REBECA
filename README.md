# REBECA
_Developing a neural network architecture to predict RBS strength_
## Biological context
The concentration of different proteins in bacteria varies significantly, spanning over five orders of magnitude. This huge variation is regulated by transcriptional and translational processes, with translation playing a dominant role. In many cases, translation initiation is the rate limiting step making its prediction a good indicator of overall protein synthesis rate. <br /> This approximation has been used to predict protein expression using thermodynamic features of the 5' untranslated region (5’ UTR). Our learning algorithm called REBECA (RibosomE Binding sitE CAlculator) was trained using data from Massively Parallel Report Assays (MPRAs) to predict protein expression rate from the 5’ UTR sequence alone. REBECA uses a combination of recurrent and convolutional neural networks, to exploit both the sequential and contextual structure of a DNA sequence during prediction. 

## Modelisation
__Labels__: DNA read count data was processed to recover the bin where most samples from one genetic construct fall into, ranging from bin no.0 to bin no.7 <br>
__Input__: 5'UTR of each mRNA sequence was extracted by defining the cut-off at the start codon AUG.<br>
We modelised this problem as a classification task.


## Key results
The final architecture had a 61% accuracy, jumping to 83% when allowing the predictions to be +/- one bin



<br>

![REBECA](https://user-images.githubusercontent.com/66125433/102891855-3f544000-445f-11eb-87cf-25b987a3da01.png)

Neural network approach to model gene regulation: Architecture, performance and benchmark. A) NN architecture of Rebeca B) Accuracy on the Evfratov validation set of each different NN architecture C) Confusion Matrix for Rebeca. The true and predicted labels are respectively on the x and y axis  D) Benchmarking Rebeca on the test set against the estimated accuracies of the Salis RBS calculator and the Evfratov algorithm. 


<img width="1026" alt="REBS_FT" src="https://user-images.githubusercontent.com/66125433/102892129-b8539780-445f-11eb-9e97-5582a155f023.png">
Figure 2: Rebeca generalizes to other datasets (a) Density plot of the Mean fluorescence predictions as a function of the ground truth after fine tuning on the fepB dataset from Kuo et al. Orange dotted lines correspond to y=x. (b) Generalization performance on the Kuo dataset

## Architectures considered and data augmentation
Sequences were one-hot encoded and padded on the right side with an array of 0 to account for the different lengths of 5'UTR sequences, resulting in a 30 ∗ 4 array for each 5'UTR. We reduced overfitting by artificially enlarging the training dataset using a label-preserving transformation consisting of generating image translations. The dataset was augmented by applying a shift of 0 to 10 nucleotide for sequences zero-padded on the left side. Using this trick, the number of training examples jumped from 25,389 to 130,299. <br>


We used custom scripts in addition to the Scikit modules RandomSearchCV and RandomForestRegressor. All of our models were trained with the Stochastic gradient descent optimizer, with the following hyperparameters: (learning rate=0.02, momentum=0.4, nesterov=False, clipvalue=1.0, batch size=32). We used ReLu for all activation functions. Early stopping was used to prevent overfitting to the training data.<br>

Cross-validation was performed to choose the model architecture. We tested combinations of the following hyperparameters: convolutional filter width: [9, 13, 17, 25], number of convolutional filters per layer: [32, 64, 128, 256], number of convolutional layers: [2, 3], number of dense layers: [1, 2,3], dropout probability in convolutionallayers: [0, 0.15],lstm hidden state dimension=[50, 70]  The best combination of hyperparameters resulted in the following model architecture: Conv1D: 128 filters, kernel size=4, dropout=0;Conv1D: 64 filters, kernel size=8, dropout=0; MaxPool1D, poolsize=2, BiderectionalLSTM, 50 units, dropout=0.1, 8 unit dense layer with softmax activation.



## Files

REBECA_tutorial.ipynb walks you through some steps of data pre-processing/ splitting and neural network deployment. <br>
Dataset folder contains both the Dataset used for training and testing (Evfratov_dataset.csv-Stratified split described in REBECA_tutorial.ipynb) and the weights and biases of REBECA ( Rebeca_v1.2.h5)
