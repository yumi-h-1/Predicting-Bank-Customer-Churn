# Predicting Bank Customer Churn
## Project Overview
Customer churn analysis is crucial for banks aiming to maintain profitability, enhance customer satisfaction, and remain competitive. By accurately identifying customers at risk of leaving, banks can take proactive steps to improve retention and strengthen customer relationships. This project focuses on developing a supervised classification model for predicting customer churn using two machine learning approaches: Multilayer Perceptrons (MLP) and Support Vector Machines (SVM). The experiment compares the performance of these two models to determine which is better suited for predicting churn based on customer data. By evaluating model accuracy and classification effectiveness, we gain insights into the strengths and characteristics of each algorithm in handling churn prediction. Ultimately, the best-performing model will enable banks to anticipate future churn trends, helping them take timely actions to retain valuable customers and improve overall service quality.


## Project Files
- **Data**: Test sets are stored in the `data`. The raw data is derived from Hugging Face.
- **Notebooks**: The notebook for training and testing is located in the `notebooks/` folder.
- **Models**: The created models are located in the `models/` folder.
- **Results**: Visualisations, such as training accuracy or loss of the model, can be found in the `results/figures` folder.

## Methodology 
- **Data Preprocessing**:
  - For ***LSTM models***: Applied preprocessing steps including lowercasing, removing numbers, punctuation, double whitespaces, and stopwords using Python packages (`re`, `string`, and `nltk`). Converted text to lowercase and removed special characters.
  - For ***DistilBERT***: Used basic preprocessing in one model to compare its impact, while another DistilBERT model retained the original text.
  - Split the dataset into training, validation, and test sets, with 20% of the training set used for validation.

- **Tokenisation and Padding**:
  - ***LSTM***: Used TensorFlow’s `Tokenizer` for tokenisation and set padding to 29 tokens (based on median text length).
  - ***DistilBERT***: Used Hugging Face’s `DistilBertTokenizer` for tokenisation and applied the same padding length of 29.

- **Modeling**:
  - ***LSTM Models***: Developed three LSTM baselines with different word embeddings: Keras embedding layer (trained from scratch), pre-trained Word2Vec, and pre-trained GloVe embeddings. Each model has:
    - Three layers: embedding, LSTM (30 hidden units), and a dense layer with softmax for 77-class classification.
    - Adam optimizer (learning rate 0.001) and sparse categorical cross-entropy as the loss function.
    - Dropout regularisation (0.2) to prevent overfitting and early stopping set to 100 epochs with 20 patience.

    ***The best LSTM training loss***
    
    ![lstm-best-model-training-loss](results/figures/lstm-best-model-training-loss.png)

    ***The best LSTM training accuracy***
    
    ![lstm-best-model-training-accuracy](results/figures/lstm-best-model-training-accuracy.png)

  - ***DistilBERT Model***: Used Hugging Face’s pre-trained DistilBERT with a linear layer adjusted for 77-class output. Training was limited to 5 epochs, with no additional hyperparameter tuning or layer adjustments.

    ***The best DistilBERT training accuracy***
    
    ![distilbert-training-plot](results/figures/distilbert-training-plot.png)

- **Evaluation**: Assessed model performance using test accuracy, precision, recall, and F1 score, alongside training time and memory usage. The best-performing LSTM and DistilBERT models were compared on these metrics to determine efficiency and practical feasibility in chatbot applications.

## Key Findings
- **Model Performance**: The DistilBERT model significantly outperformed LSTM in accuracy, precision, recall, and F1-score by approximately 10 percentage points. DistilBERT achieved nearly 90% in both accuracy and F1-score, while the best LSTM model (using Word2Vec embeddings) reached 79.8%.
- **Training Time and Resource Constraints**: DistilBERT trained far more quickly on a T4 GPU, taking only 24 seconds, compared to LSTM’s 210.85 seconds on a CPU. However, DistilBERT’s large file size (506.8 MB) makes it 169 times larger than the LSTM model, implying higher storage and operational costs.

  ***Model comparison using the test set***
  
  | Model Name | Accuracy | Precision | Recall | F1-score | Training time (sec) | File size (MB) |
  | --- | --- | --- | --- | --- | --- | --- |
  | LSTM model | 79.84 | 81.46 | 79.84 | 79.54 | 210.85 (with CPU) | 3.3 |
  | DistilBERT model | 89.55 | 90.08 | 89.55 | 89.56 | 24.42 (with T4 GPU)	| 506.8 |

## Future Work
- **Model Enhancement**: Apply **data augmentation** and **cross-validation** to mitigate training set imbalances, potentially boosting performance for both models. Explore advanced LSTM architectures (e.g., bidirectional LSTMs) and DistilBERT fine-tuning, including layer adjustments and varied learning rates.
- **Model Compression**: Research lightweight model alternatives and explore knowledge distillation to transfer learned information from DistilBERT to a smaller model, improving storage and computational efficiency without major performance trade-offs.
- **Inference Testing**: Future work should validate model inference capabilities through real-time user interaction via API integration, such as those provided by Hugging Face, to assess practical deployment viability.


## Used Datasets
- [**Banking77**](https://huggingface.co/datasets/PolyAI/banking77)
