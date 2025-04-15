# Evaluating Differential Privacy in Sentiment Analysis Models

This repository implements differentially private sentiment analysis using transformer-based models. The project explores the privacy-utility tradeoff by applying DP-SGD (Differentially Private Stochastic Gradient Descent) with various privacy budgets to a DistilBERT models and evaluating their vulnerability to membership inference attacks.


## Model Architecture

- Base: Transformer architecture (BERT/DistilBERT)
- Layer Configuration: Last 4 transformer layers trainable, earlier layers frozen
- Classification Head: Single dense hidden layer (128 neurons) with ReLU activation
- Regularization: Deliberately removed to allow observation of overfitting vs. DP effects

## Privacy Implementation

- Privacy Mechanism: DP-SGD via DPKerasAdamOptimizer
- Privacy Budgets: ε = [10.0, 1.0, 0.1] and baseline (no DP)
- Delta (δ): 1/number_of_training_examples
- Gradient Clipping: Selected from [0.3, 0.5, 1.0, 1.5]
- Microbatch Size: 1 (limited by memory constraints)

## Key Findings

1. Privacy-Utility Tradeoff: Stronger privacy guarantees (lower ε) lead to reduced model performance
2. DP as Regularization: DP prevents overfitting and model memorization
3. MIA Resistance: DP models effectively resist membership inference attacks (AUC ≈ 0.5)
4. Diminishing Returns: Minimal privacy benefit when reducing ε from 1.0 to 0.1, despite significant utility cost

## Future Work

- Fine-tuning hyperparameters to improve utility while maintaining privacy
- Testing larger microbatch sizes and dataset sizes with more computational resources
- Exploring alternative DP mechanisms
- Evaluating against different privacy attacks

## Acknowledgments

This project utilizes TensorFlow Privacy and Hugging Face Transformers libraries and the Yelp Academic Dataset. We also build on prior work in differential privacy and membership inference attacks.

## References

1. Vogel, F., & Lange, L. (2023). [Privacy-Preserving Sentiment Analysis on Twitter](https://dbs.uni-leipzig.de/files/research/publications/2023-9/pdf/SKILL2023_private_twitter_sentiment-6.pdf). In SKILL 2023: Lecture Notes in Informatics.
2. Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). [Deep Learning with Differential Privacy](https://doi.org/10.1145/2976749.2978318). *ArXiv*.
3. [TensorFlow Privacy: Get Started](https://www.tensorflow.org/responsible_ai/privacy/guide/get_started)
4. [TensorFlow Privacy GitHub Repository](https://github.com/tensorflow/privacy)
5. [TensorFlow Privacy Walkthrough Tutorial](https://github.com/tensorflow/privacy/blob/master/tutorials/walkthrough/README.md)
6. [TensorFlow Privacy Report Notebook](https://colab.research.google.com/github/tensorflow/privacy/blob/master/g3doc/tutorials/privacy_report.ipynb)
7. Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2016). [Membership Inference Attacks against Machine Learning Models](https://arxiv.org/abs/1610.05820). *ArXiv*.