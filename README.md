# Logistic Regression with a Neural Network Mindset 🧠💻

This project demonstrates a neural network mindset in approaching logistic regression, applying it to a binary classification problem (cat vs. non-cat images). It builds a logistic regression classifier using a deep learning framework, with Python and essential libraries. The project also introduces neural network concepts such as forward propagation, cost function, and gradient descent.

**Credit**: This project is inspired by the course [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/) by DeepLearning.AI. Special thanks to the creators for providing excellent resources on deep learning! 🎓💡

## 🚀 Project Overview

### Problem Set
The objective is to classify images as either containing a cat 🐱 or not containing a cat 🙅‍♂️. The dataset consists of:
- **Training set**: 209 images
- **Testing set**: 50 images
- **Each image size**: 64x64x3 (RGB)

### Key Steps:
1. **Data Preprocessing**:
   - Normalize the dataset by scaling pixel values to [0, 1] range.
   - Reshape the image data into vectors for further processing.
   
2. **Logistic Regression Model**:
   - Implement the logistic regression model using a neural network mindset.
   - Build helper functions for sigmoid activation, initializing parameters, and propagation.
   - Use gradient descent for optimization.
   
3. **Evaluation**:
   - Use forward and backward propagation to compute the cost and gradients.
   - Optimize the model parameters (weights and bias) using gradient descent.
   - Assess the performance on the test set.

### Libraries Used:
- `numpy` for numerical computations
- `matplotlib` for data visualization
- `PIL` and `scipy` for image processing
- `h5py` for handling datasets

## 🛠️ How to Use the Code
### Prerequisites:
- Python 3.x
- Jupyter Notebook (optional for inline plotting)
  
### Setup Instructions:
1. Clone the repository:
   ```bash
   git clone https://github.com/MohammedSaqibMS/Logistic-Regression-as-a-Neural-Network.git
   ```
2. Install the necessary Python libraries:
   ```bash
   pip install numpy matplotlib Pillow scipy h5py
   ```
3. Run the notebook or the script:
   ```bash
   jupyter notebook logistic_regression_neural_network.ipynb
   ```

## 📊 Dataset Overview
- **Classes**: Binary (cat/non-cat)
- **Train set**: 209 images
- **Test set**: 50 images

Example of dataset exploration:
```python
plt.imshow(train_set_x_orig[10])  # Display an image from the dataset
print(f"y = {train_set_y[0, 10]}, it's a '{classes[np.squeeze(train_set_y[:, 10])].decode('utf-8')}' picture.")
```

## 💡 Key Concepts

1. **Sigmoid Activation Function**:
   ```python
   def sigmoid(z):
       return 1 / (1 + np.exp(-z))
   ```

2. **Cost Function**:
   The negative log-likelihood cost function used for logistic regression.

3. **Gradient Descent Optimization**:
   Optimize weights `w` and bias `b` by updating them iteratively based on the computed gradients:
   ```python
   w = w - learning_rate * dw
   b = b - learning_rate * db
   ```

## 🎯 Results
The model achieves reasonable accuracy in predicting whether the images contain cats. The cost reduces significantly during training, thanks to gradient descent optimization.

### Example of Cost Output:
```bash
Cost after iteration 0: 0.693
Cost after iteration 100: 0.218
```

## 🤝 Acknowledgments
This project is based on the coursework from the [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/) by DeepLearning.AI. Many thanks to Andrew Ng and the entire team for their contributions to AI education! 🙏
