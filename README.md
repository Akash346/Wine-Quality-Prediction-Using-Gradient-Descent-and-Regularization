**Dataset:**  
This assignment uses the Wine Quality Dataset, available on Canvas. Detailed information about this dataset can be found at [Wine Quality Dataset - UCI](https://archive.ics.uci.edu/dataset/186/wine+quality).

**Objective:**  
Apply the following gradient descent algorithms to the Wine Quality Dataset:
- **Batch Gradient Descent:**
  - Without regularization
  - With L2 regularization
  - With L1 regularization
- **Mini-batch Gradient Descent:**
  - Without regularization
  - With L2 regularization
  - With L1 regularization

### Data Preparation:
- Inspect and analyze the dataset before starting the ML tasks.
- Divide the dataset randomly into three subsets:
  - **Training set (t%):** Xtrain (m × (n + 1))
  - **Validation set (v%):** Xvalid (m × (n + 1))
  - **Test set (100 - t - v)%:** Xtest (m × (n + 1))
- Save the subsets for consistency across tasks.

### Algorithm Steps (Without Regularization):
1. **Initialize Weights:**  
   - Set up weight vector w of dimension (n + 1) and initialize randomly.
   
2. **Calculate Yhat:**  
   - For each data point in Xtrain, compute the dot product with w to obtain Yhat.
   
3. **Training MSE:**  
   - Calculate the training Mean Squared Error (MSE) as the mean of (Ytrain - Yhat)^2.
   
4. **Gradient of MSE (∇MSE):**  
   - Calculate the gradient vector of dimension (n + 1) using:  
     ∇MSE(w) = (2/m) × (Xtrain)’ × (Xtrain × w – Ytrain)
   
5. **Weight Update:**  
   - Update weights using:  
     w = w - λ ∇MSE(w)  
     (where λ is the learning rate)
   
6. **Repeat:**  
   - Repeat steps 3-5 until termination (error threshold, epoch limit, or a combination).

### Regularization Approaches:
- **L2 Regularization:**  
   - Objective: argmin {MSE(w) + α ||w||²}  
   - Gradient: ∇(MSE(w) + α ||w||²) = ∇MSE(w) + 2αw  
   - Select α using validation.

- **L1 Regularization:**  
   - Objective: ∇MSE + 2α sign(w)  
   - Select α using validation.

### Attribute Elimination:
- **L2:** Eliminate the attribute corresponding to the smallest weight component.
- **L1:** Eliminate the attribute corresponding to the smallest (or zero) weight component.

### Plotting:
- Plot regression lines for the attribute corresponding to the largest component of w, for batch gradient descent, before and after applying L1 and L2 regularization.

### Submission Instructions:
1. Submit a zipped folder with team names in alphabetical order.
2. Include a statement of contribution at the top of your program file.
3. Comment your program carefully.
4. Provide a one-page analysis comparing L1 and L2 regularization approaches.

