**CIL**  
   
   
# **ETHZ CIL Monocular Depth Estimation 2026**  
Using RGB images from scence of daily life, you will predict the corresponding pixel-wise depth of the scene.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAADUlEQVR4nGP4//8/AwAI/AL+p5qgoAAAAABJRU5ErkJggg==)  
## **ETHZ CIL Monocular Depth Estimation 2026**  
## **Overview**  
Welcome to the Computational Intelligence Lab’s Kaggle competition for Spring 2026. In this challenge, you’ll apply advanced machine learning techniques to predict depth from RGB images of everyday scenes. This project is designed to evaluate your ability to extract meaningful spatial information from visual data, demonstrating both theoretical understanding and practical application. Good luck!  
Start  
a month ago  
Close  
a month to go  
### **Description**  
A detailed description of this project is available here: [https://docs.google.com/document/d/1LtDmLhSFXd6MHdQFgi7yhVvbkzMDiUi6rBbuC2i4LPI/edit?usp=sharing](https://docs.google.com/document/d/1LtDmLhSFXd6MHdQFgi7yhVvbkzMDiUi6rBbuC2i4LPI/edit?usp=sharing "https://docs.google.com/document/d/1LtDmLhSFXd6MHdQFgi7yhVvbkzMDiUi6rBbuC2i4LPI/edit?usp=sharing")  
### **Evaluation**  
Your approach is evaluated using the scale-invariant Root Mean Square Error (RMSE) metric.  
## **Scale-Invariant RMSE**  
Scale-Invariant RMSE is a metric designed to evaluate depth estimation models by focusing on the relative differences in depth rather than the absolute scale. This is especially important in monocular depth estimation, where the global scale of the scene may be ambiguous.  
### **Why Use Scale-Invariant RMSE?**  
- **Relative Accuracy:  
 ** The metric operates in the logarithmic domain, meaning it measures errors based on the ratio of predicted to true depths. This approach emphasizes the spatial relationships and depth gradients in the image rather than penalizing a consistent scale shift.  
- **Robustness to Global Scale Variations:  
 ** A model might predict depth maps that are globally scaled versions of the ground truth. The scale-invariant RMSE minimizes the effect of such discrepancies by normalizing out the overall scale, ensuring that the error reflects the true structural differences in the scene.  
- **Emphasis on Scene Structure:  
 ** By focusing on the logarithmic differences between depth values, the metric rewards models that capture the underlying geometry and structure of everyday scenes, which is crucial for practical applications like autonomous navigation or augmented reality.  
### **How It Works**  
1. **Log Transformation:  
 ** Both the predicted depths and the ground truth depths are transformed using the natural logarithm.  
2. **Compute Differences:  
 ** For each pixel i, compute the difference:   
3. **Normalize with a Global Bias:  
 ** A bias term alpha is computed to minimize the overall error:   
4. **Error Calculation:  
 ** The scale-invariant RMSE is then given by:   
## **Competition Host**  
CIL Lecture  
   
## **Prizes & Awards**  
Kudos  
Does not award Points or Medals  
## **Participation**  
103 Entrants  
70 Participants  
18 Teams  
247 Submissions  
## **Tags**  
Custom Metric  
Table of Contents  
   
