## Ideas for my project
- I want to include multi-task learning
- What is multi-task learning (MTL)?
    - Improves the performance of multiple related learning tasks by leveraging useful information among them
    - Requires 2 datasets of pretty equal size, to learn at the same time
- What is transfer learning?
    - Uses an existing model to boost the performance of a model with fewer data points

An idea for utilizing transfer learning:
    - Old Classifier: https://github.com/hukenovs/hagrid
        - This is a massive data set that contains 716GB (552k 1920x1080 images)
        - Classifies 18 gestures 
        - People standing variable distances away from their camera
        - They provide lots of models that are already trained, the idea is to use these models to transfer knowledge to ASL dataset
    - New Classifier: https://www.kaggle.com/datasets/danrasband/asl-alphabet-test
        - Small dataset containing 30 images for each letter in the alphabet
        
    - It is expect that the model performs pretty well on predicting the letter
    - But will it be able to do it from various angles/distances/people?



