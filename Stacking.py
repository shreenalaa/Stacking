import numpy as np 
import os 
os.system("cls") 
from sklearn.model_selection import train_test_split

class Stackig :
    def __init__ (self , base_models , meta_learner):    # user can choose base models and meta_learner 
        self.base_models = base_models
        self.meta_learner = meta_learner

    def fit (self , X ,y ) :
        base_prediction = []   # list to store the predictions of validation 
        # X >> All data 
        # split data into train and validation 
        X_train , X_validation , y_train , y_validation = train_test_split(X,y ,test_size=0.4 , random_state=42)
        # we will train base models 
        for model in (self.base_models):
            model.fit(X_train , y_train)
            # use base models to predict validation 
            base_prediction.append(model.predict(X_validation))

            # we can train meta learner >> input : base_prediction , output : y_validation
        base_prediction = np.column_stack(base_prediction)     # to make list in 2 d  
        self.meta_learner.fit(base_prediction,y_validation)
        # the model become able to take predection and get the final result (y) instade of voting 
    
    def predict ( self , X):
        base_prediction = np.column_stack([model.predict(X) for model in self.base_models] )
        return self.meta_learner.predict(base_prediction)