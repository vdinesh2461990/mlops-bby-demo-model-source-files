from mlflow import pyfunc
import os
import pandas as pd

#Model wrapper that serves the predictions.
class MLFlowModel(object):

    def __init__(self):
        
        
        self.pyfunc_model = pyfunc.load_model("s3://ml-demo-dinesh/minst/5adfdbfaff3d4f2ca43f8d5b85d69df8/artifacts/model")

    def predict(self,X,features_names):
        if not features_names is None and len(features_names)>0:
            df = pd.DataFrame(data=X,columns=features_names)
        else:
            df = pd.DataFrame(data=X)
        return self.pyfunc_model.predict(df)
