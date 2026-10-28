from django.shortcuts import render,redirect
import pandas as pd
import numpy as np
# import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
# from django.http import HttpResponse
import os
from .models import Images

# classess of flowers
Kidney=['Cyst', 'Normal', 'Stone', 'Tumor']

# Create your views here.
def index(request):
    return render(request,'index.html') # Home Page

def result(request):
    if request.method=='POST':
        m=int(request.POST['alg'])
        file=request.FILES['file']
        fn=Images(images=file)
        fn.save()
        path=os.path.join('webapp/static/images/',fn.filename())
        acc=pd.read_csv("webapp/Accurary.csv")

        if m==1:
            new_model=load_model("webapp/models/CNN.h5")
            test_image=image.load_img(path,target_size=(256,256))
            test_image=image.img_to_array(test_image)
            test_image/=255
            a=acc.iloc[m -1,1]
        
        else:
            new_model=load_model("webapp/models/MobileNet.h5")
            test_image=image.load_img(path,target_size=(256,256))
            test_image=image.img_to_array(test_image)
            test_image/=255
            a=acc.iloc[m-1,1]


        test_image=np.expand_dims(test_image,axis=0)
        result=new_model.predict(test_image)
        pred=Kidney[np.argmax(result)]
        print(pred)
        
        return render(request,'result.html',{'text':pred,'path':'static/images/'+fn.filename(),'a':round(a*100,3)})
    return render(request,'result.html')
