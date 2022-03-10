from django.shortcuts import render
from .models import *
from .serializers import ImageRetrievalSerializer
from rest_framework.response import Response
from rest_framework.generics import CreateAPIView
from rest_framework import status
from PIL import Image
import numpy as np
import cv2
from pickle import load
from numpy import argmax
import time
import pickle
import numpy as np
from keras.models import Model
from keras.datasets import mnist
import cv2
from keras.models import load_model
from sklearn.metrics import label_ranking_average_precision_score
import time
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from numpy import load
#from imutils import build_montages
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
sys.argv=['']
del sys
class ImageRetrievalAPIView(CreateAPIView):
    serializer_class =ImageRetrievalSerializer
    queryset = ImageRetrieval.objects.all()
    def create(self, request, format=None):
        """
                Takes the request from the post and then processes the algorithm to extract the data and return the result in a
                JSON format
                :param request:
                :param format:
                :return:
                """

        serializer = self.serializer_class(data=request.data)

        if serializer.is_valid():


            folder_path=self.request.data['folder_path']
            number_of_image_to_search=self.request.data['number_of_image_to_search']
            number_of_result=self.request.data['number_of_result']

            content = []

            
            folder_path="C:\\Users\\Shivam\\Pictures\\image_retrieval_api\\image_retrieval\\image_retrieval_api\\photos\\" + str(folder_path)
            print("main_image_url:::::",folder_path)
           
            self.image_retrieval_function(folder_path,number_of_image_to_search,number_of_result)
           

            # add result to the dictionary and revert as response
            mydict = {
                'status': True,
                'response':
                    {

                        'Description':"successfully completed",
                    }
            }
            content.append(mydict)

            return Response(content, status=status.HTTP_200_OK)
        errors = serializer.errors

        response_text = {
                "status": False,
                "response": errors
            }
        return Response(response_text, status=status.HTTP_400_BAD_REQUEST)
    def image_retrieval_function(self,image_to_search,number_of_image_to_search,number_of_result):
        def euclidean(a, b):
            # compute and return the euclidean distance between two vectors
            return np.linalg.norm(a - b)
        def perform_search(queryFeatures, index, maxResults=64):
            # initialize our list of results
            results = []
            # loop over our index
            for i in range(0, len(index["features"])):
                d = euclidean(queryFeatures, index["features"][i])
                results.append((d, i))
            # sort the results and grab the top ones
            results = sorted(results)[:maxResults]
            # return the list of results
            return results    
        
        data = load('C:\\Users\\Shivam\\Pictures\\image_retrieval_api\\image_retrieval\\image_retrieval_api\\photos\\own_data.npz')
        x_train, x_test = data['arr_0'], data['arr_1']
        print(x_test)
        print('Loaded: ',x_train.shape, x_test.shape)

        index = pickle.loads(open("C:\\Users\\Shivam\\Pictures\\image_retrieval_api\\image_retrieval\\image_retrieval_api\\photos\\index.pickle", "rb").read())
        # create the encoder model which consists of *just* the encoder
        t0 = time.time()
        autoencoder = load_model('C:\\Users\\Shivam\\Pictures\\image_retrieval_api\\image_retrieval\\image_retrieval_api\\photos\\autoencoder.h5')
        encoder = encoder = Model(autoencoder.layers[0].input, autoencoder.layers[8].output)
        t1 = time.time()
        print('Model loaded in: ', t1-t0)

        features = encoder.predict(x_test)

        # randomly sample a set of testing query image indexes
        #queryIdxs = list(range(0, x_test.shape[0]))
        #queryIdxs = np.random.choice(queryIdxs, int(number_of_image_to_search),
           # replace=False)
        # loop over the testing indexes
        for i in range(int(number_of_image_to_search)):
            # take the features for the current image, find all similar
            # images in our dataset, and then initialize our list of result
            # images
            queryFeatures = features[i]
            maxResults=int(number_of_result)
            results = perform_search(queryFeatures, index,maxResults)
            images = []
            # loop over the results
            for (d, j) in results:
                # grab the result image, convert it back to the range
                # [0, 255], and then update the images list
                image = (x_train[j] * 255).astype("uint8")
                images.append(image)
            # display the query image
            query = (x_test[i] * 255).astype("uint8")
            #plt.imshow(query,aspect="auto")
            plt.imsave(str(image_to_search)+'\\search_image' +str(i)+'.png',query)
            #plt.show()
            print("now similar image")    
            for i in range(maxResults):
                plt.imsave('C:\\Users\\Shivam\\Pictures\\image_retrieval_api\\image_retrieval\\image_retrieval_api\\photos\\retrieve_images\\retrieve_image'+str(i)+'.png',images[i])
                