print('\f')
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import csr_matrix
import codecs
import pandas as pd
import time
import random
from pylab import* 
from scipy import*
import pickle
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

with open('path') as data_file:    
    data = json.load(data_file)


def create_dict_cuisine_ingred(json):
    dictCuisineIngred = {}
    cuisines = []
    ingredients = []
    
    for i in range(len(json)):
        
       
        cuisine = json[i]['cuisine']
        if cuisine == 'indian':
            cuisine = 'Indian'

        ingredientsPerCuisine = json[i]['ingredients']
        
        if cuisine not in dictCuisineIngred.keys():
            cuisines.append(cuisine)
            dictCuisineIngred[cuisine] = ingredientsPerCuisine
            
        else: 
            currentList = dictCuisineIngred[cuisine]
            currentList.extend(ingredientsPerCuisine)
            dictCuisineIngred[cuisine] = currentList
                 
        ingredients.extend(ingredientsPerCuisine)
         
    ingredients = list(set(ingredients)) 
    numUniqueIngredients = len(ingredients)
    numCuisines = len(cuisines)
    
    return dictCuisineIngred, numCuisines, numUniqueIngredients, cuisines, ingredients

def create_term_count_matrix(dictCuisineIngred, numCuisines, numIngred, cuisines, ingredients):
    termCountMatrix = np.zeros((numCuisines,numIngred))
    i = 0
    
    for cuisine in cuisines:
        ingredientsPerCuisine = dictCuisineIngred[cuisine]

        for ingredient in ingredientsPerCuisine:
            j = ingredients.index(ingredient) 
            termCountMatrix[i,j] += 1

        i += 1

    return termCountMatrix


dictCuisineIngred, numCuisines, numIngred, cuisines, ingredients = create_dict_cuisine_ingred(data)
countsMatrix = create_term_count_matrix(dictCuisineIngred, numCuisines, numIngred, cuisines, ingredients)


def tf_idf_from_count_matrix(countsMatrix):
    
    countsMatrix = sparse.csr_matrix(countsMatrix)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(countsMatrix) 
    return tfidf.toarray()
    
tfIdf_Matrix = tf_idf_from_count_matrix(countsMatrix)



pca = PCA(n_components=2)  
reduced_data = pca.fit_transform(tfIdf_Matrix)

pca2dataFrame = pd.DataFrame(reduced_data)
pca2dataFrame.columns = ['PC1', 'PC2']


from sklearn.cluster import KMeans

def kmeans_cultures(numOfClusters):
    
    kmeans = KMeans(init='k-means++', n_clusters=numOfClusters, n_init=10)
    kmeans.fit(reduced_data)
    return kmeans.predict(reduced_data)

labels = kmeans_cultures(5)

i = 0 
j = 0 

effect_on_cluster = [0 for cuisine in cuisines]

for cuisineA in cuisines:  

    A_intersection = 0
    numInClusterBesidesA = 0
    setA = set(dictCuisineIngred[cuisineA])
    setB_forA = []
    j = 0
    
    for cuisineB in cuisines:
        if cuisineB != cuisineA: 
            if labels[j] == labels[i]: 
                setB_forA.extend(set(dictCuisineIngred[cuisineB]))
                numInClusterBesidesA += 1
        j += 1
    
    A_intersection = len(set(setA & set(setB_forA))) / float(len(set(setA.union(setB_forA))))
    effect_on_cluster[i] = A_intersection
       
    i += 1


start_time = time.time()
meal_id,cuisine,ingredients,ing,main_set,train_set,test_set =[],[],[],[],[],[],[]
predicted_cuisine = ''


def lists_creater(filename):
    with codecs.open( filename,encoding = 'utf-8') as f:
        data = json.load(f)
        
    for i in range(0,len(data)):
        meal_id.append(data[i]["id"])
        cuisine.append(data[i]["cuisine"])
        ingredients.append(data[i]["ingredients"])
        
    for i in ingredients:
        temp =u''
        for f in range(len(i)):
            temp = temp+u" "+i[f]
        ing.append(temp.encode('utf-8'))
   
        
    return meal_id
    return cuisine
    return ingredients
    return ing

 
def ing_vectorizer(exis_ing,user_ing):
    exis_ing.append(user_ing)
    vectorizer = TfidfVectorizer(use_idf = True, stop_words = 'english',max_features = 4000)
 
    ing_vect = vectorizer.fit_transform(exis_ing)
    return (ing_vect.todense())


def set_creator(main_set):
    train_set = main_set[:len(main_set)-1]
    test_set = main_set[len(main_set)-1]
    return (train_set,test_set)

def KNN_trainer(train_set,cuisine,n):  
    n = int(n)

    close_n = KNeighborsClassifier(n_neighbors=n)
    return close_n.fit(train_set,cuisine)


def KNN_predictor(test_set,close_n,no_of_neigh):
    no_of_neigh = int(no_of_neigh)
    print ("")
  
    predicted_cuisine = close_n.predict_proba(test_set)[0]
   
    predicted_single_cuisine = close_n.predict(test_set)
  
    predicted_class = close_n.classes_
    print ("The model predicts that the ingredients resembles %s" %(predicted_single_cuisine[0]))
    print ("")
    for i in range(len(predicted_cuisine)):
        if not(predicted_cuisine[i] == 0.0):
            print ("The ingredients resemble %s with %f percentage" %(predicted_class[i],predicted_cuisine[i]*100))
    
    print ("")
    print ("The %d closest meals are listed below : " % no_of_neigh)
    match_perc,match_id = close_n.kneighbors(test_set)
    for i in range(len(match_id[0])):
        print (meal_id[match_id[0][i]])
     
    print("--- It took %s seconds ---" %(time.time() - start_time))
    print ("")
    return predicted_single_cuisine


def seq_exec():
    user_ing = input("Enter the ingredients that you want to compare : ")
    main_set = ing_vectorizer(ing,user_ing)
    train_set,test_set = set_creator(main_set)
    no_of_neigh = input("Enter the number of closest items you want to find : ")
    close_n = KNN_trainer(train_set,cuisine,no_of_neigh)
    KNN_predictor(test_set,close_n,no_of_neigh)
    ing.pop()
    try:
        nextStep = int(input("Enter 1 if you want to search again or 2 if you want to quit.."))
        if not(nextStep == 1 or nextStep == 2):
            raise ValueError()
        elif (nextStep == 1):
            seq_exec()
        elif (nextStep == 2):
            quit()
    except ValueError:
        print ("Invalid Option. Enter correctly")
        seq_exec()
        
if __name__ == '__main__':
    print ("Reading all the data files and creating lists....")
    lists_creater(filename = "path.json")
    seq_exec()
    

