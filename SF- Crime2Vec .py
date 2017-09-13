
# coding: utf-8

# In[1]:

#importing and loading all dependencies- functions, packages 

import pandas as pd 
import numpy as np 
import io 
import gensim, logging 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO) 

#Credits: getLOS function from co-intern Abhishek Patil

def getLOS( df, const_cols=None, only_cols=None, skip_cols=None ):
    # if there are columns to skip
    if skip_cols is not None:
        if set(skip_cols)<set(df.columns.tolist()):
            df = df.drop(skip_cols, axis=1)
            
    # if only specific columns are needed
    if only_cols is not None:
        df = df[only_cols]
    
    # if no constant columns specified then use one row as a sentence
    if const_cols is None:
        # define list of sentences
        listOfSentences = []
        
        # get all the rows of data frame as generator object
        rowsGen = df.iterrows()
        
        # keep adding to this list a sentence(i.e. a row) for every loop
        while True:
            nextRow = next(rowsGen, None)
            if nextRow == None:
                break
            listOfSentences.append([ str(elem) for elem in nextRow[1].values ])
            
    # else use the constant columns for each sentence
    else:
        # define list of sentences
        listOfSentences = [[]]
        
        # get column names from data frame
        dfColNames = list(df.columns)
        numDFCols = len(dfColNames)
        
        # number of constant columns
        numCCols = len(const_cols)
        
        # don't progress if const_cols doesn't
        #    1. have column names as in the df's column names
        #  & 2. have atleast 1 less column names than df's column names
        # condition 1 check
        if ( len( [elem for elem in const_cols if elem not in dfColNames] ) != 0 ):
            print( "Column names in const_cols parameter not found in provided data frame!" )
            return           
        # and condition 2 check
        if ( ( numDFCols-numCCols ) <= 1 ):
            print( "To make sentences, have atleast 1 column not in the const_cols parameter!" )
            return
        
        # get the column indices which are in df's column names but not in const_cols
        # need these indices to make the sentences
        colsLeft = np.setdiff1d(dfColNames, const_cols)        
        cols_left_ind = sorted([ dfColNames.index(elem) for elem in colsLeft ])
        
        # also get indices for const_cols
        const_cols_ind = sorted([ dfColNames.index(elem) for elem in const_cols ])
        
        # sort the data frame according to the const_cols
        # by default ascending is true
        df = df.sort_values(const_cols)
        
        # get all the rows of data frame as generator object
        rowsGen = df.iterrows()
        
        # iterator to go through the loop for indexing
        itr = 0
        
        # define the previous values of constant columns in a list using the first values
        prevColsVal = [ df[colName].values[0] for colName in const_cols ]
        
        # get a sentence for each different month of an year
        while True:
            nextRow = next(rowsGen, None)
            # break when end of data frame's rows
            if nextRow == None:
                break
            
            # get the row values
            rowValues = list( nextRow[1].values )
            
            # get the current values for columns in const_cols
            # basically index the values from the row we are currently in
            curColsVal = [rowValues[ind] for ind in const_cols_ind]
            
            # if previous and current list values are same add to existing sentence
            if (prevColsVal == curColsVal):
                listOfSentences[itr].extend( [ str(rowValues[ind]) for ind in cols_left_ind ] )
            # else add a new sentence            
            else:
                listOfSentences.append( [ str(rowValues[ind]) for ind in cols_left_ind ] )            
                # increment iterator
                itr = itr + 1
        
            # store current values as previous ones for the next iteration
            prevColsVal = curColsVal
            
    # return a dictionary object of the data frame and the list of sentences
    retDObj = { 'DF': df,
                'LOS': listOfSentences }
        
    return retDObj


# In[2]:

#Function to split date/time into date and time in separate columns 

def TimeGenerator(Stamp):
    NewTime=''
    i=8
    if (Stamp[8]!=' '):
        while (i<len(Stamp)):
            NewTime=NewTime+Stamp[i]
            i+=1 
    
    if (Stamp[8]==' ' and Stamp[9]!=' '):
        i+=1
        while (i<len(Stamp)):
            NewTime=NewTime+Stamp[i]
            i+=1
    return NewTime 

print (TimeGenerator('12/22/10 9:30')) 
                
#Our time generator function successfully separates the time from the date/time stamp 


# In[3]:

#Setting up the dataframe for Crime2Vec 

df=pd.read_csv("Desktop/SFtrain.csv", header=0, delimiter=",")  
df.drop(df.columns[[2, 5, 6,7, 8]], axis=1, inplace=True) 
#df['Category'].replace(['WARRANTS', 'OTHER OFFENSES', 'LARCENY/THEFT', 'VEHICLE THEFT', 'VANDALISM',
#'NON-CRIMINAL', 'ROBBERY', 'ASSAULT', 'WEAPON LAWS', 'BURGLARY',
#'SUSPICIOUS OCC', 'DRUNKENNESS', 'FORGERY/COUNTERFEITING', 'DRUG/NARCOTIC',
#'STOLEN PROPERTY', 'SECONDARY CODES', 'TRESPASS', 'MISSING PERSON', 'FRAUD',
#'KIDNAPPING', 'RUNAWAY', 'DRIVING UNDER THE INFLUENCE',
#'SEX OFFENSES FORCIBLE' ,'PROSTITUTION', 'DISORDERLY CONDUCT', 'ARSON',
#'FAMILY OFFENSES', 'LIQUOR LAWS' ,'BRIBERY', 'EMBEZZLEMENT', 'SUICIDE',
#'LOITERING' ,'SEX OFFENSES NON FORCIBLE', 'EXTORTION', 'GAMBLING',
#'BAD CHECKS', 'TREA', 'RECOVERED VEHICLE', 'PORNOGRAPHY/OBSCENE MAT'],
                                              
#['VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'VIOLENT', 'VIOLENT', 'VIOLENT', 'VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'VIOLENT', 'NON-VIOLENT', 'VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT', 'NON-VIOLENT'], inplace= True)

Time=[] 

for i in range(len(df['Dates'])):
    Vals=TimeGenerator(df['Dates'][i]) 
    Time.append(Vals) 
    
newDF=pd.DataFrame({'Time': Time}) #List has been converted to a dataframe 
    
df['Time']=newDF.values
df.drop(df.columns[[0,2]], axis=1, inplace= True)
df.tail() 


# In[4]:

#Applying Crime2Vec to the San Francisco Kaggle data set 

train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)

F=getLOS(train,const_cols=None, only_cols=None, skip_cols=None)  #This gives us the list of sentences to be fed into the word2vec model.   
TrainingSentences= F['LOS']   #storing the list of sentences value portion of the dictionary object returned by getLOS 
print (TrainingSentences[1]) 
print (test.head()) 

#TrainingSentences are now ready to be fed into word2vec model for training 


# In[5]:

#Train Crime2Vec 

from gensim.models import Word2Vec

model = gensim.models.Word2Vec(min_count=5, window=500, size=300, workers=3, sg=1, negative=5)       # an empty model, no training
model.build_vocab(TrainingSentences)  #1-pass step to build vocab
print ("Vocabulary successfully built!")  
model.train(TrainingSentences, total_examples=model.corpus_count,epochs=model.iter) #1 pass step to train model

#The model's vocabulary has been built and the model has now been trained on the corpus as well 


# In[6]:

tester=test.copy() 
tester.drop(tester.columns[[0]], axis=1, inplace=True) 

G=getLOS(tester,const_cols=None, only_cols=None, skip_cols=None)  #This gives us the list of sentences to be fed into the word2vec model.   
TestSentences=G['LOS'] 
print (TestSentences[1])  

#TestingSentences are now ready to be fed into the accuracy tester for testing 


# In[7]:

#Accuracy testing using predict_output_word- Gensim 

Counter=0 
Score=0
for i in range(0,len(TestSentences)):
    prediction=model.predict_output_word(TestSentences[i])
    Counter=Counter+1  
    if i==1:
        print (prediction) 
    if i==1000:
        print (prediction) 
    if i==20000: 
        print (prediction)  
    for j in range(len(prediction)):
        P=prediction[j]
        if (P[0]==(test['Category'].iloc[i])): 
            Score=Score+1
            if i==1: 
                print (test['Category'].iloc[i])   
print (Score) 
print (Counter) 
SampleSize=len(test['Category'])
Accuracy= (Score/SampleSize)*100 #Accuracy score is generated by dividing the number of matches by the number of cases 
Accuracy= "{0:.5f}".format(Accuracy)
Output = "The accuracy is " + Accuracy + "%." 
print (Output)            


# In[ ]:

#Accuracy testing using model.score()  

model.score(['VIOLENT NORTHERN 23:53'.split()]) 


# In[58]:

model.score(['VIOLENT CENTRAL 23:00'.split()]) 


# In[59]:

model.score(['CRIME IS BAD'.split()]) 


# In[9]:

#Cosine similarity checks

print (model.similarity('LARCENY/THEFT', 'KIDNAPPING')) 
print (model.similarity('FRAUD', 'FRAUD')) 
print (model.similarity('ARSON', 'FRAUD')) 
print (model.similarity('EMBEZZLEMENT', 'FRAUD')) 


# In[ ]:




# In[ ]:



