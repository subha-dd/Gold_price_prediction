import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open("project1.pkl" , 'rb'))

mnth = {'January':1, 'February':2, 'March':3, 
        'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}

lst = ['January','February','March','April','May', 'June','July', 'August','September','October','November','December']

lst2 = [1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960,
       1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971,
       1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982,
       1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993,
       1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004,
       2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
       2016, 2017, 2018, 2019, 2020]

#creating a function for prediction

def diabetes_pred(input_data):

    #For prerdicting the output


    #changing the input_data into numpy array
    input_data_as_np_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instances
    input_data_reshaped = input_data_as_np_array.reshape(1 , -1)

    prediction = np.exp(loaded_model.predict(input_data_reshaped))

    return prediction[0]

    
    
def main():
    #giving a title
    st.title('Gold Price Prediction Model')
    
    #getting the input data from the user
    month = st.selectbox('enter month', lst)
    month = mnth[month]
    
    yr = st.selectbox('Enter Year', lst2)
    
    #code for prediction
    diagnosis = ''
    
    #creating a buuton for prediction
    if st.button('Gold price prediction'):
        diagnosis = diabetes_pred([month, yr])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()

