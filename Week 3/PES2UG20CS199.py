'''
UE20CS302 (D Section)
Machine Intelligence
Week 3: Decision Tree Classifier

Mitul Joby
PES2UG20CS199
'''

import numpy as np

'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    entropy = 0
    colLast = df.columns[-1]
    vals = df[colLast].unique()
    length = len(df[colLast])
    for x in vals:
        value = df[colLast].value_counts()[x] / length
        if (value != 0):
            entropy = entropy -(value * np.log2(value))
    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    avg_info = 0
    vals = df[attribute].unique()
    length = len(df[attribute])
    for x in vals:
        value = df[attribute].value_counts()[x] / length
        dfAttr = df[df[attribute] == x]
        avg_info = avg_info + value * get_entropy_of_dataset(dfAttr)
    return avg_info


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    information_gain = get_entropy_of_dataset(df) - get_avg_info_of_attribute(df, attribute)
    return information_gain


'''
Return a tuple with the first element as a dictionary which has IG of all columns 
and the second element as a string with the name of the column selected

example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
'''
#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    informationGain = {}
    maxGain = float("-inf")
    for x in df.columns[:-1]:
        attr = get_information_gain(df, x)
        if attr > maxGain:
            col_name = x
            maxGain = attr
        informationGain[x] = attr
    return (informationGain, col_name)