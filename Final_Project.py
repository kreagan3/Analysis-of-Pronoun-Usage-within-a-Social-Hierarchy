#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import Libraries
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import nltk
import pandas as pd
import os
import re
from collections import defaultdict
import glob


# # Preprocessing

# ### Create Dataframe, Tokenization, Word Counts

# In[3]:


#open files
directory = r'/Users/kennareagan/Corpus_Linguistics/speech_events'
with open("/Users/kennareagan/Corpus_Linguistics/speech_events/all.txt", "w") as outfile:
    for filename in os.listdir(directory):
        file_name=filename
        if filename.endswith(".txt"):
            print(file_name)
            file = open(filename).read()
            outfile.write(file)
               
        
#create dictionary
        new_dict=defaultdict(list)
        pattern=re.compile(r'(S\d+):\s+(.+)',re.MULTILINE)

#create a new class for each speaker id
        for match in pattern.finditer(file):
            key,value=match.groups()
            new_dict[key].append(value)

#create dataframe with dictionary
        df1=pd.DataFrame(list(new_dict.items()),columns=['speaker_id','utterances'])
    
#create another dataframe for speaker info
        person_list=re.findall(r'(ID=\".+\")\s(LANG=\".+\")\s(ROLE=\".+\")\s(SEX=\".+\")\s(RESTRICT=\".+\")\s(AGE=\".+\")',file)
        df2=pd.DataFrame([{x.split('=')[0] : x.split('=')[1].replace('"', '') for x in person} for person in person_list])

#join list as a string and remove uppercase
        df1['utterances']=df1['utterances'].str.join(' ')
        df1["utterances"]=df1["utterances"].str.lower()

#create new column for tokenized utterances without punctuation
        tokenizer=RegexpTokenizer(r'\w+')
        df1['utterances']=df1.apply(lambda row: tokenizer.tokenize(row['utterances']),axis=1)

#get word count
        df1['#_of_words']=df1.apply(lambda row: len(row['utterances']),axis=1)

#get 1P SG pronouns
#df['1p_sg']=
        df1['i']=df1.apply(lambda row: row['utterances'].count("i"), axis = 1)
        df1['me']=df1.apply(lambda row: row['utterances'].count("me"), axis = 1)
        df1['my']=df1.apply(lambda row: row['utterances'].count(""), axis = 1)
        df1['mine']=df1.apply(lambda row: row['utterances'].count("mine"), axis = 1)
        df1['myself']=df1.apply(lambda row: row['utterances'].count("myself"), axis = 1)

        df1['1p_sg']=df1['i']+df1['me']+df1['my']+df1['mine']+df1['myself']
        del df1['i']
        del df1['me']
        del df1['my']
        del df1['mine']
        del df1['myself']

#get 1P PL pronouns
        df1['we']=df1.apply(lambda row: row['utterances'].count("we"), axis = 1)
        df1['us']=df1.apply(lambda row: row['utterances'].count("us"), axis = 1)
        df1['our']=df1.apply(lambda row: row['utterances'].count("our"), axis = 1)
        df1['ours']=df1.apply(lambda row: row['utterances'].count("ours"), axis = 1)
        df1['ourselves']=df1.apply(lambda row: row['utterances'].count("ourselves"), axis = 1)

        df1['1p_pl']=df1['we']+df1['us']+df1['our']+df1['ours']+df1['ourselves']
        del df1['we']
        del df1['us']
        del df1['our']
        del df1['ours']
        del df1['ourselves']

#get 2P pronouns
        df1['you']=df1.apply(lambda row: row['utterances'].count("you"), axis = 1)
        df1['your']=df1.apply(lambda row: row['utterances'].count("your"), axis = 1)
        df1['yours']=df1.apply(lambda row: row['utterances'].count("yours"), axis = 1)
        df1['yourself']=df1.apply(lambda row: row['utterances'].count("yourself"), axis = 1)
        df1['yourselves']=df1.apply(lambda row: row['utterances'].count("yourselves"), axis = 1)

        df1['2p']=df1['you']+df1['your']+df1['yours']+df1['yourself']+df1['yourselves']
        del df1['you']
        del df1['your']
        del df1['yours']
        del df1['yourself']
        del df1['yourselves']
        
#count third person pronouns        
        df1['they']=df1.apply(lambda row: row['utterances'].count("they"), axis = 1)
        df1['them']=df1.apply(lambda row: row['utterances'].count("them"), axis = 1)
        df1['themselves']=df1.apply(lambda row: row['utterances'].count("themselves"), axis = 1)
        df1['theirs']=df1.apply(lambda row: row['utterances'].count("theirs"), axis = 1)
        df1['their']=df1.apply(lambda row: row['utterances'].count("their"), axis = 1)
        df1['he']=df1.apply(lambda row: row['utterances'].count("he"), axis = 1)
        df1['him']=df1.apply(lambda row: row['utterances'].count("him"), axis = 1)
        df1['himself']=df1.apply(lambda row: row['utterances'].count("himself"), axis = 1)
        df1['his']=df1.apply(lambda row: row['utterances'].count("his"), axis = 1)
        df1['she']=df1.apply(lambda row: row['utterances'].count("she"), axis = 1)
        df1['her']=df1.apply(lambda row: row['utterances'].count("her"), axis = 1)
        df1['hers']=df1.apply(lambda row: row['utterances'].count("hers"), axis = 1)
        df1['herself']=df1.apply(lambda row: row['utterances'].count("herself"), axis = 1)

        df1['3p']=df1['they']+df1['them']+df1['theirs']+df1['their']+df1['he']+df1['him']+df1['his']        +df1['she']+df1['her']+df1['hers']+df1['themselves']+df1['himself']+df1['herself']
        del df1['they']
        del df1['them']
        del df1['themselves']
        del df1['theirs']
        del df1['their']
        del df1['he']
        del df1['him']
        del df1['himself']
        del df1['his']
        del df1['she']
        del df1['her']
        del df1['hers']
        del df1['herself']
#get pronoun count
        df1['#_of_pronouns']=df1['1p_sg']+df1['1p_pl']+df1['2p']+df1['3p']
        del df1['3p']

#merge dataframes and remove unnecessary columns
        df=pd.merge(left=df2, right=df1, how='left', right_on='speaker_id', left_on='ID')
        del df['speaker_id']
        del df['LANG']
        del df['RESTRICT']
        del df['AGE']
        
#percentages 
        df['%_1p_SG']=(df['1p_sg']/df['#_of_pronouns'])*100
        df['%_1p_pl']=(df['1p_pl']/df['#_of_pronouns'])*100
        df['%_2p']=(df['2p']/df['#_of_pronouns'])*100
        
        print (df)
        df.to_csv(file_name+'.csv')


# ### Concatenate csv Files

# In[13]:


extension='csv'
all_filenames=[i for i in glob.glob('*.{}'.format(extension))]

#combine files in list
full_set=pd.concat([pd.read_csv(f) for f in all_filenames ])

#export to csv
full_set.to_csv("full_set.csv", index=False)


# ### Create Separate Files for Different Roles

# In[14]:


#open full data set
new_df=pd.read_csv('full_set.csv')

#set index to ROLE
new_df=new_df.set_index("ROLE")

#create csv for roles higher in social hierarchy
df_higher=new_df.drop(["JU, Student", "SU, Student", "UN, Student", 'JG, Interviewee', 'UN, Unknown',                          'UN, Students', 'SU, Group Member', 'SU, Whole Group', 'SU, Student (Kurt)',                          'JU, Student (Phil)', 'SU, Student (Richie)', 'UN, Student (Buddy)', 'JG, Student',                          'UN, Students', 'SG, Student', 'JU, Students', 'SU, Students', 'SU, Student (Andy)',                          'JU, Student (John)','SU, Student (Mary)','JU, Student (Jim)','SU, Student (Tanya)',
                          'JU, Student (Paul)','JU, Student (Nicole)', 'ST, MICASE Researcher',\
                          'SG, Student (Samantha)','JG, Student (Marvin)', 'SG, Student (John)','JG, Student (Sung)',\
                          'UN, Friend','JG, MICASE researcher','JG, Geoff, Student', 'JG, Julio, Student',\
                          'JG, Martin, Student', 'JG, Hiro, Student', 'JF, Unknown','JG, Students', 'SF, Interviewee',\
                          'ST, Student (auditing)','ST, Student', 'SU, MICASE Researcher', 'UN, Unkown', 'SG, Visitor'])
df_higher.to_csv('df_higher.csv')


# In[15]:


#create csv for roles lower in social hierarchy
df_lower=new_df.drop(['SU, Peer Group Leader','SG, Instructor','JG, Graduate Student Instructor','JF, Instructor',                      'SF, Instructor','ST, MICASE Researcher','RE, Instructor, (Peter Nelson)','ST, Advisor',                      'SG, Graduate Student Instructor','UN, Unknown','SF, Unknown','SF, Interviewer','JF, Unknown',                      'JG, MICASE researcher','SF, Advisor','SU, Tutor','SU, Peer Leader','SG, Graduate Student',                      'SG, Discussion Leader','JU, Group Leader','JF, Interviewer','JF, Lecturer'])
df_lower.to_csv('df_lower.csv')


# In[ ]:


#create csv for settings where roles are equal
df_1=pd.read_csv('Math_Study_Group.txt.csv')
df_2=pd.read_csv('Senior_Thesis_Study_Group.txt.csv')
df_3=pd.read_csv('Natural_Resources_Group_Meeting.txt.csv')
df_4=pd.read_csv('Chemical_Engineering_Group_Project_Meeting.txt.csv')

df_equal=pd.concat([df_1,df_2,df_3,df_4])
df_equal.to_csv('df_equal.csv')


# ### Averages and graphs for equal roles

# In[ ]:


#set index to sex and create new csv for each gender
new_df=df_equal.set_index("SEX")
equal_m=new_df.drop(['F','U'])
equal_f=new_df.drop(['M','U'])
equal_m.to_csv('equal_m.csv')
equal_f.to_csv('equal_f.csv')


# In[ ]:


#collect avg percentages
avg_2p_f=equal_f['%_2p'].mean()
avg_2p_m=equal_m['%_2p'].mean()
avg_1p_sg_f=equal_f['%_1p_SG'].mean()
avg_1p_sg_m=equal_m['%_1p_SG'].mean()
avg_1p_pl_f=equal_f['%_1p_pl'].mean()
avg_1p_pl_m=equal_m['%_1p_pl'].mean()

#create df of averages
avg_equal={'Gender': ['M','F',],
        'AVG_%_1p_sg': [avg_1p_sg_m,avg_1p_sg_f],
        'AVG_%_1p_pl': [avg_1p_pl_m,avg_1p_pl_f],
        'AVG_%_2p': [avg_2p_m,avg_2p_f]
       }
df_avg_equal=pd.DataFrame(avg_equal,columns=['Gender','AVG_%_1p_sg','AVG_%_1p_pl','AVG_%_2p'])
print(df_avg_equal)
#df_avg_equal.to_csv('avg_equal.csv')

#plot averages
eq_1p_sg_fig=df_avg_equal.plot(x ='Gender', y='AVG_%_1p_sg', kind = 'bar')
eq_1p_pl_fig=df_avg_equal.plot(x ='Gender', y='AVG_%_1p_pl', kind = 'bar')
eq_2p_fig=df_avg_equal.plot(x ='Gender', y='AVG_%_2p', kind = 'bar')
eq_1p_sg_fig.figure.savefig('eq_1p_sg_fig.pdf')
eq_1p_pl_fig.figure.savefig('eq_1p_pl_fig.pdf')
eq_2p_fig.figure.savefig('eq_2p_fig.pdf')


# ### Averages and graphs for higher status

# In[16]:


df=pd.read_csv('df_higher.csv')
cms= df.columns
df_higher_m = pd.DataFrame(columns = cms)
df_higher_f = pd.DataFrame(columns = cms)
df_higher_f = df.loc[df['SEX'] == "F"]
df_higher_m = df.loc[df['SEX'] =="M"]
df_higher_f.to_csv('df_higher_f.csv')
df_higher_m.to_csv('df_higher_m.csv')


# In[22]:


#collect avg percentages
avg_2p_f=df_higher_f['%_2p'].mean()
avg_2p_m=df_higher_m['%_2p'].mean()
avg_1p_sg_f=df_higher_f['%_1p_SG'].mean()
avg_1p_sg_m=df_higher_m['%_1p_SG'].mean()
avg_1p_pl_f=df_higher_f['%_1p_pl'].mean()
avg_1p_pl_m=df_higher_m['%_1p_pl'].mean()

#create df of averages
avg_higher={'Gender': ['M','F',],
        'AVG_%_1p_sg': [avg_1p_sg_m,avg_1p_sg_f],
        'AVG_%_1p_pl': [avg_1p_pl_m,avg_1p_pl_f],
        'AVG_%_2p': [avg_2p_m,avg_2p_f]
       }
df_avg_higher=pd.DataFrame(avg_higher,columns=['Gender','AVG_%_1p_sg','AVG_%_1p_pl','AVG_%_2p'])
print(df_avg_higher)
df_avg_higher.to_csv('avg_higher.csv')

#plot averages
hi_1p_sg_fig=df_avg_higher.plot(x ='Gender', y='AVG_%_1p_sg', kind = 'bar')
hi_1p_pl_fig=df_avg_higher.plot(x ='Gender', y='AVG_%_1p_pl', kind = 'bar')
hi_2p_fig=df_avg_higher.plot(x ='Gender', y='AVG_%_2p', kind = 'bar')
hi_1p_sg_fig.figure.savefig('hi_1p_sg_fig.pdf')
hi_1p_pl_fig.figure.savefig('hi_1p_pl_fig.pdf')
hi_2p_fig.figure.savefig('hi_2p_fig.pdf')


# ### Averages and Graphs for Lower Status

# In[19]:


df=pd.read_csv('df_lower.csv')
cms= df.columns
df_lower_m = pd.DataFrame(columns = cms)
df_lower_f = pd.DataFrame(columns = cms)
df_lower_f = df.loc[df['SEX'] == "F"]
df_lower_m = df.loc[df['SEX'] =="M"]
df_lower_f.to_csv('df_lower_f.csv')
df_lower_m.to_csv('df_lower_m.csv')


# In[21]:


#collect avg percentages
avg_2p_f=df_lower_f['%_2p'].mean()
avg_2p_m=df_lower_m['%_2p'].mean()
avg_1p_sg_f=df_lower_f['%_1p_SG'].mean()
avg_1p_sg_m=df_lower_m['%_1p_SG'].mean()
avg_1p_pl_f=df_lower_f['%_1p_pl'].mean()
avg_1p_pl_m=df_lower_m['%_1p_pl'].mean()

#create df of averages
avg_lower={'Gender': ['M','F',],
        'AVG_%_1p_sg': [avg_1p_sg_m,avg_1p_sg_f],
        'AVG_%_1p_pl': [avg_1p_pl_m,avg_1p_pl_f],
        'AVG_%_2p': [avg_2p_m,avg_2p_f]
       }
df_avg_lower=pd.DataFrame(avg_lower,columns=['Gender','AVG_%_1p_sg','AVG_%_1p_pl','AVG_%_2p'])
print(df_avg_lower)
df_avg_lower.to_csv('avg_lower.csv')

#plot averages
lo_1p_sg_fig=df_avg_lower.plot(x ='Gender', y='AVG_%_1p_sg', kind = 'bar')
lo_1p_pl_fig=df_avg_lower.plot(x ='Gender', y='AVG_%_1p_pl', kind = 'bar')
lo_2p_fig=df_avg_lower.plot(x ='Gender', y='AVG_%_2p', kind = 'bar')
lo_1p_sg_fig.figure.savefig('lo_1p_sg_fig.pdf')
lo_1p_pl_fig.figure.savefig('lo_1p_pl_fig.pdf')
lo_2p_fig.figure.savefig('lo_2p_fig.pdf')


# In[ ]:




