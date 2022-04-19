{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "896b9c4c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "df = pd.read_csv('d:/data/data/application_train.csv')\n",
    "df = df.sample(frac=0.1) # Take some records just to build a toy model\n",
    "variables = ['APARTMENTS_AVG','LIVINGAPARTMENTS_AVG','OCCUPATION_TYPE']\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(df[variables],df['TARGET'],random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4e107b2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "###### Data pre-processing: fillna for numeric variables #####\n",
    "    \n",
    "# Impute with the mean of the training data\n",
    "# Keep the same mean to impute the test data or any future data\n",
    "APARTMENTS_AVG_MEAN = X_train['APARTMENTS_AVG'].mean()\n",
    "APARTMENTS_AVG_MAX = X_train['APARTMENTS_AVG'].mean()\n",
    "APARTMENTS_AVG_MIN = X_train['APARTMENTS_AVG'].min()\n",
    "LIVINGAPARTMENTS_AVG_MEAN = X_train['LIVINGAPARTMENTS_AVG'].mean()\n",
    "LIVINGAPARTMENTS_AVG_MAX = X_train['LIVINGAPARTMENTS_AVG'].max()\n",
    "LIVINGAPARTMENTS_AVG_MIN = X_train['LIVINGAPARTMENTS_AVG'].min()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "b197a622",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.0, 1.0)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "LIVINGAPARTMENTS_AVG_MIN, LIVINGAPARTMENTS_AVG_MAX"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "06a86e65",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.0, 0.1166635277582568)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "APARTMENTS_AVG_MIN, APARTMENTS_AVG_MAX"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8d40d076",
   "metadata": {},
   "outputs": [],
   "source": [
    "mean_values = {'APARTMENTS_AVG': APARTMENTS_AVG_MEAN, 'LIVINGAPARTMENTS_AVG': LIVINGAPARTMENTS_AVG_MEAN}\n",
    "X_train = X_train.fillna(value=mean_values)\n",
    "X_test = X_test.fillna(value=mean_values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "da48d16a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Laborers                 3981\n",
       "Sales staff              2436\n",
       "Core staff               2101\n",
       "Managers                 1604\n",
       "Drivers                  1397\n",
       "High skill tech staff     870\n",
       "Accountants               751\n",
       "Medicine staff            637\n",
       "Security staff            469\n",
       "Cooking staff             442\n",
       "Cleaning staff            363\n",
       "Private service staff     199\n",
       "Low-skill Laborers        151\n",
       "Secretaries               112\n",
       "Waiters/barmen staff       96\n",
       "Realty agents              61\n",
       "HR staff                   42\n",
       "IT staff                   35\n",
       "Name: OCCUPATION_TYPE, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "######## Data pre-processing: categorical variables #####\n",
    "\n",
    "X_train['OCCUPATION_TYPE'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5b50ed55",
   "metadata": {},
   "outputs": [],
   "source": [
    "OCCUPATION_list  = ['Laborers','Sales staff','Core staff','Managers','Drivers','High skill tech staff',\n",
    "'Accountants','Medicine staff','Security staff','Cooking staff','Cleaning staff']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "08fe7d6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train['OCCUPATION_GRP'] = np.where(X_train['OCCUPATION_TYPE'].isin(OCCUPATION_list), X_train['OCCUPATION_TYPE'], 'OTHER')\n",
    "X_test['OCCUPATION_GRP'] = np.where(X_test['OCCUPATION_TYPE'].isin(OCCUPATION_list), X_test['OCCUPATION_TYPE'], 'OTHER')\n",
    "X_train = X_train.drop('OCCUPATION_TYPE',axis=1)\n",
    "X_test = X_test.drop('OCCUPATION_TYPE',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "90bf1738",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "OTHER                    8012\n",
       "Laborers                 3981\n",
       "Sales staff              2436\n",
       "Core staff               2101\n",
       "Managers                 1604\n",
       "Drivers                  1397\n",
       "High skill tech staff     870\n",
       "Accountants               751\n",
       "Medicine staff            637\n",
       "Security staff            469\n",
       "Cooking staff             442\n",
       "Cleaning staff            363\n",
       "Name: OCCUPATION_GRP, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train['OCCUPATION_GRP'].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "867d2441",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get_Dummy for One-Hot #\n",
    "def getDummy(df,var):\n",
    "    df[var] = df[var].str.replace(' ','_')\n",
    "    df[var] = df[var].str.replace('/','_')\n",
    "    dummies = pd.get_dummies(df[var])\n",
    "    df2 = pd.concat([df, dummies], axis=1)\n",
    "    df2 = df2.drop([var], axis=1)\n",
    "    return(df2)\n",
    "    \n",
    "X_train = getDummy(X_train,'OCCUPATION_GRP')\n",
    "X_test = getDummy(X_test,'OCCUPATION_GRP')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "62949f41",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['APARTMENTS_AVG', 'Accountants', 'Cleaning_staff', 'Cooking_staff',\n",
       "       'Core_staff', 'Drivers', 'High_skill_tech_staff',\n",
       "       'LIVINGAPARTMENTS_AVG', 'Laborers', 'Managers', 'Medicine_staff',\n",
       "       'OTHER', 'Sales_staff', 'Security_staff'], dtype=object)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.sort(X_train.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "d1a34523",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(max_depth=6, min_samples_leaf=5)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier # for classification\n",
    "\n",
    "# First, specify the model. \n",
    "dtree = DecisionTreeClassifier(min_samples_leaf = 5, max_depth = 6)\n",
    "# Then, train the model.\n",
    "dtree.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a3656131",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='feature'>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x216 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(dtree.feature_importances_,3)})\n",
    "importances = importances.sort_values('importance',ascending=False)\n",
    "importances.plot.bar(x='feature', figsize=(4,3),fontsize=6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "9cbb3c7d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.11086475, 0.07039373, 0.07039373, 0.07039373, 0.07039373])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Predict a few records\n",
    "predictions = dtree.predict_proba(X_test[0:5])[:,1]\n",
    "predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "82b1579f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.11086475])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pickle \n",
    "# save the model to disk\n",
    "modelname = path + '/mymodel.pkl'\n",
    "pickle.dump(dtree, open(modelname, 'wb'))\n",
    " \n",
    "# load the model from disk\n",
    "loaded_model = pickle.load(open(modelname, 'rb'))\n",
    "predictions = loaded_model.predict_proba(X_test[0:1])[:,1]\n",
    "predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "e6bb6512",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>APARTMENTS_AVG</th>\n",
       "      <th>LIVINGAPARTMENTS_AVG</th>\n",
       "      <th>Accountants</th>\n",
       "      <th>Cleaning_staff</th>\n",
       "      <th>Cooking_staff</th>\n",
       "      <th>Core_staff</th>\n",
       "      <th>Drivers</th>\n",
       "      <th>High_skill_tech_staff</th>\n",
       "      <th>Laborers</th>\n",
       "      <th>Managers</th>\n",
       "      <th>Medicine_staff</th>\n",
       "      <th>OTHER</th>\n",
       "      <th>Sales_staff</th>\n",
       "      <th>Security_staff</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>168018</th>\n",
       "      <td>0.116664</td>\n",
       "      <td>0.099971</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        APARTMENTS_AVG  LIVINGAPARTMENTS_AVG  Accountants  Cleaning_staff  \\\n",
       "168018        0.116664              0.099971            0               0   \n",
       "\n",
       "        Cooking_staff  Core_staff  Drivers  High_skill_tech_staff  Laborers  \\\n",
       "168018              0           0        0                      0         1   \n",
       "\n",
       "        Managers  Medicine_staff  OTHER  Sales_staff  Security_staff  \n",
       "168018         0               0      0            0               0  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test[0:1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef644fd9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
