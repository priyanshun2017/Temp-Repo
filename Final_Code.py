# Import all the tools we need
import numpy as np  # For purpose of array operations, as array is not available in python(instead list is available)
import pandas as pd  # For the purpose of data processing and reading data from files
from sklearn.svm import OneClassSVM  # Algorithm selected from the scikit-learn
from sklearn.metrics import accuracy_score  # Metric selected for testing the model, other metrics such as f1-score
import warnings
warnings.filterwarnings("ignore")  # Ignoring all the warnings as some methods used here are legacy now

# Pursuit denote Signals of interest
# Clutter denote non-Signals of interest
pursuit = pd.concat(map(pd.read_csv, ["D:\\ELSEC\\SortiModel\\data\\1.csv",
                                      "D:\\ELSEC\\SortiModel\\data\\2.csv",
                                      "D:\\ELSEC\\SortiModel\\data\\3.csv",
                                      "D:\\ELSEC\\SortiModel\\data\\4.csv"]), ignore_index=True)
# Fill all the files location with a comma to concatenate them into one dataframe
# The dataframe can be checked by dataframe_name.info() [or] dataframe_name.describe()
clutter = pd.concat(map(pd.read_csv, ["D:\\ELSEC\\SortiModel\\data\\1.csv",
                                      "D:\\ELSEC\\SortiModel\\data\\2.csv",
                                      "D:\\ELSEC\\SortiModel\\data\\3.csv",
                                      "D:\\ELSEC\\SortiModel\\data\\4.csv"]), ignore_index=True)
print('----------------------------------------------------------------------------------------')
pursuit = pursuit.drop(pursuit[pursuit.WeaponSystem == 'T5'].index)  # Dropping all Clutter signals from pursuit
clutter = clutter.drop(clutter[clutter.WeaponSystem != 'T5'].index)  # Dropping all signals of interest from clutter

pursuit.loc[
    pursuit.WeaponSystem == 'T5', 'Target'] = 0  # Filling 'Target' value with 0 where there are no signals of interest
# The only reason this step is being done is to make-sure there were no T5 that came accidentally to pursuit
pursuit.loc[
    pursuit.WeaponSystem != 'T5', 'Target'] = 1  # Filling 'Target' value with 1 where there are signals of interest
pursuit.loc[pursuit[
                'Target'] > 1.0, 'Target'] = 1  # Filling 'Target' value with 1 where there are signals of interest's original target value was greater than 1.0
pursuit['Target'] = pursuit['Target'].fillna(0)  # Filling 'Target' with 0 where ever 'Target' was not predefined

clutter.loc[
    clutter.WeaponSystem == 'T5', 'Target'] = 0  # Filling 'Target' value with 0 where there are no signals of interest
clutter.loc[
    clutter.WeaponSystem != 'T5', 'Target'] = 1  # Filling 'Target' value with 1 where there are signals of interest
# The only reason this step is being done is to make-sure there were no T1-T4 that came accidentally to clutter
clutter.loc[clutter[
                'Target'] > 1.0, 'Target'] = 0  # Filling 'Target' value with 1 where there are signals of interest's original target value was greater than 1.0
clutter['Target'] = clutter['Target'].fillna(0)  # Filling 'Target' with 0 where ever 'Target' was not predefined

pursuit['DF Type'] = pursuit['DF Type'].replace('BLI1',
                                                1)  # Replacing all the string type with integers(still of string data type), to check all the object type of a particular parameter use dataframe_name['param_name'].value_counts()
pursuit['DF Type'] = pursuit['DF Type'].replace('ADF', 2)
pursuit['DF Type'] = pursuit['DF Type'].replace('BLI2', 3)
pursuit['DF Type'] = pursuit['DF Type'].replace('ADF+BLI2', 4)
pursuit['DF Type'] = pursuit['DF Type'].replace('ADF+BLI1', 5)
pursuit['DF Type'].astype(int)  # Changing all the numbers(of string type) to integer type

pursuit['Frequency Agility'] = pursuit['Frequency Agility'].replace('Fixed Frequency', 1)  # Same as DF Type
pursuit['Frequency Agility'] = pursuit['Frequency Agility'].replace('LFM', 2)
pursuit['Frequency Agility'] = pursuit['Frequency Agility'].replace('LFM (Chirp-Up)', 3)
pursuit['Frequency Agility'] = pursuit['Frequency Agility'].replace('Frequency Agile', 4)
pursuit['Frequency Agility'] = pursuit['Frequency Agility'].replace("LFM (Chirp-Up) / Frequency Agile", 5)
pursuit['Frequency Agility'] = pursuit['Frequency Agility'].replace('LFM / Frequency Agile', 6)
pursuit['Frequency Agility'].astype(int)  # Changing all the numbers(of string type) to integer type

pursuit['Pulse Agility'] = pursuit['Pulse Agility'].replace('Fixed PRI', 1)  # Same as DF Type
pursuit['Pulse Agility'] = pursuit['Pulse Agility'].replace('Fixed Frequency', 2)
pursuit['Pulse Agility'] = pursuit['Pulse Agility'].replace('Stagger PRI', 3)
pursuit['Pulse Agility'] = pursuit['Pulse Agility'].replace('Dwell&Switch PRI', 4)
pursuit['Pulse Agility'] = pursuit['Pulse Agility'].replace('Frequency Agile', 5)
pursuit['Pulse Agility'] = pursuit['Pulse Agility'].replace('LFM', 6)
pursuit['Pulse Agility'].astype(int)  # Changing all the numbers(of string type) to integer type

pursuit['TOLA'] = pursuit['TOLA'].str.replace(' ', '').str.split(':').apply(
    lambda x: int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2]))
# Changing the time in TOLA and TOFA(originally in string type format of HH:MM:SS to seconds)
pursuit['TOLA'] = pursuit['TOLA'].astype(int)  # Changing all the seconds(of string type) to integer type
pursuit['TOFA'] = pursuit['TOFA'].str.replace(' ', '').str.split(':').apply(
    lambda x: int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2]))
pursuit['TOFA'] = pursuit['TOFA'].astype(int)

pursuit['TD'] = (pursuit['TOLA'] - pursuit['TOFA'])
# Dividing the pursuit data into (T1-T4)
pursuit_t1 = pursuit.drop(pursuit[pursuit.WeaponSystem != 'T1'].index)  # Pursuit with only T1
pursuit_t2 = pursuit.drop(pursuit[pursuit.WeaponSystem != 'T2'].index)  # Pursuit with only T2
pursuit_t3 = pursuit.drop(pursuit[pursuit.WeaponSystem != 'T3'].index)  # Pursuit with only T3
pursuit_t4 = pursuit.drop(pursuit[pursuit.WeaponSystem != 'T4'].index)  # Pursuit with only T4

pursuit = pursuit.select_dtypes(exclude='object')  # Excluding all the object types
pursuit = pursuit.drop(['Emitter Name'], axis=1)  # Excluding the Emitter name as it is an empty column

pursuit_t1 = pursuit_t1.select_dtypes(exclude='object')  # Excluding all the object types
pursuit_t1 = pursuit_t1.drop(['Emitter Name'], axis=1)  # Excluding the Emitter name as it is an empty column
pursuit_t2 = pursuit_t2.select_dtypes(exclude='object')  # Excluding all the object types
pursuit_t2 = pursuit_t2.drop(['Emitter Name'], axis=1)  # Excluding the Emitter name as it is an empty column
pursuit_t3 = pursuit_t3.select_dtypes(exclude='object')  # Excluding all the object types
pursuit_t3 = pursuit_t3.drop(['Emitter Name'], axis=1)  # Excluding the Emitter name as it is an empty column
pursuit_t4 = pursuit_t4.select_dtypes(exclude='object')  # Excluding all the object types
pursuit_t4 = pursuit_t4.drop(['Emitter Name'], axis=1)  # Excluding the Emitter name as it is an empty column

# The same steps are repeated for clutter
clutter['DF Type'] = clutter['DF Type'].replace('BLI1', 1)
clutter['DF Type'] = clutter['DF Type'].replace('ADF', 2)
clutter['DF Type'] = clutter['DF Type'].replace('BLI2', 3)
clutter['DF Type'] = clutter['DF Type'].replace('ADF+BLI2', 4)
clutter['DF Type'] = clutter['DF Type'].replace('ADF+BLI1', 5)
clutter['DF Type'].astype(int)

clutter['Frequency Agility'] = clutter['Frequency Agility'].replace('Fixed Frequency', 1)
clutter['Frequency Agility'] = clutter['Frequency Agility'].replace('LFM', 2)
clutter['Frequency Agility'] = clutter['Frequency Agility'].replace('LFM (Chirp-Up)', 3)
clutter['Frequency Agility'] = clutter['Frequency Agility'].replace('Frequency Agile', 4)
clutter['Frequency Agility'] = clutter['Frequency Agility'].replace("LFM (Chirp-Up) / Frequency Agile", 5)
clutter['Frequency Agility'] = clutter['Frequency Agility'].replace('LFM / Frequency Agile', 6)
clutter['Frequency Agility'].astype(int)

clutter['Pulse Agility'] = clutter['Pulse Agility'].replace('Fixed PRI', 1)
clutter['Pulse Agility'] = clutter['Pulse Agility'].replace('Fixed Frequency', 2)
clutter['Pulse Agility'] = clutter['Pulse Agility'].replace('Stagger PRI', 3)
clutter['Pulse Agility'] = clutter['Pulse Agility'].replace('Dwell&Switch PRI', 4)
clutter['Pulse Agility'] = clutter['Pulse Agility'].replace('Frequency Agile', 5)
clutter['Pulse Agility'] = clutter['Pulse Agility'].replace('LFM', 5)
clutter['Pulse Agility'].astype(int)

clutter['TOLA'] = clutter['TOLA'].str.replace(' ', '').str.split(':').apply(
    lambda x: int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2]))
clutter['TOLA'] = clutter['TOLA'].astype(int)
clutter['TOFA'] = clutter['TOFA'].str.replace(' ', '').str.split(':').apply(
    lambda x: int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2]))
clutter['TOFA'] = clutter['TOFA'].astype(int)
clutter['TD'] = (clutter['TOLA'] - clutter['TOFA'])

clutter = clutter.select_dtypes(exclude='object')
clutter = clutter.drop(['Emitter Name'], axis=1)
print('Dataset processing done !')
print('----------------------------------------------------------------------------------------')
pursuit_test = pursuit.drop('Target', axis=1)  # Making a new dataframe with all columns except the 'Target'
pursuit_res = pursuit[
    'Target']  # Making a new dataframe with only 'Target' columns, which can be used in checking the final result of the model to the exact value
pursuit_t1_test = pursuit_t1.drop('Target', axis=1)  # Same steps for T1
pursuit_t1_res = pursuit_t1['Target']
pursuit_t2_test = pursuit_t2.drop('Target', axis=1)  # Same steps for T2
pursuit_t2_res = pursuit_t2['Target']
pursuit_t3_test = pursuit_t3.drop('Target', axis=1)  # Same steps for T3
pursuit_t3_res = pursuit_t3['Target']
pursuit_t4_test = pursuit_t4.drop('Target', axis=1)  # Same steps for T4
pursuit_t4_res = pursuit_t4['Target']
print(f'Size of Pursuit T1:{len(pursuit_t1)}')  # Checking the length of rows in T1
print(f'Size of Pursuit T2:{len(pursuit_t2)}')  # Checking the length of rows in T2
print(f'Size of Pursuit T3:{len(pursuit_t3)}')  # Checking the length of rows in T3
print(f'Size of Pursuit T4:{len(pursuit_t4)}')  # Checking the length of rows in T4
print('----------------------------------------------------------------------------------------')
train_size = int(len(pursuit_t3) * 0.8)  # For the training dataset the Top-80 % of T3 data is used
X_train = pursuit_t3_test[:train_size]  # From the 80 % T3 training dataset, with all columns except the 'Target'
y_train = pursuit_t3_res[:train_size]  # From the 80 % T3 training dataset, only the 'Target' column
X_test = pursuit_t3_test[train_size:]  # From the remaining 20 % T3 testing dataset, with all columns except the 'Target
y_test = pursuit_t3_res[train_size:]  # From the remaining 20 % T3 testing dataset, only the 'Target' column

y_train_np = np.array(y_train)  # Converting the dataframe to array type using numpy
num_ones_train = (y_train_np == 1).sum()  # Checking all the 1's in the training dataset using lambda function
print('Dataset Info: Top-80%(as in the file) of T3 data in training set')
print('----------------------------------------------------------------------------------------')
print(f'Total Number of Ones in Target Column of the training dataset: {num_ones_train}')
print(f'Total Number of Zeros in Target Column of the training dataset: {len(y_train) - num_ones_train}')
y_test_np = np.array(y_test)  # Converting the dataframe to array type using numpy
num_ones_test = (y_test_np == 1).sum()  # Checking all the 1's in the testing dataset using lambda function
print(f'Total Number of Ones in Target Column of the test dataset: {num_ones_test}')
print(f'Total Number of Zeros in Target Column of the test dataset: {len(y_test) - num_ones_test}')
num_ones_test + num_ones_train, (len(y_test) - num_ones_test) + (
            len(y_train) - num_ones_train)  # Can be used to check size of T3
print('----------------------------------------------------------------------------------------')
# One-Class SVM
model_svm_01 = OneClassSVM(kernel='poly', gamma=0.5, nu=0.9)  # Declaration of the One-class Support Vector Machine(SVM)

model_svm_01.fit(X_train, y_train)  # Putting the model to training

# Predict on the test data
y_pred_svm = model_svm_01.predict(X_test)  # Now predict the test values and store in variable y_pred_svm
# Convert the predictions to 1 (anomaly) and 0 (normal)
y_pred_svm = np.where(y_pred_svm == -1, 1, 0)  # Convert the -1's to 0's and 0's to 1's

# Calculate and print the accuracy
accuracy_svm = accuracy_score(y_test,
                              y_pred_svm)  # Now check the model accuracy by comparing the actual target values to predicted values
print(
    f"Accuracy of One-Class SVM on remaining 20% data of T3: {accuracy_svm}")  # Print the accuracy of the model on test data
print('----------------------------------------------------------------------------------------')
# Predict on the test data
y_pred_svm = model_svm_01.predict(pursuit_t1_test)  # Now test the model on T1

# Convert the predictions to 1 (anomaly) and 0 (normal)
y_pred_svm = np.where(y_pred_svm == -1, 1, 0)  # Convert the -1's to 0's and 0's to 1's

# Calculate and print the accuracy
accuracy_svm = accuracy_score(pursuit_t1_res,
                              y_pred_svm)  # Now check the model accuracy by comparing the actual target values to predicted values
print(f"Accuracy of One-Class SVM on T1: {accuracy_svm}")  # Print the accuracy of the model on test data

# Predict on the test data
y_pred_svm = model_svm_01.predict(pursuit_t2_test)  # Similarly for T2 to T4

# Convert the predictions to 1 (anomaly) and 0 (normal)
y_pred_svm = np.where(y_pred_svm == -1, 1, 0)

# Calculate and print the accuracy
accuracy_svm = accuracy_score(pursuit_t2_res, y_pred_svm)
# print(f'Prediction:', y_pred_svm)
print(f"Accuracy of One-Class SVM on T2: {accuracy_svm}")

# Predict on the test data
y_pred_svm = model_svm_01.predict(pursuit_t3_test)

# Convert the predictions to 1 (anomaly) and 0 (normal)
y_pred_svm = np.where(y_pred_svm == -1, 1, 0)

# Calculate and print the accuracy
accuracy_svm = accuracy_score(pursuit_t3_res, y_pred_svm)
# print(f'Prediction:', y_pred_svm)
print(f"Accuracy of One-Class SVM on T3: {accuracy_svm}")

# Predict on the test data
y_pred_svm = model_svm_01.predict(pursuit_t4_test)

# Convert the predictions to 1 (anomaly) and 0 (normal)
y_pred_svm = np.where(y_pred_svm == -1, 1, 0)

# Calculate and print the accuracy
accuracy_svm = accuracy_score(pursuit_t4_res, y_pred_svm)
# print(f'Prediction:', y_pred_svm)
print(f"Accuracy of One-Class SVM on T4: {accuracy_svm}")

clutter_X = clutter.drop("Target", axis=1)  # Making a new dataframe with all columns except the 'Target'
clutter_y = clutter[
    "Target"]  # Making a new dataframe with only 'Target' columns, which can be used in checking the final result of the model to the exact value

# Predict on the test data
y_pred_svm = model_svm_01.predict(clutter_X)  # Store the predicted values in y_pred_svm

# Convert the predictions to 1 (anomaly) and 0 (normal)
y_pred_svm = np.where(y_pred_svm == -1, 1, 0)
# Calculate and print the accuracy
accuracy_svm = accuracy_score(clutter_y,y_pred_svm)  # Now check the model accuracy by comparing the actual target values to predicted values
clutter['Target'] = y_pred_svm  # Print the accuracy of the model on test data
clutter_X_arr = np.array(clutter_X)
print('----------------------------------------------------------------------------------------')
print(f"Performance of the model on Clutter: {accuracy_svm}")
j = 0
for i in range(len(clutter_X_arr)):  # Compute the total Signals of interest from clutter
    if y_pred_svm[i] == 1:
        j = j + 1

print('----------------------------------------------------------------------------------------')
print("Total Signals of interest determined from Clutter", j)
clutter['Target'].value_counts()
print('----------------------------------------------------------------------------------------')
# clutter.to_csv('nu_94.csv', index=False), generate the csv file for the clutter with Target=1, Target=0
