Notes on HW2...
Pull_data.py won't run unless you create your own userNamePW.csv file containing one row of : username,password
or if you modify the code in some other manner to get around authorization.

There is one other file called titanicCleaning.py.  This file is used by both the train_model.py and the score_model.py to do the necessary clean up and related tasks on the downloaded files.

The classification report is generated when executing the train_model.py file, rather than the score_model.py.  It is executed there on the test set of data (split off from the training dataset).  I did not see how one could run a classification report in the score_model.py module, against the test data from kaggle itself, in that we don't know what the actual Y values are...we only have the predicted values.

Critical Thinking HW2

i:  The process to download files I have created is an admittedly fragile process.  I am using what appears to be an undocumented function to login and download the file.  If the login process does change, a quick way to resolve the process is to login externally to Kaggle, and remove the login code (e.g., comment out).  The file location is an easier solve, if that changes, it is purely a text change to the python script and could be quickly updated.  If file content changes, that would only cause problems if the columns have changed...as opposed to the rows of data.  If additional columns are added they is shouldn't present an issue (they would be ignored) but removing or changing a column from numeric to text could cause the program to fail, and certainly would alter results.

ii: Running out of space or not having permission to save the file doesn't seem like a programmatic solve is necessary, although having a friendly error explaining why the program aborts obviously helps the user, so they can quickly locate their issue, and correct it so they can then rerun the script once they free space or run in a folder they have permission to do so.

iii:  Hopefully updated python packages are not an issue in that future versions work the same for existing functions, but that isn't always the case.  To avoid this once can package (e.g., in a docker image) the script with the needed python packages installed in the image at the specific version the program was originally created with.  That would make the python script immune to any upgrades to packages it uses.

iv: If one runs in a docker container and cannot connect to internet (for whatever reason), the program clearly will not execute, as it is not designed to work with purely local data.  Binding or routing issues would need to be corrected and then the program within docker re-executed.
