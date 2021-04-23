######### README FOR AISE - COMS6156 FINAL PROJECT by Dominick DeLucia (dmd2203) ##########

Overview: 
Running a user study on the usefulness of leading next-token prediction models with a focus on evaluating 
performance on major logical steps. Leading academic models boast high precision (~75%) on global token prediction
but real impact is likely much lower due to the non-uniform distribution of importance across tokens. 
This study was constructed such that various types of python development are examined for each 
model and their performance compared. 

How To Run: 
To run the academic models, go into the jupyter notebook and run with the desired code as text. 
To run the production software - I installed PyCharm and installed Kite and Tabnine 
(have to swap which is active, as only one can be on at one time) 


1. Academic_Model_API_Usage.ipynb - the simple file used to access both academic models
 through the huggingface API. I left some additional exploration code I used for checking
 the possible input types, but that was not important. The main code is very simple at the 
 top of the notebook. 

2. eval_code - This contains all of the source code files that I used for the user study. All files
are named according to their base file (included) and correspond to the rows in the Google Sheet below.

3. SummaryStats - Includes all of the results from my user study. All results were recorded in 
 a google sheet found here: 
https://docs.google.com/spreadsheets/d/1IE9cjc_EcBbL9xL_dcMYOFYU6ouE-s0OcIXZPAapcEI/edit?usp=sharing

4. Milestone_Documents - simply includes all milestones for this project as requested

5. CodeCompletion-token - This is a clone from part of the CodeXGLUE repo. This was the code
I utilized in order to "fine tune" the CodeGPT-adapted model (to no avail). Code works properly
with some futzing for CUDA version and some difficulty around ascii encoding (literally had to 
go into the encoding file it referenced at change encodings away from nonsensical ones and 
then it worked). The CodeXGLUE instructions for utilizing this code can be found here: 
https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token

*** NOTE: Had to remove CodeCompletion-token from this github push, too many filees and files were too large. 
Can refer to the above link, but this was a dead end for me and not actually used in the user study. 






~ Thanks Professor for teaching this course. I enjoyed learning from you. ~
