######### README FOR AISE - COMS6156 FINAL PROJECT by Dominick DeLucia (dmd2203) ##########


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



** Thanks Professor for teaching this course. I enjoyed learning from you. 