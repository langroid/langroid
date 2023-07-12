kaggle_description = """
    While it's universally interesting to understand what methods were used by the 
    top participants (especially in this contest where there are some large gaps in 
    AUC at the top), I suspect that many others who participated also have clever 
    methods or insights.  While we wait for the top finishers to post on "No Free 
    Hunch", I thought it would be interesting to hear from anyone else who might wish 
    to share.  Many of the models are quite good and would produce better results 
    than the methods used by persons in industry.      

    My results (#15):

    Overall method: 

    randomForest() in R, 199 trees, min node size of 25, default setting for other 
    values 

    Sampling: 

    Used 10% of the training dataset to train the randomForest.  Also included any 
    data points that were within 500ms of a state change (where isalert shifted from 
    1 to 0 or vice-versa).  About 110,000 rows total.  

    Data Transformations: 

Tossed out correlated variables, such as p7 (inverse correlation with p6) and p4 (
inverse correlation with p3) 
Transformed p3 into an element of ["High", "Mid", "Low"] based on the probability of 
being alert.  Where p3 is an even multiple of 100, the probability of being alert is 
systematically higher.  Where "p3 mod 100" is 84, 16, or 32, there is also a greater 
chance of being alert ("Mid").  Call everything else "Low".   
The histogram of p5 clearly shows a bimodal distribution.  Transformed p5 into a 1/0 
indicator variable with a breakpoint at p5=0.1750. 
Transformed e7 and e8 to lump together all buckets greater than or equal to 4.
Transformed v11 into 20-tiles to convert strangely shaped distribution into a 
discrete variable. 

Tried and Denied:

Lagging values
Moving average


Color Commentary:

RandomForest's ability to "fit" the training data presented was very strong.  
However, the out-of-bucket (OOB) error rate, as reported by R, was highly misleading. 
The OOB error rate could be driven down to the 1-3% range.  However, those models 
produced somewhat worse results on a true out-of-sample validation set.  Keeping 
randomForest tuned to produce OOB error rates of 8-10% produced the best results in 
this case.   

Because many of the training cases are similar, randomForest performed better when 
using just a sample of the overall training data (hence the decision to train on only 
about 110,000 rows).  RandomForest also under-performed when the default nodesize (
either 1 or 5) was used.  The explicit adjustment of nodesize to other values, 
such as 10, 25, and 50, produced noticeably different error rates on true 
out-of-sample data.   
"""
