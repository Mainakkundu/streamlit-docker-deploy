# streamlit-docker-deploy

Streamlit App for measuring the effectiveness of A/B testing scenario in 3 different verticals.
This app focous on only continious KPI.

## Components:
1. Sample Size Estimation: This is the tab for pre-experiment analysis, where you can input base-mean,base-standard-deviation,average-traffic on your websites or store, and incremental lift you are anticipating and it will throw you the required sample size and duration for a test to go live. 

2. Stat Base Measurement: In this tab, you can measure the Test results is it significant or not.
  It can measure with 3 views:
  A.Control Vs Test: Measuring the incrementality between Test & Control group using T-Test and also tied the Confidence Intervals and p-value for significance. 
  
  B.CUPED: Microsoft's approach to reducing the variance and increase the power of the test using pre-experiment data. Much more applicable if a metric is highly correlated with pre-behavior. 

  C.Non Parametric Test: Measuring the incrementality between Test & Control group using Non-Parametric Method.

  D.Pre Vs Post: Measuring the effectiveness of Test after and before the intervention(offer day, campaign day, UI change any intentional change). This approach is highly recommended for the scenario where there is a no-hold out set aka Control group. This methodology will help to understand what is the cumulative incremental effect and relative effect on Pre and Post.            

3. Analysis & Recommendation: In this tab, you can measure the effectiveness of the Test on a day level which helps to understand the seasonality effects & novelty effect and how the overall test converges or not.                       




       
