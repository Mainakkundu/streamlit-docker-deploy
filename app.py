import pandas as pd
import numpy as np
import streamlit as st
import datetime
import datetime as dt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
import os
from statsmodels.tsa.arima_process import ArmaProcess
#from causalimpact import CausalImpact
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pylab import rcParams
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
import textwrap
from numpy import nansum
from numpy import nanmean
import pandas as pd
import statsmodels.stats.api as sms
from scipy import stats as sc
#from causalimpact import CausalImpact
from statsmodels.formula.api import ols
from PIL import Image
import statistics
warnings.filterwarnings("ignore")
rcParams['figure.figsize'] = 20,10
rcParams['font.size'] = 30
sns.set()
np.random.seed(8)
import utility

### Data for Normal analysis 
cup_df = pd.read_csv('cuped_data.csv') 
## Data for Causal Inferene 
np.random.seed(12345)
ar = np.r_[1, 0.9]
ma = np.array([1])
arma_process = ArmaProcess(ar, ma)
X = 100 + arma_process.generate_sample(nsample=100)
y = 1.2 * X + np.random.normal(size=100)
y[70:] += 5
pre_post_data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X']) ###3
pre_period = [0, 69]
post_period = [70, 99]

st.title("""AB-Testing Tool """)
print('======================================================')
print('----------- Sample Size Estimation--------------------')
print('======================================================')
MENU = ['Sample-Size-Estimation','Stat Base Measurement','Analysis & Recommendation']
choice = st.sidebar.radio(''' Click here ''', MENU)
if choice == 'Sample-Size-Estimation':
    mean_sales = st.sidebar.number_input('Base-Mean',1)
    std_sales = st.sidebar.number_input('Base-StdDev',1)
    expected_lift = st.sidebar.number_input('Expected-Lift',1)
    alpha = st.sidebar.number_input('Alpha_Value',0.05)
    power = st.sidebar.number_input('Power_Value',0.8)
    avg_footfall_per_day = st.sidebar.number_input('Avg_foot_fall_per_day',100)
    pre_exp = utility.PreExpirement(alpha=alpha,power=power)
    sample_size = pre_exp.sample_size_calculator(MU_BASE=mean_sales,EXPECTED_LIFT =expected_lift,STD_DEV=std_sales)
    st.text(f"Sample Sizes Estimate as per choosen parameters is {sample_size} customers" )
    #st.subheader(sample_size)
    no_of_days = int(pre_exp.duration_calculation(sample_size,TRAFFIC_PER_DAY=avg_footfall_per_day))
    st.text(f"Need to run the test atleast {no_of_days} days , to attain seasonality")
    #st.subheader(sample_size)
elif choice=='Stat Base Measurement':
    df = pd.read_csv('cuped_data.csv')
    st.write(df.head())
    METRIC  = st.sidebar.selectbox('Choose the metric', ['page_views'])
    METHOD = st.sidebar.selectbox('Choose the method', ['Post (Control) Vs Post (Test)','Pre (Test) Vs Post(Test)','CUPED','Post (Control) Vs Post (Test) NonParametric'])
    engine = utility.StatsEngine(df=df) ## intantiate the class 
    if METHOD == 'Post (Control) Vs Post (Test)':
        test = df[df['Variant']=='test']
        control = df[df['Variant']=='control']
        plt.figure()
        ax1 = sns.distplot(test[METRIC],hist=False,kde=True)
        ax2 = sns.distplot(control[METRIC],hist=False,kde=True)
        plt.axvline(np.mean(test[METRIC]), color='b', linestyle='dashed', label='TEST',linewidth=5)
        plt.axvline(np.mean(control[METRIC]), color='orange', linestyle='dashed',label='CONTROL', linewidth=5)
        plt.legend(labels=['TEST','CONTROL'])
        st.subheader('Distribution Comparison(Density Plot)')
        st.pyplot()
        sns.boxplot(data=[test[METRIC],control[METRIC]],showmeans=True)
        st.subheader('Distribution Comparison(Box Plot)')
        st.pyplot()
        print('--Step-2:T-Test for Mean Comparison--')
        st.subheader('Mean comparison between Test & Control Distribution using  T-Test')
        
        t_test = engine.t_distribution_ci(df,test_flag='Variant',control='control',test='test',metric=METRIC,period='post',alpha=0.05)
        st.dataframe(t_test)
        if t_test['p-value'].iloc[0] > 0.1:
            st.markdown('''### Inference ''')
            st.write('''According to the null hypothesis, there is no difference between the means.
            The plot above shows the distribution of the difference of the means that
            we would expect under the null hypothesis.''')
        else:
            st.markdown('''### Inference ''')
            st.write('''According to the null hypothesis, there is siginificant difference between the means.
            The plot above shows the distribution of the difference of the means that
            we would expect under the null hypothesis.''')
    elif METHOD == 'CUPED':
        pre_data = df['page_views'].values
        post_data = df['pre_page_views'].values
        cor_df = df[['page_views','pre_page_views']].corr()
        st.subheader('How Pre-Experiment data bias on Post-period Metric {Correlation-Plot}')
        #sns.set(rc={'figure.figsize':(9.7,8.27)})
        sns.jointplot(df['page_views'],df['pre_page_views'],kind="reg")
        st.pyplot()
        cuped_df=  engine.CUPED(KPI='page_views')
        test_cuped = cuped_df[cuped_df['Variant']=='test']
        control_cuped = cuped_df[cuped_df['Variant']=='control']
        plt.figure()
        ax1 = sns.distplot(test_cuped['CUPED-adjusted_metric'],hist=False,kde=True)
        ax2 = sns.distplot(control_cuped['CUPED-adjusted_metric'],hist=False,kde=True)
        plt.axvline(np.mean(test_cuped['CUPED-adjusted_metric']), color='b', linestyle='dashed', label='TEST',linewidth=5)
        plt.axvline(np.mean(control_cuped['CUPED-adjusted_metric']), color='orange', linestyle='dashed',label='CONTROL', linewidth=5)
        plt.legend(labels=['TEST','CONTROL'])
        st.subheader('CUPED-Distribution Comparison(Density Plot) after removing variance ')
        st.pyplot()
        cuped_t_test = engine.t_distribution_ci(cuped_df,test_flag='Variant',control='control',test='test',metric='page_views',period='CUPED',alpha=0.05)
        st.write(cuped_t_test)

           
    
    


