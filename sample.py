Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@Mainakkundu 
Mainakkundu
/
ab_testing
1
00
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
ab_testing/app.py /
@Mainakkundu
Mainakkundu Updated app with streamlit functions
Latest commit 880008e on 14 Apr
 History
 1 contributor
433 lines (376 sloc)  18.7 KB
  
import pandas as pd
import numpy as np
import streamlit as st
import datetime
import datetime as dt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
import os
from statsmodels.tsa.arima_process import ArmaProcess
from causalimpact import CausalImpact
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
from causalimpact import CausalImpact
from statsmodels.formula.api import ols
from PIL import Image
import statistics
warnings.filterwarnings("ignore")
rcParams['figure.figsize'] = 20,10
rcParams['font.size'] = 30
sns.set()
np.random.seed(8)

def min_detectable_data_prep(base_mean,base_std,min_detectable_change):
  '''
  >>> Prepare the data to incorporate the min-detectable changes which will
  pass into sample_size_calculator
  '''
  mu_final_l = []
  df_f = pd.DataFrame()
  for i in min_detectable_change:
    mu = base_mean+(base_mean*i)
    mu_final_l.append(mu)
    fd = pd.DataFrame()
    #fd['business_metric'] = [business_metric]
    fd['mu_base'] = [base_mean]
    fd['std_base'] =[base_std]
    fd['detectable_effect'] = [i]
    fd['mu_hat'] = [mu]
    df_f = df_f.append(fd)
    return (df_f)

def sample_size_calculator(mu_base,mu_hat,std_base,alpha=0.05,power=0.8):
    '''
    >>> Sample size calculation for Hypothesis Testing
    '''
    from math import sqrt
    from statsmodels.stats.power import TTestIndPower
    mean0 = mu_base
    mean1 = mu_hat
    std = std_base

    cohens_d = (mean0 - mean1) / (sqrt((std ** 2 + std ** 2) / 2))

    effect = cohens_d
    #alpha = 0.05
    #power = 0.8

    analysis = TTestIndPower()
    sample_size=analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
    return(int(sample_size))

def week_flg(pos_busn_dt,pre_prd_start='2020-07-06',pre_prd_end='2020-10-26',post_prd_strt='2020-11-02',post_prd_end='2020-11-29'):
    '''
    >>> Take input from streamlit app dates for pre & post
    '''
    if (pre_prd_end >= str(pos_busn_dt) >= pre_prd_start):
        return 'pre'
    elif(post_prd_end >= str(pos_busn_dt) >= post_prd_strt):
        return 'post'

def metric_data_prep(df_merg,KPI_original='tot_bask_spend',primary_key='card_code',KPI_rename='sales'):
    '''
    >>> Prepare the Base metric data
    '''
    df_merg['week_flg'] = np.vectorize(week_flg)(df_merg['pos_busn_dt'])
    df_merg['week_flg'] = df_merg['week_flg'].fillna(0)
    g = df.groupby([primary_key,'week_flg']).agg({KPI:['sum']})
    g.rename(columns = {KPI_original:KPI_rename}, inplace = True)
    g_pvt = pd.pivot(g,columns='week_flg',values=KPI_rename,index=primary_key)
    g_pvt =g_pvt.fillna(0)
    g_pvt = pd.DataFrame(g_pvt)
    g_pvt['post_'+KPI_rename] =  g_pvt['post']
    g_pvt['pre_'+KPI_rename] = g_pvt['pre']
    del g_pvt['post']
    del g_pvt['pre']
    del g_pvt[0]
    return g_pvt



def r2(x, y):
    '''
    >>> Stronger the relationship high the R-square
    '''
    return stats.pearsonr(x, y)[0] ** 2

def CUPED(df,KPI):
    '''
    >>> Microsoft's classical approcah to reduce the variance and increase the power
    'https://frankhopkins.github.io/py_ab_fh/5_Variance_Reduction_Methods.html'
    '''
    pre_experiment_KPI = KPI+'_pre_experiment'
    covariance = np.cov(df[KPI], df[pre_experiment_KPI])
    variance = np.cov(df[pre_experiment_KPI])
    theta_calc = covariance / variance
    theta_calc_reshape = theta_calc.reshape(4,1)
    theta = theta_calc_reshape[1]
    print(theta)
    df['CUPED-adjusted_metric'] = df[KPI] - (df[pre_experiment_KPI] - statistics.mean(df[pre_experiment_KPI])) * theta

    std_pvs = statistics.stdev(df[KPI])
    std_CUPED = statistics.stdev(df['CUPED-adjusted_metric'])
    mean_pvs = statistics.mean(df[KPI])
    mean_CUPED = statistics.mean(df['CUPED-adjusted_metric'])

    relative_pvs = std_pvs / mean_pvs
    relative_cuped = std_CUPED / mean_CUPED
    relative_diff = (relative_cuped - relative_pvs) / relative_pvs

    print("The mean of the KPI  is %s."
    % round(mean_pvs,4),
    "The mean of the CUPED-adjusted KPI is % s."
    % round(mean_CUPED,4))


    print ("The standard deviation of KPI is % s."
        % round(std_pvs,4),
          "The standard deviation of the CUPED-adjusted KPI is % s."
           % round(std_CUPED,4))

    print("The relative reduction in standard deviation was % s"
        % round(relative_diff*100,5),"%",)

    return df


def t_distribution_ci(df,metric='post_sales_temp',control='Control',test='Test_1',test_flag = 'test_flag',alpha=0.05):

  signi = []
  p_value = []

  test_data_A = df[df[test_flag]==control]
  test_data_B = df[df[test_flag]==test]
  test_data_A[metric] = test_data_A[metric].astype('float')
  test_data_B[metric] = test_data_B[metric].astype('float')
  print(test_data_A[metric].quantile(.995))
  #test_data_A_clean = test_data_A[(test_data_A[metric]>0) & (test_data_A[metric]<test_data_A[metric].quantile(.995))]
  test_data_A_clean = test_data_A
  print(test_data_B[metric].quantile(.995))
  #test_data_B_clean = test_data_B[(test_data_B[metric]>0) & (test_data_B[metric]<test_data_B[metric].quantile(.995))]
  test_data_B_clean = test_data_B
  #Combine the cleaned data sets as one
  test_data_clean = test_data_A_clean.append(test_data_B_clean)
  #Summarize the metrics:- Calculating totals
  test_summary1 = test_data_clean.groupby(test_flag).agg({
      metric:'sum'

  })
  #Summarize the metrics:- Calculating means
  test_summary2 = test_data_clean.groupby(test_flag).agg({
            metric:'mean'
  })
  #Transposing the summaries
  test_summary1 = test_summary1.T
  test_summary2 = test_summary2.T

  #Initialize a dataframe with test stats
  test_stats = pd.DataFrame(columns = ['pct_lft','conf_int_lb','conf_int_ub','p-value'])
  #Concatenate the test stats with both the summaries
  test_summary1 = pd.concat([test_summary1,test_stats],axis=1,ignore_index=False,sort=False)
  #Calculate pct_lift for all the metrics
  test_summary1['pct_lft'] = (test_summary1[test]-test_summary1[control])/test_summary1[control]*100
  test_summary2 = pd.concat([test_summary2,test_stats],axis=1,ignore_index=False,sort=False)
  #Calculate pct_lift for all the metrics
  test_summary2['pct_lft'] = (test_summary2[test]-test_summary2[control])/test_summary2[control]*100

  cm = sms.CompareMeans(sms.DescrStatsW(test_data_A_clean[metric][test_data_A_clean[metric].notnull()]),
			sms.DescrStatsW(test_data_B_clean[metric][test_data_B_clean[metric].notnull()]))
  lb,rb = cm.tconfint_diff(usevar='unequal',alternative='two-sided',alpha = 0.10)

  test_summary2['conf_int_lb'] = (rb*-1)/test_data_A_clean[metric].mean()
  test_summary2['conf_int_ub']=  (lb*-1)/test_data_A_clean[metric].mean()

  t_stat,test_summary2['p-value'] = sc.ttest_ind(test_data_A_clean[metric][test_data_A_clean[metric].notnull()],
               				test_data_B_clean[metric][test_data_B_clean[metric].notnull()],equal_var = False)



  if (test_summary2['p-value'].iloc[0] < alpha) and (test_summary2['pct_lft'].iloc[0]  > 0):
    signi.append('Significant with lift')
  elif (test_summary2['p-value'].iloc[0] < alpha) and (test_summary2['pct_lft'].iloc[0] < 0):
    signi.append('Significanct ,control performance better than test')
  elif (test_summary2['p-value'].iloc[0] > alpha) and (test_summary2['pct_lft'].iloc[0] < 0):
    signi.append('Not significanct with negative lift')
  elif(test_summary2['p-value'].iloc[0] > alpha) and (test_summary2['pct_lft'].iloc[0] > 0):
    signi.append('Not significant with positive lift')
  else:
    signi.append('Nothing')

  print(signi)

  test_summary2['sigificance'] = signi
  return test_summary2

def mann_whitney_u_test(df,metric='pos_sales_temp',control='Control',test='Test',test_flag = 'test_flag',alpha=0.1):
    """
    Perform the Mann-Whitney U Test, comparing two different distributions.
    Args:
       distribution_1: List. First distribution that we want to compare.
       distribution_2: List. Second distribution that we want to compare.
    Outputs:
        u_statistic: Float. U statisitic for the test.
        p_value: Float. p-value for the test.
    """
    signi = []
    test_data_A = df[df['test_flg']==control]
    test_data_B = df[df['test_flg']==test]
    test_data_A[metric] = test_data_A[metric].astype('float')
    test_data_B[metric] = test_data_B[metric].astype('float')
    test_data_A_clean = test_data_A
    test_data_B_clean = test_data_B
    test_data_clean = test_data_A_clean.append(test_data_B_clean)
    test_summary1 = test_data_clean.groupby('test_flg').agg({
      metric:'sum'

  })
    #Summarize the metrics:- Calculating means
    test_summary2 = test_data_clean.groupby('test_flg').agg({
            metric:'mean'
  })
  #Transposing the summaries
    test_summary1 = test_summary1.T
    test_summary2 = test_summary2.T

   #Initialize a dataframe with test stats
    test_stats = pd.DataFrame(columns = ['pct_lft','u_statistic','p-value'])
   #Concatenate the test stats with both the summaries
    test_summary1 = pd.concat([test_summary1,test_stats],axis=1,ignore_index=False,sort=False)
   #Calculate pct_lift for all the metrics
    test_summary1['pct_lft'] = (test_summary1[test]-test_summary1[control])/test_summary1[control]*100
    test_summary2 = pd.concat([test_summary2,test_stats],axis=1,ignore_index=False,sort=False)
    #Calculate pct_lift for all the metrics
    test_summary2['pct_lft'] = (test_summary2[test]-test_summary2[control])/test_summary2[control]*100
    u_statistic, p_value = stats.mannwhitneyu(test_data_A_clean[metric], test_data_B_clean[metric])
    test_summary2['p-value'] = p_value
    test_summary2['u_statistic'] = u_statistic
    if (test_summary2['p-value'].iloc[0] < alpha) and (test_summary2['pct_lft'].iloc[0]  > 0):
        signi.append('Significant with lift')

    elif (test_summary2['p-value'].iloc[0] < alpha) and (test_summary2['pct_lft'].iloc[0] < 0):
         signi.append('Significanct ,control performance better than test')

    elif (test_summary2['p-value'].iloc[0] > alpha) and (test_summary2['pct_lft'].iloc[0] < 0):
        signi.append('Not significanct with negative lift')

    elif(test_summary2['p-value'].iloc[0] > alpha) and (test_summary2['pct_lft'].iloc[0] > 0):
        signi.append('Not significant with positive lift')

    else:
        signi.append('Nothing')


    print(signi)

    test_summary2['sigificance'] = signi

    return test_summary2



def main():
    st.title("""AB-Testing Tool """)
    html_temp = """
    <div style="background-color:orange;padding:10px">
    <h2 style="color:black;text-align:center;">Online Marketing Campaigns</h2>
    </div>
    """

    print('----Data Imports ------')
    df = pd.read_csv('./main/streamlit/data/fake_data.csv') ###1
    cup_df = pd.read_csv('./main/streamlit/data/fake_data_cuped.csv') ###2

    test_f = {'Control_Matrix':'Control', 'Variant_BT':'Test'}

    df['test_flag'] = df['Variant'].replace(test_f)
    cup_df['test_flag'] = cup_df['Variant'].replace(test_f)
    test = df[df['test_flag']=='Test']
    control = df[df['test_flag']=='Control']

    test_cuped = cup_df[cup_df['test_flag']=='Test']
    control = cup_df[cup_df['test_flag']=='Control']


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

    print('======================================================')
    print('----------- Sample Size Estimation--------------------')
    print('======================================================')
    st.markdown(html_temp,unsafe_allow_html=True)
    detectable_change = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    MENU = ['Sample-Size-Estimation','Stat Base Measurement','Analysis & Recommendation']

    choice = st.sidebar.radio(''' Click here ''', MENU)
    if choice == 'Sample-Size-Estimation':
        mean_sales = st.sidebar.number_input('Base-Mean',1)
        std_sales = st.sidebar.number_input('Base-StdDev',1)
        alpha = st.sidebar.number_input('Alpha_Value',0.05)
        power = st.sidebar.number_input('Power_Value',0.8)
        k=min_detectable_data_prep(mean_sales,std_sales,detectable_change)
        k['require_sample_size']=np.vectorize(sample_size_calculator)(k['mu_base'],k['mu_hat'],k['std_base'])
        st.subheader('Sample Sizes for different scenario of Minimum Detectable Effect')
        st.write("""
                    Enter your  data into the sidebar and choose what will be Base Mean of KPI & Base Std Deviation of KPI.
                    Below table shows the different sample sizes for different MDE(Minimum detectable Effect) """)
        st.dataframe(k)
        k['effect_in_%'] =(k['detectable_effect']*100)
        sns.pointplot(x=k['effect_in_%'],y=k['require_sample_size'],color='blue',)
        st.pyplot()
    elif choice=='Stat Base Measurement':
        METRIC  = st.sidebar.selectbox('Choose the metric', ['Pvs_per_session'])
        METHOD = st.sidebar.selectbox('Choose the method', ['Post (Control) Vs Post (Test)','Pre (Test) Vs Post(Test)','CUPED','Post (Control) Vs Post (Test) NonParametric'])
        if METHOD == 'Post (Control) Vs Post (Test)':
            print('---Step-1:Distribution Plot---')
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
            st.subheader('Mean comparison between Test & Control Distribution using Welsh T-Test')
            r = t_distribution_ci(df,metric=METRIC,control='Control',test='Test',alpha=0.05)
            st.dataframe(r)
            if r['p-value'].iloc[0] > 0.1:
                st.markdown('''### Inference ''')
                st.write('''According to the null hypothesis, there is no difference between the means.
        The plot above shows the distribution of the difference of the means that
        we would expect under the null hypothesis.''')
            else:
                st.markdown('''### Inference ''')
                st.write('''According to the null hypothesis, there is siginificant difference between the means.
        The plot above shows the distribution of the difference of the means that
        we would expect under the null hypothesis.''')

        elif METHOD =='Pre (Test) Vs Post(Test)':
            figsize = (20, 6)
            ci = CausalImpact(pre_post_data, pre_period, post_period)
            print(ci.summary())
            print(ci.summary(output='report'))
            pre_post_report = ci.summary_data
            pre_post_report['p_value'] = ci.p_value
            pre_post_report['siginificance'] = np.where(pre_post_report['p_value']>0.1,'Not Significant','Significant')
            st.subheader('Causal Inference Analysis')
            ci.plot()
            st.pyplot()
            st.subheader('Causal Inference statistical output')
            st.write(ci.summary(output='report'))
            st.dataframe(pre_post_report)
        elif METHOD =='CUPED':
            cup_df =CUPED(cup_df,KPI=METRIC)
            test_cuped = cup_df[cup_df['test_flag']=='Test']
            control_cuped = cup_df[cup_df['test_flag']=='Control']
            cup_r = t_distribution_ci(cup_df,metric='CUPED-adjusted_metric',control='Control',test='Test',alpha=0.05)
            cor_df = cup_r.corr()
            st.subheader('Pre Vs Post Correlation to understand Variance')
            sns.jointplot(cup_df[METRIC],cup_df[METRIC+'_pre_experiment'],kind="reg",stat_func=r2)
            st.pyplot()
            ax1 = sns.distplot(test_cuped['CUPED-adjusted_metric'],hist=False,kde=True)
            ax2 = sns.distplot(control_cuped['CUPED-adjusted_metric'],hist=False,kde=True)
            plt.axvline(np.mean(test_cuped['CUPED-adjusted_metric']), color='b', linestyle='dashed', label='TEST',linewidth=5)
            plt.axvline(np.mean(control_cuped['CUPED-adjusted_metric']), color='orange', linestyle='dashed',label='CONTROL', linewidth=5)
            plt.legend(labels=['TEST','CONTROL'])
            st.subheader('CUPED-Distribution Comparison(Density Plot) after removing variance')
            st.pyplot()
            st.subheader('CUPED-Mean comparison between Test & Control Distribution using Welsh T-Test after removing variance')
            st.dataframe(cup_r)
        elif METHOD == 'Post (Control) Vs Post (Test) NonParametric':
            print('---Step-1:Distribution Plot---')
            plt.figure()
            ax1 = sns.distplot(test[METRIC],hist=False,kde=True)
            ax2 = sns.distplot([METRIC],hist=False,kde=True)
            plt.axvline(np.mean(test[METRIC]), color='b', linestyle='dashed', label='TEST',linewidth=5)
            plt.axvline(np.mean([METRIC]), color='orange', linestyle='dashed',label='CONTROL', linewidth=5)
            plt.legend(labels=['TEST','CONTROL'])
            st.subheader('Distribution Comparison(Density Plot)')
            st.pyplot()
            sns.boxplot(data=[test[METRIC],[METRIC]],showmeans=True)
            st.subheader('Distribution Comparison(Box Plot)')
            st.pyplot()
            print('--Step-2:T-Test for Mean Comparison--')
            st.subheader('Mean comparison between Test & Control Distribution using Welsh T-Test')
            df[METRIC] = df[METRIC].astype('float')
            r = mann_whitney_u_test(df,metric=METRIC,control='Control',test='Test',test_flag='test_flag',alpha=0.05)
            st.dataframe(r)
            if r['p-value'].iloc[0] > 0.1:
                st.markdown('''### Inference ''')
                st.write('''According to the null hypothesis, there is no difference between the means.
        The plot above shows the distribution of the difference of the means that
        we would expect under the null hypothesis.''')
            else:
                st.markdown('''### Inference ''')
                st.write('''According to the null hypothesis, there is siginificant difference between the means.
        The plot above shows the distribution of the difference of the means that
        we would expect under the null hypothesis.''')

if __name__ == '__main__':
	main()
