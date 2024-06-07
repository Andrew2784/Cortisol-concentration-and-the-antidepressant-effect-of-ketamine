import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import re
from scipy.stats import sem
import matplotlib.pyplot as plt


data_dict = {'Control_L_19_Day1_BFT':[2,3,4],'Mem_H_14_Day1_AFT':[4,5,6],'Control_L_12_Day1_BFT':[7,8,9],'Mem_H_11_Day1_AFT':[10,11,12],'Lac_HH_15_Day2_AFT':[11,12,13],'Lac_HH_21_Day2_AFT':[15,16,17],'Tra_LL_13_Day3_AFT':[15,16,17],'Tra_LL_14_Day3_AFT':[20,21,22],'Tra_LL_10_Day2_AFT':[3,4,5],'Tra_LL_04_Day2_AFT':[11,12,13]}
# data_dict = {'Group':np.repeat(['Control','Mem','Lan','Tra'],16),'Cortisol_Concentration':np.tile([np.repeat(['Lowest','Low','High','Highest'],4)],4).flatten(),'Day':np.tile(['Day1 BFT','Day1 AFT','Day2','Day3'],16),'Spikes':np.random.randint(1,20,64)}
# data_dict = {'Group':np.repeat(['Control','Mem','Lan','Tra'],16),'Cortisol_Concentration':np.tile([np.repeat(['Lowest','Low','High','Highest'],4)],4).flatten(),'Day':np.tile(['Day1 BFT','Day1 AFT','Day2','Day3'],16),'Spikes1':np.random.randint(1,20,64),'Spikes2':np.random.randint(1,30,64)}


combined_dict = {}
combined_datframe = pd.DataFrame()
for key in data_dict.keys():
    list_terms = key.split('_')
    pattern = '(?=.*%s)(?=.*%s)(?=.*%s)(?=.*%s).*' % (list_terms[0],list_terms[1],list_terms[3],list_terms[4])
    for key2 in data_dict.keys():
            if re.findall(pattern,key2) != []:
                temp_key = '_'.join([list_terms[0],list_terms[1],list_terms[3],list_terms[4]])
                if temp_key in combined_dict:
                    if data_dict[key][0] not in combined_dict[temp_key]:
                        combined_dict[temp_key].append(data_dict[key][0])
                        combined_datframe = combined_datframe.append({'Group':list_terms[0],'Cortisol_Concentration':list_terms[1],'Day':list_terms[3]+'_'+list_terms[4],'Spike1':combined_dict[temp_key][0],'Spike2':combined_dict[temp_key][1]},ignore_index=True)
                else:
                    combined_dict[temp_key] = [data_dict[key][0]] 


row_means = combined_datframe.mean(axis=1)
dfm = combined_datframe.melt(id_vars = ['Group','Cortisol_Concentration','Day'],value_name=('Spike'))



sns.set_theme()
sns.catplot(data=dfm, kind="bar", x="Day", y="Spike",hue='Day',row='Cortisol_Concentration',capsize=0.2,col='Group',row_order=['LL','L','H','HH']).set_xticklabels(['Day1 BFT','Day1 AFT','Day2','Day3'])

# %% Normalized graphs
dfmcopy = dfm.copy()
group_dat = dfmcopy.groupby(['Group','Cortisol_Concentration','Day'],as_index=False).mean()
group_dat_sem = dfmcopy.groupby(['Group','Cortisol_Concentration','Day'],as_index=False).sem()
normed_dat_col = group_dat.Spike / group_dat.groupby(['Group','Cortisol_Concentration']).Spike.transform(np.min)
normed_dat_col_sem = group_dat_sem.Spike / group_dat.groupby(['Group','Cortisol_Concentration']).Spike.transform(np.min)
normed_dat_frame = normed_dat_col.to_frame()
group_dat['Spike'] = normed_dat_frame
group_dat['SEM'] = normed_dat_col_sem
sns.catplot(data=group_dat, kind="bar", x="Day", y="Spike",hue='Day',row='Cortisol_Concentration',capsize=0.2,col='Group',row_order=['LL','L','H','HH']).set_xticklabels(['Day1 BFT','Day1 AFT','Day2','Day3'])

# def errplot(x, y, yerr, **kwargs):
#     ax = plt.gca()
#     data = kwargs.pop("data")
#     data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)

# g = sns.FacetGrid(group_dat, col="Group", row="Cortisol_Concentration")
# g.map_dataframe(errplot, "Day", "Spike", "SEM")
# %%
# immuno
immuno_table = pd.read_excel('Immunofluorescence Data.xlsx',header=8,usecols='C:H')

meltimmuno_table = immuno_table.melt(id_vars = ['Group','BFT/AFT','Dish','Pic'],value_name=('Max'))
sns.set_theme()
sns.catplot(x='BFT/AFT',y='Max',hue='Group',row='variable',data=meltimmuno_table,kind='box',col='Group',palette=['r','b'],saturation=1).set_xlabels('Treatment').set_xticklabels(['Before Treatment','After Treatment']).set_ylabels('Count')


# Paired t-test use stats.ttest_rel(pre,post)
# Twice: once for length and once for number

# %%
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
d,e = sym.symbols('d e')
sym.init_printing()
t = -(-((-d/(6+d)+1)-(2*(-d/(1+d)+1))) + e/(39.9+e) - (2*(-e/(179.9+e))))

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import fmin

def func(x):
    efficacy = (x[0]/(39.9+x[0])) + (-(x[1]/(15+x[1])+1))
    toxicity = (x[0]/(200.2+x[0])) + (-(x[1]/(4+x[1])+1))
    net_benefit = efficacy - toxicity
    return -net_benefit
def func2(x):
    efficacy = (x[0]/(39.9+x[0])) + (-(x[1]/(15+x[1])+1))
    toxicity = (x[0]/(179.2+x[0])) + (-(x[1]/(5+x[1])+1))
    net_benefit = efficacy - toxicity
    return -net_benefit
def func3(x):
    efficacy = (x[0]/(39.9+x[0])) + np.exp(-((x[1]-8)**2/2))
    toxicity = (x[0]/(179.2+x[0])) + (np.exp(-(x[1]-4)**2/2))+1
    net_benefit = efficacy - toxicity
    return -net_benefit
#x = (np.linspace(0,200,100),np.linspace(0,20,100))
x = np.linspace(0,200,100)
y = np.linspace(0,20,100)
z = func(x)
res = minimize(func,x0=(50,10),method='SLSQP',bounds=((0,200),(0,60)))
res2 = brute(func,(slice(0,100,0.5),slice(0,100,0.5)),full_output=True,finish=fmin)
fig = plt.figure()
ax = plt.axes(projection='3d')
x1,y1 = np.meshgrid(x,y)
z1 = func2((x1,y1))
ax.plot_surface(x1,y1,z1)
#ax.plot3D(x[0],x[1],z)
ax.plot3D(res.x[0],res.x[1],func(res.x),'ro')
#ax.plot3D(res2[0][0],res2[0][1],func((res2[0][0],res2[0][1])),'ro')
#plt.xlabel('Dose')
#plt.ylabel('Diffusion Rate')

# %%
import numpy as np
def ef_func(x):
    return (x[0]/(39.9+x[0])) + (-(x[1]/(15+x[1])+1))

def tox_func(x):
    return (x[0]/(179.2+x[0])) + (-(x[1]/(4+x[1])+1))
ED50 = 39.9
TD50 = 179.2
dose = np.linspace(0,200,100)
diff = np.linspace(0,20,100)
xx1,yy1 = np.meshgrid(x,y)
zzeff = ef_func((xx1,yy1))

A_vals = 10**np.arange(-2,4,0.1)

D_vals = 10**np.arange(-2,3,0.1)

A_diff = -D_vals/(15+D_vals) + 1
T_diff = -D_vals/(5+D_vals) + 1

A_dose = A_vals/(39.9+A_vals) 
T_dose = A_vals/(179.2+A_vals) 

plt.semilogx(D_vals,A_diff)
plt.semilogx(D_vals,T_diff)
plt.xlabel('Diffusion Rate $um^2/s$')
plt.ylabel('Effect')
plt.title('Diffusion Rate Response Curves')
plt.legend(['Efficacy','Toxicity'])
#ax2 = plt.axes(projection='3d')
#ax2.plot_surface(xx1,yy1,zzeff)
#ax2.plot3D(dose,diff,-z)

# %%
import numpy as np
import matplotlib.pyplot as plt
plt.plot([1,2],[1,2],label=r'$\sin^2 (x)$')
plt.legend()