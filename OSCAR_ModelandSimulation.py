# Neuron spiking model/simulation
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tkinter import *
from scipy import interpolate


class Neuron(object):
    """A neuron that will fire at a "normal" rate"""
    
    def __init__(self):
        """ Resting Membrane Potential (in mV)"""

        self.voltage = -70
    
    def has_spiked(self,cort_con):
        """ 
        Determines whether or not this neuron has spiked
        
        Arg: cortical concentration (float) 1.2, 1.4, 1.6, 1.8, 2.0 representing lowest to highest cortisol concentration
        
        Returns: Boolean (True or False) whether or not neuron spiked depending on voltage
        If the voltage is greater than the threshold of -55 mV, the neuron will fire. 
        Otherwise, it will not
        """
        
        new_volt = self.voltage*np.exp(-(((np.random.lognormal(0,0.2))**cort_con)/((3+cort_con))))
        if new_volt > -55:
            return True
        else:
            return False
        
    
    def __repr__(self):
        return 'Neuron'
    
    
    
class DepressedNeuron(object):
    """ A "depressed" neuron that will spike less by having a lower resting membrane potential"""
    
    def __init__(self):
        """Resting Membrane Potential (in mV)"""
        
        self.voltage = -90
        
    def has_spiked(self,diff_con,cort_con):
        """ 
        Determines whether or not this neuron has spiked
        
        Args: diffusion concentration (float): Diffusion concentration at a specific position and time
              cort_con (float): 1.2, 1.4, 1.6, 1.8, 2.0 representing lowest to highest cortisol concentration
        
        Returns: Boolean (True or False) whether or not neuron spiked depending on voltage
        If the voltage is greater than the threshold of -55 mV, the neuron will fire. 
        Otherwise, it will not
        """
        new_volt = self.voltage*np.exp(-(((1+diff_con)**cort_con)*((np.random.lognormal(0,0.2))**cort_con)/((3+cort_con))))

        if new_volt > -55:
            return True
        else:
            return False
        
    
    def __repr__(self):
        return 'DepNeuron'
        
class Patient(object):
    """A patient with Major Depressive Disorder"""
    
    def __init__(self,neurons,cort_con):
        """
        Args: neurons (list of Neuron and DepressedNeuron patient has)
              cort_con (float representing cortisol concentration)
        """
        self.neurons = neurons
        self.cort_con = cort_con
        self.on_medication = False
        
    def set_on_medication(self):
        """Put patient on antidepressant"""
        self.on_medication = True
        
    def check_if_on_medication(self):
        """
        Checks if patient is on antidepressant
        
        Returns: Boolean (True or False)   
        """
        return self.on_medication
        
    def get_neurons(self):
        """ Returns neuron grid in patient"""
        
        return self.neurons


def createNeuronGrid(gridsize,prob_Normal,prob_Depressed):
    """Creates an nxn grid full of neurons
    
       Args: gridsize (tuple): gives size of nxn grid
             prob_normal (float): gives probability of placing a normal neuron on grid
             prob_Depressed (float): gives probability of placing a depressed neuron on grid
             
       Returns: tuple, grid of neurons and depressed neurons
    """
   
    return np.random.choice([Neuron(),DepressedNeuron()],size=gridsize,p=[prob_Normal,prob_Depressed])

def visualizeGrid(grid):
    """Visualize the nxn grid of neurons
    
       Arg: grid (tuple): grid of Neurons and DepNeurons
    """
    root = Tk()
    for r in range(len(grid)):
        for c in range(len(grid)):
            if isinstance(grid[r][c],DepressedNeuron):
                btn_ipady = Button(root, text="Neuron", bg="blue",fg="white")
                btn_ipady.grid(column=c,row=r,padx=10,pady=10)
            else:
                btn_rowspan = Button(root, text="Neuron",bg="red",fg="white")
                btn_rowspan.grid(column=c,row=r,padx=10,pady=10)
    root.mainloop()
    
def visualizeDrugDiffusion(grid,diffrate,dose):
    """
       Visualize the diffusion of drug across grid over time with a heatmap
       The point (0,0) is located at the center of the grid where the drug is
       administered
    
       Args: grid (tuple): grid of Neurons and DepNeurons
             diffrate (int): diffusion coefficient of drug in um^2/s
             dose (int): dose of drug in mg
             
    """
    midpoint = int(np.floor(len(grid)/2))
    grid_copy = np.zeros([len(grid),len(grid)])
    grid_copy[midpoint][midpoint] = dose
    sns.heatmap(grid_copy,xticklabels=np.arange(-midpoint,midpoint+1),yticklabels=-np.arange(-midpoint,midpoint+1),cbar=False)
    plt.title('Drug Diffusion Heatmap when t = 0 days')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.pause(0.5)
    for t in np.linspace(0.1,3,30):
        total_dose = 0
        dif_grid = np.empty([len(grid),len(grid)])
        midpoint = int(np.floor(len(grid)/2))
        Diff_x,Diff_y = np.meshgrid([np.arange(-midpoint,midpoint+1)],-np.arange(-midpoint,midpoint+1))
        for row in range(len(grid)):
            for col in range(len(grid)):
                dif_grid[row][col] = calcDiffusion(Diff_x[row][col],Diff_y[row][col],diffrate,t,dose)
        sns.heatmap(dif_grid,xticklabels=np.arange(-midpoint,midpoint+1),yticklabels=-np.arange(-midpoint,midpoint+1),cbar=False)
        plt.title('Drug Diffusion Heatmap when t = '+str(np.round(t,2))+' days')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.pause(0.5)
    

def calcDiffusion(x,y,D,t,dose):
    """Gives the value from the diffusion equation analytical solution
    
       Args: x (int): reprents x position on neuron grid
             y (int): represent y position on neuron grid
             D (int): represents diffusion coefficient of drug in um^2/s
             t (float): represents time passed since drug delivery
             dose (int): represents dose of drug in mg
             
      Returns: diff_val (float): value of concentration of drug at a specific 
      time and position
    
    """
    diff_val = dose*np.exp(-(x**2+y**2)/(4*D*t))*1/(4*np.pi*D*t)
    return diff_val

def exp_func(t,A1,tau1,A2,tau2):
    """ Bi-exponential decay function used for scipy.optimize.curve_fit. Other
        args besides t are calculated by the curve_fit function
    
        Arg: only t (float): time
        
        Returns: bi-exponential decay function
    """
    return A1*np.exp(-(t/tau1)) + A2*np.exp(-(t/tau2))

def model_func_values(t,cort_con):
    """ Function that gives values from model developed from simulation
    
        Args: t (float): time 
        cort_con (float) 1.2, 1.4, 1.6, 1.8, 2.0 representing lowest to highest cortisol concentration

        Returns: value of model at a given time and cortisol concentration
    """
    return (35.4*cort_con+22.36)*np.exp(-(t/3.04)) - (36.85*cort_con+5.68)*np.exp(-(t/(0.142*cort_con+0.0408)))

def model_graphs():
    t = np.arange(0,3,0.01)
    cort_con = [1.2,1.4,1.6,1.8,2.0]
    for cort in cort_con:
        plt.plot(t,model_func_values(t, cort))
    plt.xlabel('Time (Days)')
    plt.ylabel('Number of Spikes')
    plt.title('Model graphs with different cortisol concentrations')
    plt.legend(['Lowest','Low','Medium','High','Highest'])

def spiking_simulation(gridsize,prob_Normal,prob_Depressed,diffrate,dose,cort_con):
    """
    Runs the simulation of neurons spiking. Calculates drug concentration at a
    specific position and time to help a depressed neuron spike. Also plots 
    time and number of spikes
    
    Args: gridsize (tuple): gives size of nxn grid
          prob_normal (float): gives probability of placing a normal neuron on grid
          prob_Depressed (float): gives probability of placing a depressed neuron on grid
          diffrate (int): diffusion coefficient of drug in um^2/s
          dose (int): dose of drug in mg
          cort_con (float) 1.2, 1.4, 1.6, 1.8, 2.0 representing lowest to highest cortisol concentration
          
    Returns: total_spikes (list): a list of total spikes over time
    """
    total_spikes = []
    patient = Patient(createNeuronGrid(gridsize, prob_Normal, prob_Depressed),cort_con)
    for t in np.linspace(0.1,3,100):
        spikes = []
        grid = patient.get_neurons()
        midpoint = int(np.floor(len(grid)/2))
        Diff_x,Diff_y = np.meshgrid([np.arange(-midpoint,midpoint+1)],-np.arange(-midpoint,midpoint+1))
        for row in range(len(grid)):
            for col in range(len(grid)):
                if isinstance(grid[row][col],Neuron):
                    spikes.append(grid[row][col].has_spiked(cort_con))
                else:
                    diff_val = calcDiffusion(Diff_x[row][col],Diff_y[row][col],diffrate,t,dose)
                    spikes.append(grid[row][col].has_spiked(diff_val,cort_con))
        total_spikes.append(sum(spikes))
    sns.set_theme()
    sns.relplot(x=np.linspace(0.1,3,100),y=total_spikes)
    return total_spikes

def spiking_simulation_with_delay(gridsize,prob_Normal,prob_Depressed,diffrate,dose,cort_con,stepsize):
    """
    Runs the simulation of neurons spiking with a time delay before medication
    Calculates drug concentration at a specific position and time to help a 
    depressed neuron spike. Also plots time and number of spikes
    
    Args: gridsize (tuple): gives size of nxn grid
          prob_normal (float): gives probability of placing a normal neuron on grid
          prob_Depressed (float): gives probability of placing a depressed neuron on grid
          diffrate (int): diffusion coefficient of drug in um^2/s
          dose (int): dose of drug in mg
          cort_con (float) 1.2, 1.4, 1.6, 1.8, 2.0 representing lowest to highest cortisol concentration
          step_size (float): represents step size of time data points
          
    Returns: total_spikes (list): a list of total spikes over time
    """
    
    total_spikes = []
    patient = Patient(createNeuronGrid(gridsize, prob_Normal, prob_Depressed),cort_con)
    time = np.arange(0,3,stepsize)
    time_counter = 1
    for t in np.arange(0,3,stepsize):
        if t > 0.1:
            patient.set_on_medication()
        spikes = []
        grid = patient.get_neurons()
        midpoint = int(np.floor(len(grid)/2))
        Diff_x,Diff_y = np.meshgrid([np.arange(-midpoint,midpoint+1)],-np.arange(-midpoint,midpoint+1))
        for row in range(len(grid)):
            for col in range(len(grid)):
                if isinstance(grid[row][col],Neuron):
                    spikes.append(grid[row][col].has_spiked(cort_con))
                else:
                    if patient.check_if_on_medication():
                        diff_val = calcDiffusion(Diff_x[row][col],Diff_y[row][col],diffrate,time[time_counter],dose)
                        spikes.append(grid[row][col].has_spiked(diff_val,cort_con))
                        
                    else:
                        spikes.append(grid[row][col].has_spiked(0,cort_con))
        if patient.check_if_on_medication():
            time_counter += 1
        total_spikes.append(sum(spikes))
    sns.set_theme()
    sns.relplot(x=time,y=total_spikes)
    plt.xlabel('Time (Days)')
    plt.ylabel('Number of Spikes')
    plt.title('Simulation Graph')
    return total_spikes

def spiking_simulation_with_multiple_doses(gridsize,prob_Normal,prob_Depressed,diffrate,dose,cort_con,stepsize,num_days,next_dose):
    """
    Runs the simulation of neurons spiking with additional doses taken after the first one
    Calculates drug concentration at a specific position and time to help a 
    depressed neuron spike. Also plots time and number of spikes
    
    Args: gridsize (tuple): gives size of nxn grid
          prob_normal (float): gives probability of placing a normal neuron on grid
          prob_Depressed (float): gives probability of placing a depressed neuron on grid
          diffrate (int): diffusion coefficient of drug in um^2/s
          dose (int): dose of drug in mg
          cort_con (float) 1.2, 1.4, 1.6, 1.8, 2.0 representing lowest to highest cortisol concentration
          step_size (float): represents step size of time data points
          num_days (int): represents how many days are shown on graph
          next_dose (int): represents how many days before another dose
          
    Returns: total_spikes (list): a list of total spikes over time
    
    """
    
    total_spikes = []
    patient = Patient(createNeuronGrid(gridsize, prob_Normal, prob_Depressed),cort_con)
    time = np.arange(0,num_days,stepsize)
    dose_time_array = np.zeros(int(num_days/next_dose+1))
    dose_counter = 0
    for t in np.arange(0,num_days,stepsize):
        if t > 0.1:
            patient.set_on_medication()
        if t % next_dose < 1e-6:
            dose_time_array[dose_counter] = 1
            dose_counter += 1
        spikes = []
        grid = patient.get_neurons()
        midpoint = int(np.floor(len(grid)/2))
        Diff_x,Diff_y = np.meshgrid([np.arange(-midpoint,midpoint+1)],-np.arange(-midpoint,midpoint+1))
        for row in range(len(grid)):
            for col in range(len(grid)):
                if isinstance(grid[row][col],Neuron):
                    spikes.append(grid[row][col].has_spiked(cort_con))
                else:
                    if patient.check_if_on_medication():
                        diff_val = []
                        for num in range(np.count_nonzero(dose_time_array)):
                            time_counter = int(dose_time_array[num])
                            diff_val.append(calcDiffusion(Diff_x[row][col],Diff_y[row][col],diffrate,time[time_counter],dose))
                        spikes.append(grid[row][col].has_spiked(np.sum(diff_val),cort_con))    
                    else:
                        spikes.append(grid[row][col].has_spiked(0,cort_con))
        if patient.check_if_on_medication():
            for num_counter in range(np.count_nonzero(dose_time_array)):
                dose_time_array[num_counter] += 1
        total_spikes.append(sum(spikes))
    sns.set_theme()
    sns.relplot(x=time,y=total_spikes)
    plt.xlabel('Time (Days)')
    plt.ylabel('Number of Spikes')
    plt.title('Simulation Graph')
    return total_spikes


#x = spiking_simulation((9,9), 0.2, 0.8, 6, 75, 1.2)


y = spiking_simulation_with_delay((15,15), 0.2, 0.8, 6, 75, 1.2, 0.01)
popt, pcov = curve_fit(exp_func,np.arange(0,3,0.01),y,p0=[54,30,-60,0.5],bounds=([0,0,-100,0],(500,1000,0,100)))
plt.plot(np.arange(0,3,0.01),exp_func(np.arange(0,3,0.01),*popt),'r-')
plt.xlabel('Time (Days)')
plt.ylabel('Spikes')
plt.legend(['Model','Data'])
plt.title('Simulation Graph')

#z = spiking_simulation_with_multiple_doses((15,15), 0.2, 0.8, 6, 75, 1.2, 0.05,14,3)


# %% See how well model fits data
combined_dict = {'Control_LL_Day1_BFT':[12,15],'Control_LL_Day1_AFT':[50,60],'Control_LL_Day2_AFT':[35,42],'Control_LL_Day3_AFT':[20,24]}
cort_con_num = 1.0
x = [0,1,2,3]
for cort_con in ['LL_', 'L_', 'H_', 'HH_']:
    if cort_con == 'H_':
        cort_con_num += 0.4
    else:
        cort_con_num += 0.2
    for group in ['Control_', 'Mem_', 'Lac_', 'Tra_']:
        temp_spikes = []
        for day in ['Day1_BFT', 'Day1_AFT', 'Day2_AFT', 'Day3_AFT']:
                temp_spikes.append(np.mean(combined_dict[group+cort_con+day]))
        interp_func = interpolate.interp1d(x, temp_spikes,kind='cubic')
        xnew = np.arange(0,3.1,0.1)
        ynew = interp_func(xnew)
        modelyvals = model_func_values(xnew,cort_con_num)
        SSE = np.sum((ynew-modelyvals)**2)
        S = np.sqrt(SSE/(len(xnew)-4))
        plt.plot(xnew,ynew,'o',xnew,modelyvals,'-')
        plt.vlines(xnew,ynew,modelyvals)
        plt.xlabel('Time (Days)')
        plt.ylabel('Number of Spikes')
        plt.legend(['Data','Model'])
        plt.title('More Data Points')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(2.4,45,'S = '+ str(round(S)) + ' Spikes',bbox=props)
        plt.title('Comparing Data and Model')
        #plt.title('Cort_con: ' + cort_con[0:-1] + '| Group:' + group[0:-1] + ' S: '+ str(S))
        
        
        
        
        
