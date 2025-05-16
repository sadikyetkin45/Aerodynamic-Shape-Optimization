import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
from glob import glob
import os, fnmatch
from pathlib import Path
warnings.filterwarnings('ignore')
import sys
import re
import copy
from Pre_MonteCarlo import *
import shutil
import math

################# M, P, T VALUES ##################
path = os.getcwd()
naca_path = "/naca_lhs/"
dir_URL = Path(path + naca_path)  # e.g. Path("/tmp")
mpt = [file.name for file in dir_URL.glob("*.dat")]
sorted_mpt = sorted(mpt)
mpt_arr = np.array(sorted_mpt)

global workdirBase
global currentWorkdir
global mainDirectoryName
global numberOfNode
global solver
global run_log_file
global averageValuePercentage
global lastRunCoefficientsDict

workdirBase = "workdir."
currentWorkdir = ""
mainDirectoryName = ""
numberOfNode = 48
solver = "hisa"
run_log_file = "hisa_out_log"
averageValuePercentage = 0.3
lastRunCoefficientsDict = dict([("CD", 0), ("CL", 0)])
curentRunCoefficientsDict = dict([("CD", 0), ("CL", 0)])
datPreName = "mptValues_"

gmres_iteration = dict([("GM",0)])

m = 0.09468598
p = 0.52987669
t = 0.57352240

# Başlangıç noktası
start_point = np.array([m, p, t])

# Adım boyutu
#step_size = 0.1
step_size = 0.7

# Öğrenme oranı
learning_rate = 0.5

# Iterasyon sayısı
iterations = 2000
epsilon = 0.5
awards = 0
penalty = 0

# Metropolis-Hasting Algorithm

def reinforcement(f,proposed_point, current_point, best_point, last_CD, current_CD , last_CL, current_CL,awards,penalty):
    is_workdir_deletion_allowed = True
    last_CD = lastRunCoefficientsDict["CD"]
    current_CD = curentRunCoefficientsDict["CD"]
    last_CL = lastRunCoefficientsDict["CL"]
    current_CL = curentRunCoefficientsDict["CL"]
    
    """
    
    condition_1_best = ((lastRunCoefficientsDict["CD"] - curentRunCoefficientsDict["CD"])/ lastRunCoefficientsDict["CD"])
    condition_2_best = ((lastRunCoefficientsDict["CL"] - curentRunCoefficientsDict["CL"])/ lastRunCoefficientsDict["CL"])
    
    if condition_1_best < epsilon and condition_1_best > 0 and condition_2_best < -epsilon:
        
    """   
    #acceptance_probability = np.exp((lastRunCoefficientsDict["CD"] - curentRunCoefficientsDict["CD"]) / learning_rate) * np.exp((lastRunCoefficientsDict["CL"] - curentRunCoefficientsDict["CL"]) / learning_rate)
    # Accept or reject the proposed point
    if (curentRunCoefficientsDict["CD"] < lastRunCoefficientsDict["CD"] and 
        curentRunCoefficientsDict["CL"] > lastRunCoefficientsDict["CL"] and 
        curentRunCoefficientsDict["CL"] > 0 and 
        gmres_iteration["GM"] < 0.01):
        best_point = proposed_point
        current_point = proposed_point
        best_values.append(best_value)  # En iyi değeri listeye ekle
        
        awards = awards + 10
        f.write('\n')
        f.write('If Condition is satisfied')
        f.write('\n')
        f.write('\n')
        f.write("Iteration Number is " + str(lowestCDIter))
        f.write('\n')
        f.write('All conditions are satisfied and new best point assigned' )
        f.write('\n')
        
        condition_1_best = ((lastRunCoefficientsDict["CD"] - curentRunCoefficientsDict["CD"])/ lastRunCoefficientsDict["CD"])
        condition_2_best = ((lastRunCoefficientsDict["CL"] - curentRunCoefficientsDict["CL"])/ lastRunCoefficientsDict["CL"])
        acceptance_probability = np.exp(condition_1_best/ learning_rate)*np.exp(condition_2_best/ learning_rate)
        
        lastRunCoefficientsDict["CD"] = curentRunCoefficientsDict["CD"]
        lastRunCoefficientsDict["CL"] = curentRunCoefficientsDict["CL"]
        
        f.write("Best Cd value is " + str(curentRunCoefficientsDict["CD"]))
        f.write('\n')
        f.write("Best Cl value is " + str(curentRunCoefficientsDict["CL"]))
        f.write('\n')
        f.write("Best Cl/Cd value is " + str(curentRunCoefficientsDict["CL"]/curentRunCoefficientsDict["CD"]))
        f.write('\n')           
        is_workdir_deletion_allowed = False
        
        f.write('\n')
        f.write('\n')
        
        f.write('\n')
        f.write(f'Award is {awards}')
        f.write('\n')
        
        
        
    else: 
        
        condition_1_best = ((lastRunCoefficientsDict["CD"] - curentRunCoefficientsDict["CD"])/ lastRunCoefficientsDict["CD"])
        condition_2_best = ((lastRunCoefficientsDict["CL"] - curentRunCoefficientsDict["CL"])/ lastRunCoefficientsDict["CL"])
        acceptance_probability = np.exp(condition_1_best/ learning_rate)*np.exp(condition_2_best/ learning_rate)
        penalty = penalty - 10
        f.write('\n')
        f.write(f'penalty is {penalty}')
        f.write('\n')
        f.write(f'current_point {current_point}')
        f.write('\n')
        
        current_point= proposed_point
        f.write(f'best_point {best_point}')
        f.write('\n')
    

    

    
    f.write('\n')
    f.write(f'condition_1 {condition_1_best}')
    f.write('\n')
        
    f.write('\n')
    f.write(f'condition_2 {condition_2_best}')
    f.write('\n')
    
    number_of_area_constrainted_failed_points = 0
    
    if acceptance_probability < epsilon: 
        
        is_criteria_satisfied = False
        while (is_criteria_satisfied == False):
            
            #m = np.random.normal(loc=0, scale=0.010, size=1)
            #p = np.random.normal(loc=0, scale=0.010, size=1)
            #t = np.random.normal(loc=0, scale=0.010, size=1)
            
            m = np.random.normal(loc=0, scale=0.1, size=1)
            p = np.random.normal(loc=0, scale=0.1, size=1)
            t = np.random.normal(loc=0, scale=0.1, size=1)
            
            random_step = (m[0],p[0],t[0])

            f.write('\n')
            f.write(f'random_step is  {random_step}')
            f.write('\n')
            # random_step = np.random.normal(loc=0, scale=0.025, size=3)
            
            
            
            proposed_point = best_point + random_step
                    
            proposed_point = np.clip(proposed_point, 0.001, 0.99)
            is_criteria_satisfied = ControlArea(proposed_point) 
            number_of_area_constrainted_failed_points += 1 

            
            
        f.write('\n')
        f.write("Iteration Number is " + str(lowestCDIter))
        f.write('\n')
        f.write(f'Acceptance Probability is {acceptance_probability}')
        f.write('\n')
        f.write("Condition is satisfied.")
        f.write('\n')
        f.write(f'best_point {best_point}')
        f.write('\n')
        #f.write(f'step_size  {step_size}')
        f.write('\n')
        f.write(f'proposed_point   {proposed_point}')
        f.write('\n')
        f.write('\n')
        f.write("Number of area constrainted failed points " + str(number_of_area_constrainted_failed_points-1))   
        f.write('\n')
        
        f.write('\n')
        f.write(f'First Condition condition_1 < epsilon is {condition_1_best < epsilon}')
        f.write('\n')
        f.write(f'Second Condition condition_2 < -epsilon is {condition_2_best  < -epsilon}')
        f.write('\n') 
        f.write("Bizim istediğimiz.")
        f.write('\n')
        
        return proposed_point, current_point, best_point, is_workdir_deletion_allowed,awards,penalty

    else:
        
        is_criteria_satisfied = False
        while (is_criteria_satisfied == False):
                
            random_step = np.random.uniform(-step_size, step_size, size=3)

            proposed_point = best_point + random_step

            proposed_point = np.clip(proposed_point, 0.001, 0.99)
            is_criteria_satisfied = ControlArea(proposed_point) 
            number_of_area_constrainted_failed_points += 1 
            
        f.write('\n')
        f.write("Iteration Number is " + str(lowestCDIter))
        f.write('\n')
        f.write(f'Acceptance Probability is {acceptance_probability}')
        f.write('\n')
        f.write("Random points are generated...................")
        f.write('\n')
        f.write(f'best_point {best_point}')
        f.write('\n')
        f.write(f'step_size  {step_size}')
        f.write('\n')
        f.write(f'proposed_point   {proposed_point}')
        f.write('\n')
        f.write('\n')
        f.write("Number of area constrainted failed points " + str(number_of_area_constrainted_failed_points-1)) 
        f.write('\n')
        
        
        return proposed_point, current_point, best_point, is_workdir_deletion_allowed,awards,penalty
    
    '''
    elif condition_2_best > -epsilon:
        
        is_criteria_satisfied = False
        while (is_criteria_satisfied == False):
            
            #m = np.random.normal(loc=0, scale=0.010, size=1)
            #p = np.random.normal(loc=0, scale=0.010, size=1)
            #t = np.random.normal(loc=0, scale=0.010, size=1)
            
            m = np.random.normal(loc=0, scale=0.1, size=1)
            p = np.random.normal(loc=0, scale=0.1, size=1)
            t = np.random.normal(loc=0, scale=0.1, size=1)
            
            random_step = (m[0],p[0],t[0])

            f.write('\n')
            f.write(f'random_step is  {random_step}')
            f.write('\n')
            # random_step = np.random.normal(loc=0, scale=0.025, size=3)
            
            
            
            proposed_point = best_point + random_step
                    
            proposed_point = np.clip(proposed_point, 0.001, 0.99)
            is_criteria_satisfied = ControlArea(proposed_point) 
            number_of_area_constrainted_failed_points += 1 

            
            
        f.write('\n')
        f.write("Iteration Number is " + str(lowestCDIter))
        f.write('\n')
        f.write(f'Acceptance Probability is {acceptance_probability}')
        f.write('\n')
        f.write("Condition is satisfied.")
        f.write('\n')
        f.write(f'best_point {best_point}')
        f.write('\n')
        #f.write(f'step_size  {step_size}')
        f.write('\n')
        f.write(f'proposed_point   {proposed_point}')
        f.write('\n')
        f.write('\n')
        f.write("Number of area constrainted failed points " + str(number_of_area_constrainted_failed_points-1))   
        f.write('\n')
        
        f.write('\n')
        f.write(f'First Condition condition_1 < epsilon is {condition_1_best < epsilon}')
        f.write('\n')
        f.write(f'Second Condition condition_2 < -epsilon is {condition_2_best  < -epsilon}')
        f.write('\n') 
        f.write("Bizim istediğimiz.")
        f.write('\n')
        
        return proposed_point, current_point, best_point, is_workdir_deletion_allowed,awards,penalty
    
    '''

# Yeni Fonksiyon
def naca4_airfoil_area(m, p, t, num_points=100):

    x = np.linspace(0, 1, num_points)  # Kordon boyunca noktalar

    # Üst yüzey koordinatları
    yt = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] < p:
            yt[i] = t * (0.2969 * np.sqrt(x[i]) - 0.1260 * x[i] - 0.3516 * x[i] ** 2 + 0.2843 * x[i] ** 3 - 0.1015 * x[i] ** 4)
        else:
            yt[i] = t * (0.2969 * np.sqrt(x[i]) - 0.1260 * x[i] - 0.3516 * x[i] ** 2 + 0.2843 * x[i] ** 3 - 0.1015 * x[i] ** 4)

    # Alt yüzey koordinatları
    yb = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] < p:
            yb[i] = -t * (0.2969 * np.sqrt(x[i]) - 0.1260 * x[i] - 0.3516 * x[i] ** 2 + 0.2843 * x[i] ** 3 - 0.1015 * x[i] ** 4)
        else:
            yb[i] = -t * (0.2969 * np.sqrt(x[i]) - 0.1260 * x[i] - 0.3516 * x[i] ** 2 + 0.2843 * x[i] ** 3 - 0.1015 * x[i] ** 4)

    # Alanı hesaplamak için trapez kuralını kullanın
    dx = np.diff(x)
    dx = np.append(dx, dx[-1])  # Son elemanı tekrar ekleyin
    area = 0.5 * abs(np.sum((yt - yb) * dx))

    return area, x, yt, yb  # x, yt ve yb değerlerini döndür

# Yeni Fonksiyonun kullanımı
def ControlArea(proposed_point, min_area=0.0040):
    m, p, t = proposed_point
    area, _, _, _ = naca4_airfoil_area(m, p, t)  # Alanı hesaplayın
    if area < min_area:  # Minimum alandan düşükse, CL'yi artırın
        return False  # Değişiklik yapma
    else:
        return True  # Kabul edilebilir

def GetNewWorkDirName(iteration: int):
    return os.chdir(os.path.join(mainDirectoryName, (workdirBase + str(iteration))))

def SetDirectoryBase():
    os.chdir(mainDirectoryName)

def SetDirectory(iteration: int, currentWorkdir: str, workdirBase: str):
    currentWorkdir_2 = "111"
    if (
        os.path.basename(os.path.normpath(currentWorkdir)) != (workdirBase + str(iteration))
    ):
        currentWorkdir = GetNewWorkDirName(iteration)
        os.chdir(currentWorkdir)
    return True

def PrepareMeshDict():
    return_value = os.system("python genblockmesh.py")
    return (return_value == 0)  ## Error output is 256

def RunBlockMesh():
    return_value = os.system("blockMesh")
    print("BlockMesh Finished")
    return_value_2 = os.system("transformPoints -rotate-x -90")
    return_value_3 = (return_value and return_value_2)
    return (return_value_3 == 0)

def CreateMesh():

    if (PrepareMeshDict()):
        return RunBlockMesh()
    else:
        return False

def GetAverage(_vals):

    startIndex = math.floor(len(_vals) * (1 - averageValuePercentage))
    count = len(_vals)
    sum = 0
    divider = 0

    while (startIndex < count):
        sum += float(_vals[startIndex][0]) 
        divider += 1
        startIndex += 1
    return sum / divider

def GetCoefficients():

    with open(run_log_file, "r") as file:
        file_content = file.read()

    cd_vals = re.findall(
        "\s*Cd\s*:\s*([0-9.\+\-eE]+)\s*([0-9.\+\-eE]+)\s*([0-9.\+\-eE]+)\s*[0-9.\+\-eE]+\s+",
        file_content,
    )
    cl_vals = re.findall(
        "\s*Cl\s*:\s*([0-9.\+\-eE]+)\s*([0-9.\+\-eE]+)\s*([0-9.\+\-eE]+)\s*[0-9.\+\-eE]+\s+",
        file_content,
    )

    GMRES_iteration = re.findall( "0\s+Residual: ([0-9.\+\-eE]+) \(([0-9.\+\-eE]+) ([0-9.\+\-eE]+) ([0-9.\+\-eE]+)\) ([0-9.\+\-eE]+)", file_content,)
    
    curentRunCoefficientsDict["CD"] = GetAverage(cd_vals)
    curentRunCoefficientsDict["CL"] = GetAverage(cl_vals)
    gmres_iteration["GM"] = GetAverage(GMRES_iteration)

def decomposePar():
    return_value = os.system("decomposePar")
    return (return_value == 0)

def RunCase():
    if (decomposePar()):

        return_value = os.system(
            f"mpirun -np {numberOfNode - 1} {solver} -parallel > {run_log_file}"
        )
        return (return_value == 0)
    else:
        return False

def CasePostProcess():

    GetCoefficients()
    os.system("python parav.py")


def sadikBrock(iteration: int, proposedPoints):
    
    originalDirectory = os.getcwd()

    os.chdir(workdirBase + str(iteration))
    if (CreateMesh() and RunCase()):
        CasePostProcess()

    os.chdir(originalDirectory)


data = []


os.chdir("..")

mainDirectoryName = os.getcwd()
currentWorkdir = workdirBase + "0"

RETURN_VAL = os.system("OF2206")


# CreateWorkDir(currentWorkdir)
if (os.path.exists(workdirBase + "0")):
    shutil.rmtree(workdirBase + "0")
copyBaseLineToWorkDir(workdirBase + "0")

PrepareDatFile(start_point, workdirBase + "0", "0")

if (sadikBrock("0", [m, p, t])):
    raise UnboundLocalError

iterList = []
cdList = []
clList = []
cl_cd_list = []
cl_cd_list_best=[]

lastRunCoefficientsDict["CD"] = curentRunCoefficientsDict["CD"] 
lastRunCoefficientsDict["CL"] = curentRunCoefficientsDict["CL"]

iterList.append(0)
cdList.append(curentRunCoefficientsDict["CD"])
clList.append(curentRunCoefficientsDict["CL"])  
cl_cd_list.append(curentRunCoefficientsDict["CL"]/ curentRunCoefficientsDict["CD"])
cl_cd_list_best.append(curentRunCoefficientsDict["CL"]/ curentRunCoefficientsDict["CD"])
current_point = start_point
best_point = start_point
best_value = lastRunCoefficientsDict["CD"]
points = []
values = []
best_values = [best_value]  # En iyi değerleri kaydetmek için bir liste

lowestCDIter = 0
mainDirectoryName = os.getcwd()
awards = 0
penalty = 0

# Yeni bir adımı rasgele seç
random_step = np.random.uniform(-step_size, step_size, size=3)
proposed_point = current_point + random_step
proposed_point = np.clip(proposed_point, 0.001, 0.99)
###################################################
best_point = current_point
is_workdir_deletion_allowed = False
for i in range(1, iterations):
    
    f=open("criteria.txt","a")
    
    f.write("\n")
    
    f.write(f'--------------------------------------') 
    f.write(f'{proposed_point}') 
    f.write(f'--------------------------------------') 
    f.write("\n")
    
    ###################################################
    
    #if ControlArea(proposed_point):
    currentWorkdir = workdirBase + str(i)
    
    if (is_workdir_deletion_allowed == True) and (os.path.exists(workdirBase + str(i-1))):
        shutil.rmtree(workdirBase + str(i-1))
        
    
    if (os.path.exists(workdirBase + str(i))):
        shutil.rmtree(workdirBase + str(i))
    copyBaseLineToWorkDir(workdirBase + str(i))

    PrepareDatFile(proposed_point, workdirBase + str(i), str(i))

    # Yeni nokta için değeri hesapla
    if (sadikBrock(i, proposed_point) == False):
        raise UnboundLocalError
        continue
        
    f.write("\n")
    print("Last Run Coefficient Values " + str(lastRunCoefficientsDict["CD"]))
    f.write("\n")
    f.write(f'----------------------------------- {i} ------------------------------------------------')
        
    f.write("\n")
    print("Current Run Coefficient Values " + str(curentRunCoefficientsDict["CD"]))

    f.write("\n")
    f.write(f'lastRunCoefficientsDict[CD] -- > {lastRunCoefficientsDict["CD"] }') 
    
    f.write("\n")
    f.write(f'curentRunCoefficientsDict[CD] -- > {curentRunCoefficientsDict["CD"] }') 
    
    f.write("\n")
    f.write(f'lastRunCoefficientsDict[CL] -- > {lastRunCoefficientsDict["CL"] }') 
    
    f.write("\n")
    f.write(f'curentRunCoefficientsDict[CL] -- > {curentRunCoefficientsDict["CL"] }') 
    
    f.write("\n")
    f.write(f'curentRunCoefficientsDict[CD] < lastRunCoefficientsDict[CD] ---> {curentRunCoefficientsDict["CD"] < lastRunCoefficientsDict["CD"]}' )
    
    f.write("\n")
    f.write(f'curentRunCoefficientsDict[CL] > lastRunCoefficientsDict[CL] ---> {curentRunCoefficientsDict["CL"] > lastRunCoefficientsDict["CL"]}' )
    
    f.write("\n")
    f.write(f'curentRunCoefficientsDict[CL] > 0 ---> {curentRunCoefficientsDict["CL"] > 0 }' )
    
    f.write("\n")
    f.write(f'gmres_iteration[GM] < 0.01 ---> {gmres_iteration["GM"] < 0.01}' )
    
    f.write("\n")
    f.write(f'----------------------------------- {i} ------------------------------------------------')
    f.write("\n")
        

    lowestCDIter = i
    
    proposed_point, current_point, best_point, is_workdir_deletion_allowed,awards,penalty = reinforcement(f,proposed_point, current_point, best_point, lastRunCoefficientsDict["CD"],curentRunCoefficientsDict["CD"],lastRunCoefficientsDict["CL"], curentRunCoefficientsDict["CL"],awards,penalty)
    # best_point = proposed_point
    iterList.append(i)
    f.write("\n")
    f.write(f'--------------------------------------') 
    f.write(f'{proposed_point}') 
    f.write(f'--------------------------------------') 
    f.write("\n")
    
    f.close()
    cdList.append(lastRunCoefficientsDict["CD"])
    
    clList.append(lastRunCoefficientsDict["CL"])
    cl_cd_list.append(curentRunCoefficientsDict["CL"]/ curentRunCoefficientsDict["CD"])
    cl_cd_list_best.append(lastRunCoefficientsDict["CL"]/lastRunCoefficientsDict["CD"])

    fig_1 = plt.figure(figsize=(10, 6))
    # En iyi değer vs iteration grafiği
    plt.plot(iterList, cdList)  # En iyi değerler listesi çizilir
    plt.xlabel("İterasyon")
    plt.ylabel("En İyi Değer")
    plt.title("En İyi Değer vs İterasyon")
    plt.savefig("markov_decision_monte_carlo_vs_iteration_CD.png")

    fig_2 = plt.figure(figsize=(10, 6))
    plt.plot(iterList, clList)  # En iyi değerler listesi çizilir
    plt.xlabel("İterasyon")
    plt.ylabel("En İyi Değer")
    plt.title("En İyi Değer vs İterasyon")
    plt.savefig("markov_decision_monte_carlo_vs_iteration_CL.png")

    fig_3 = plt.figure(figsize=(10, 6))
    plt.plot(iterList, cl_cd_list)  # En iyi değerler listesi çizilir
    plt.xlabel("İterasyon")
    plt.ylabel("En İyi Değer Cl/Cd")
    plt.title("En İyi Değer vs İterasyon")
    plt.savefig("markov_decision_monte_carlo_vs_iteration_CL_CD.png")
    
    fig_4 = plt.figure(figsize=(10, 6))
    plt.plot(iterList, cl_cd_list_best)  # En iyi değerler listesi çizilir
    plt.xlabel("İterasyon")
    plt.ylabel("En İyi Değer Best Cl/Cd")
    plt.title("En İyi Değer vs İterasyon")
    plt.savefig("best_CL_CD.png")
    
    print("Iteration Number is " + str(lowestCDIter))
    print(curentRunCoefficientsDict["CD"])
    print(curentRunCoefficientsDict["CL"])
    print(curentRunCoefficientsDict["CL"]/curentRunCoefficientsDict["CD"])


print(f"En iyi nokta: {best_point}")
print(f"En iyi değer: {best_value}")
print(f"lowest CD iter : {lowestCDIter}")


# En iyi değer vs iteration grafiği
fig_3 = plt.figure(figsize=(10, 6))
plt.plot(iterList, cdList)  # En iyi değerler listesi çizilir
plt.xlabel("İterasyon")
plt.ylabel("En İyi Değer")
plt.title("En İyi Değer vs İterasyon")
plt.savefig("markov_decision_monte_carlo_vs_iteration.png")
