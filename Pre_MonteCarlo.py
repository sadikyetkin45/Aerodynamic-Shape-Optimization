import os
import shutil
import numpy as np
import copy 

def RunNacaZero():
    originalDirectory = os.getcwd()
    os.chdir(originalDirectory)
    return_value = os.system("python naca_zero.py")
    os.chdir(originalDirectory)
    return (return_value == 0)

def RunNacaLHS():
    originalDirectory = os.getcwd()
    os.chdir(originalDirectory)
    return_value = os.system("python naca.py")
    os.chdir(originalDirectory)
    return (return_value == 0)

def TransferDatFiles(fileName , LocationToTransfer):
    shutil.copy(fileName, LocationToTransfer)


    return 0


def copyBaseLineToWorkDir( LocationToTransfer):
 

    src_dir = '/root/capsule/baseline'
 
    # path to destination directory
    dest_dir = LocationToTransfer
    
    # getting all the files in the source directory
    files = os.listdir(src_dir)
    
    shutil.copytree(src_dir, dest_dir)
    return 0



def save_to_file(airfoil_points, filename):
    np.savetxt(filename, airfoil_points, delimiter=' ', fmt='%.8f')


def PrepareDatFile(proposedPoints , directory , iteration):
    
    originalDirectory = os.getcwd()

    os.chdir(directory)

    filename = f"mptValues_{iteration}.dat"
    
    save_to_file(proposedPoints, filename)
    os.chdir(originalDirectory)
    return 0

def CreateWorkDir(workdirName):
    if not os.path.exists(workdirName):
        os.makedirs(workdirName)
