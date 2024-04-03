### Partition Functions ###
import numpy as np

kb = 0.69 
h = 2*np.pi 
hbar = 1 
NA = 6.02 * 10**(23) 

def qtrans(m,t):
    qtrans = (2*np.pi*m*kb*t)**(3/2)/(h**3) 
    return qtrans 

def qrotNL(ia,ib,ic,sigma,t):
    qrotNL = (((np.pi)**(1/2)) * (8*(np.pi**(2)*kb*t)**(3/2))/(h**(3)*sigma)) * ((ia * ib * ic)**(1/2))
    return qrotNL 

def qV(w,t):
    beta = 1/(kb*t) 
    qV = np.exp(- 1/2 * hbar * beta * w)/(1 - np.exp(- 1 * hbar * beta * w))
    return qV 

def qe(energy,gj,t):
    beta = 1/(kb*t)
    eg = energy
    qe = gj * np.exp(-beta*eg)
    return qe

def QV(lv,t):
    QV = 1
    for i in range(len(lv)):
        w = lv[i]
        QV = QV * qV(w,t)
    return QV #

def qETH_Cl(t):
    m = 75.02/NA 
    ia = 1.021
    ib = 0.171
    ic = 0.155
    lv = [2967,2946,2881,1463,1448,1385,1289,1081,974,677,336,3014,2986,1448,1251,974,786,251]
    sigma = 1
    energy = -538.797 * 220000
    gj = 1
    qETH_Cl = qtrans(m,t) * qrotNL(ia,ib,ic,sigma,t) * QV(lv,t) * qe(energy,gj,t)
    return qETH_Cl

def qETH_Br(t):
    m = 108.97/NA
    ia = 0.973
    ib = 0.122
    ic = 0.113
    lv = [2988,2937,2880,1451,1451,1386,1252,1061,964,583,290,3018,2988,1451,1248,964,770,247]
    sigma = 1
    energy = -2651.775712 * 220000
    gj = 1
    qETH_Br = qtrans(m,t) * qrotNL(ia,ib,ic,sigma,t) * QV(lv,t) * qe(energy,gj,t)
    return qETH_Br

def qCl(t):
    m = 35.45/NA 
    energy = -459.795555 * 220000
    gj = 1
    qCl = qtrans(m,t) * qe(energy,gj,t)
    return qCl
 
def qBr(t):
    m = 79.9/NA
    energy = -2572.853721 * 220000
    gj = 1
    qBr = qtrans(m,t) * qe(energy,gj,t)
    return qBr

def eq(t):
    a = (qETH_Cl(t) * qBr(t))/(qETH_Br(t) * qCl(t))
    if np.isnan(a) == True:
        eq = 0
    else:
        eq = (qETH_Cl(t) * qBr(t))/(qETH_Br(t) * qCl(t))
    return eq
  
