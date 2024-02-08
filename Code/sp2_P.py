### Functions for computing the thermodynamic properties of single-mode cavity coupled reaction
### under product VSC

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import matplotlib.ticker as mtick
import mpmath as m

## See sp2_R.py for information about variables and functions (applied there to the case of R strong coupling, here to the case of P strong coupling)

def fsolver(f,precision,boundary1,boundary2):
    while np.abs(boundary1 - boundary2) > precision:
        if f(boundary1) * f(boundary2) <= 0:
            boundary0 = 1/2 * (boundary2 + boundary1)
            if f(boundary0) * f(boundary1) <= 0:
                boundary2 = boundary0 
            else:
                boundary1 = boundary0
        else:
            print("Redefine boundaries please.")
            break
    return boundary1

def wph(m,t,g,wc,eq):
    for i in range(4):
        n_trail = m - m/(np.sqrt(eq(t)) + 1) -2 + i
        omega_R1 = 2*g * np.sqrt(n_trail)
        wk1 = round(np.sqrt(wc**2 - omega_R1**2),0)
        n_return_test = fd_np(t,g,wc,wk1,m,eq)
        omega_R2 = 2*g * np.sqrt(n_return_test)
        wk2 = round(np.sqrt(wc**2 - omega_R2**2),0)
        if wk1 == wk2:
            return wk1

def wlp(g,n,wc,wk):
    gk = g * np.sqrt(n)
    wkt = np.sqrt(wk**2 + 4*(gk**2))
    a = wkt**2 + wc**2
    b = (wkt**2 - wc**2)**2 + 16*(gk**2)*(wc**2)
    wlp = (1/np.sqrt(2)) * np.sqrt(a - np.sqrt(b))
    return wlp

def wup(g,n,wc,wk):
    gk = g * np.sqrt(n)
    wkt = np.sqrt(wk**2 + 4*(gk**2))
    a = wkt**2 + wc**2
    b = (wkt**2 - wc**2)**2 + 16*(gk**2)*(wc**2)
    wup = (1/np.sqrt(2)) * np.sqrt(a + np.sqrt(b))
    return wup

def qlp(t,g,n,wc,wk):
    hbar = 1
    k = 0.69
    beta = 1/(k*t)
    qlp = np.exp(- 1/2 * hbar * beta * wlp(g,n,wc,wk))/(1 - np.exp(- 1 * hbar * beta * wlp(g,n,wc,wk)))
    return qlp

def qup(t,g,n,wc,wk):
    hbar = 1
    k = 0.69
    beta = 1/(k*t)
    qup = np.exp(- 1/2 * hbar * beta * wup(g,n,wc,wk))/(1 - np.exp(- 1 * hbar * beta * wup(g,n,wc,wk)))
    return qup

def flp(t,g,n,wc,wk):
    hbar = 1
    k = 0.69
    beta = 1/(k*t)
    flp = (1 + np.exp(- beta * hbar * wlp(g,n,wc,wk)))/((1 - np.exp(- beta * hbar * wlp(g,n,wc,wk)))**2) * (-1/2 * hbar * beta) * np.exp(-1/2 * beta * hbar * wlp(g,n,wc,wk))
    return flp

def fup(t,g,n,wc,wk):
    hbar = 1
    k = 0.69
    beta = 1/(k*t)
    fup = (1 + np.exp(- beta * hbar * wup(g,n,wc,wk)))/((1 - np.exp(- beta * hbar * wup(g,n,wc,wk)))**2) * (-1/2 * hbar * beta) * np.exp(-1/2 * beta * hbar * wup(g,n,wc,wk))
    return fup

def dwlp(g,n,wc,wk):
    f = lambda n: wlp(g,n,wc,wk)
    gr = grad(f)
    return gr(n)

def dwup(g,n,wc,wk):
    f = lambda n: wup(g,n,wc,wk)
    gr = grad(f)
    return gr(n)

def dlp(t,g,n,wc,wk):
    dlp = flp(t,g,n,wc,wk) * dwlp(g,n,wc,wk)
    return dlp

def dup(t,g,n,wc,wk):
    dup = fup(t,g,n,wc,wk) * dwup(g,n,wc,wk)
    return dup

def fd_np(t,g,wc,wk,m,eq):
    def fun(n):
        k = 0.69
        beta = 1/(k*t)
        hbar = 1
        fun = np.exp((1 / (qlp(t,g,n,wc,wk) * qup(t,g,n,wc,wk))) * (qlp(t,g,n,wc,wk) * dup(t,g,n,wc,wk) + qup(t,g,n,wc,wk) * dlp(t,g,n,wc,wk))) * eq(t) - ((m-n)**(2))/(n**(2))
        return fun
    n = m - fsolver(fun,2E-10,2E-11,m)
    fd_np = n
    return fd_np

def cavity_effect(t,g,wc,wk,nr,eq):
    if nr == 0.0:
        cavity_effect = 1
    else:
        cavity_effect = np.exp((1 / (qlp(t,g,nr,wc,wk) * qup(t,g,nr,wc,wk))) * (qlp(t,g,nr,wc,wk) * dup(t,g,nr,wc,wk) + qup(t,g,nr,wc,wk) * dlp(t,g,nr,wc,wk)))
    return cavity_effect

def entlp(t,g,wc,wk,nr,eq):
    hbar = 1
    k = 0.69
    beta = 1/(k*t)
    entlp = (1/2 * hbar * wlp(g,nr,wc,wk)) * (1 + np.exp(-hbar * beta * wlp(g,nr,wc,wk)))/((1 - np.exp(-hbar * beta * wlp(g,nr,wc,wk)))**2) * np.exp(-1/2 * hbar * beta * wlp(g,nr,wc,wk))/qlp(t,g,nr,wc,wk)
    return entlp

def entup(t,g,wc,wk,nr,eq):
    hbar = 1
    k = 0.69
    beta = 1/(k*t)
    entup = (1/2 * hbar * wup(g,nr,wc,wk)) * (1 + np.exp(-hbar * beta * wup(g,nr,wc,wk)))/((1 - np.exp(-hbar * beta * wup(g,nr,wc,wk)))**2) * np.exp(-1/2 * hbar * beta * wup(g,nr,wc,wk))/qup(t,g,nr,wc,wk)
    return entup

def entml(t,g,wc,wk,m,eq):
    hbar = 1
    k = 0.69
    beta = 1/(k*t)
    qc = (np.exp(-1/2 * hbar * beta * wc))/(1 - np.exp(-hbar * beta * wc))
    entml = (1/2 * hbar * wc) * (1 + np.exp(-hbar * beta * wc))/((1 - np.exp(-hbar * beta * wc))**2) * np.exp(-1/2 * hbar * beta * wc)/qc
    return entml

def entph(t,g,wc,wk,m,eq):
    hbar = 1
    k = 0.69
    beta = 1/(k*t)
    qp = (np.exp(-1/2 * hbar * beta * wk))/(1 - np.exp(-hbar * beta * wk))
    entph = (1/2 * hbar * wk) * (1 + np.exp(-hbar * beta * wk))/((1 - np.exp(-hbar * beta * wk))**2) * np.exp(-1/2 * hbar * beta * wk)/qp
    return entph

def etplp(t,g,wc,wk,nr,eq):
    hbar = 1
    k = 0.69
    beta = 1/(k*t)
    dlp = (-1/2 * hbar * wlp(g,nr,wc,wk)) * (1 + np.exp(-hbar * beta * wlp(g,nr,wc,wk)))/((1 - np.exp(-hbar * beta * wlp(g,nr,wc,wk)))**2) * np.exp(-1/2 * hbar * beta * wlp(g,nr,wc,wk))
    etplp = k * np.log(qlp(t, g, nr, wc, wk)) + (-1 / t) * dlp / qlp(t, g, nr, wc, wk)
    return etplp

def etpup(t, g, wc, wk, nr, eq):
    hbar = 1
    k = 0.69
    beta = 1 / (k * t)
    dup = (-1 / 2 * hbar * wup(g, nr, wc, wk)) * (1 + np.exp(-hbar * beta * wup(g, nr, wc, wk))) / ((1 - np.exp(-hbar * beta * wup(g, nr, wc, wk))) ** 2) * np.exp(-1 / 2 * hbar * beta * wup(g, nr, wc, wk))
    etpup = k * np.log(qup(t, g, nr, wc, wk)) + (-1 / t) * dup / qup(t, g, nr, wc, wk)
    return etpup
  

def etpml(t, g, wc, wk, m, eq):
    hbar = 1
    k = 0.69
    beta = 1 / (k * t)
    qc = (np.exp(-1 / 2 * hbar * beta * wc)) / (1 - np.exp(-hbar * beta * wc))
    dml = (-1 / 2 * hbar * wc) * (1 + np.exp(-hbar * beta * wc)) / ((1 - np.exp(-hbar * beta * wc)) ** 2) * np.exp(-1 / 2 * hbar * beta * wc)
    etpml = k * np.log(qc) + (-1 / t) * dml / qc
    return etpml


def etpph(t, g, wc, wk, m, eq):
    hbar = 1
    k = 0.69
    beta = 1 / (k * t)
    qp = (np.exp(-1 / 2 * hbar * beta * wk)) / (1 - np.exp(-hbar * beta * wk))
    dph = (-1 / 2 * hbar * wk) * (1 + np.exp(-hbar * beta * wk)) / ((1 - np.exp(-hbar * beta * wk)) ** 2) * np.exp(-1 / 2 * hbar * beta * wk)
    etpph = k * np.log(qp) + (-1 / t) * dph / qp
    return etpph


def delta_delta_A(t, g, wc, wk, m, eq):
    delta_delta_S = etplp(t, g, wc, wk, m, eq) + etpup(t, g, wc, wk, m, eq) - etpml(t, g, wc, wk, m, eq) - etpph(t, g, wc, wk, m, eq)
    delta_delta_E = entlp(t, g, wc, wk, m, eq) + entup(t, g, wc, wk, m, eq) - entml(t, g, wc, wk, m, eq) - entph(t, g, wc, wk, m, eq)
    delta_delta_A = delta_delta_E - t * delta_delta_S
    return delta_delta_A

def delta_delta_S(t, g, wc, wk, m, eq):
    delta_delta_S = etplp(t, g, wc, wk, m, eq) + etpup(t, g, wc, wk, m, eq) - etpml(t, g, wc, wk, m, eq) - etpph(t, g, wc, wk, m, eq)
    return delta_delta_S

def delta_E(t, g, wc, wk, m, eq):
    delta_delta_E = entlp(t, g, wc, wk, m, eq) + entup(t, g, wc, wk, m, eq) - entml(t, g, wc, wk, m, eq) - entph(t, g, wc, wk, m, eq)
    return delta_delta_E