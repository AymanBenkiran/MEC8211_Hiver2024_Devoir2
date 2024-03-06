"""
MEC8211 - Devoir 2 : Verification de code - MMS
Fichier : devoir1_functions.py
Description : Fichier secondaire contenant les fonctions pour le devoir 2
              (a utiliser conjointement avec devoir1_main.py)
Auteur.e.s : Amishga Alphonius (2030051), Ayman Benkiran (1984509) et Maxence Farin (2310129)
Date de creation du fichier : 5 février 2024
"""

#%% Importation des modules
import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from typing import Tuple
import sympy as sp
import os
import matplotlib.pyplot as plt


#%% Classe stockant les objets nécessaires à la MMS
class MMS_Func:
    def __init__(self, f, source, x):
        
        self.f = f
        self.df = sp.diff(f,x)
        self.source = source
    
    def lambdify(self, symbols):
        
        self.f = sp.lambdify(symbols, self.f, modules="sympy")
        self.df = sp.lambdify(symbols, self.df, modules="sympy")
        self.source = sp.lambdify(symbols, self.source, modules="sympy")
        
    def evaluate_f(self, variables: Tuple[float]):
        
        x,t = variables
        
        return self.f(x,t)
    
    def evaluate_df(self, variables: Tuple[float]):
        
        x,t = variables
        
        return self.df(x,t)
    
    def evaluate_s(self, variables: Tuple[float]):
        
        x,t = variables

        
        return self.source(x,t)

#%% mdf1_rxn_0
def mdf1_rxn_0(prm_prob, prm_sim):
    """
    Fonction qui resout par le probleme transitoire jusqu'a l'atteinte du regime
    permanent par la methode des differences finies (Schemas d'ordre globaux 1
    en temps et en espace).
        - En r = 0 : Un schema de Gear avant est utilise pour approximer le
                     gradient de concentration (ordre 2)
        - Pour les points centraux :
            - Derivee premiere : differentiation avant (ordre 1)
            - Derivee seconde : differentiation centree (ordre 2)
        - En r = R : Une condition de Dirichlet est imposee

    Entrees :
        - prm_prob : Objet qui contient les parametres du probleme
            - c0 : float - Concentrations initiales [mol/m^3]
            - ce : float - Concentration de sel de l'eau salee [mol/m^3]
            - r : float - Rayon du pilier cylindrique [m]
            - d_eff : float - Coefficient de diffusion effectif de sel dans
              le beton [m^2/s]
            - ordre_de_rxn : int - Ordre de la cinetique de reaction du
              terme source (0 ou 1) []
            - s : float - Terme source constant (reaction d'ordre 0) [mol/m^3/s]
            - k : float - Constante de réaction pour la reaction
              d'ordre 1 [s^{-1}]
        - prm_sim : Objet qui contient les parametres de simulation
            - n_noeuds : int - Nombre de noeuds dans le maillage [noeud]
            - dr : float - Pas en espace des differents maillages [m]
            - dt : float - Pas de temps des differents maillages [s]
            - mesh : array of floats - Vecteur conteant les noeuds (r_i)
              du probleme 1D [m]
            - tol : float - Tolerance relative pour l'atteinte du regime
              permanent []
            - c : array of floats - Solution une fois l'atteinte du regime
              permanent [mol/m^3]
            - tf : float - Temps de fin de la simulation [s]
            - mdf : int - Ordre global en espace de la methode des differences
              finies utilisee []
            - ordre_de_rxn : int - Ordre de la cinetique de reaction du terme
              source []

    Sortie : aucune
    """
    tf = 0
    diff = 1
    n = prm_sim.n_noeuds
    a = np.zeros((n, n))
    b = np.zeros(n)

    # Condition initiale
    c = np.full(n, prm_prob.c0)
    c[-1] = prm_prob.ce

    while diff > prm_sim.tol:
        sum_c_prec = sum(c)

        # Conditions frontieres
        appliquer_conditions_frontieres(a, b, 0., prm_prob.ce)

        # Points centraux
        cst1 = prm_sim.dt*prm_prob.d_eff
        for i in range(1, n-1):
            cst2 = prm_sim.dr**2 * prm_sim.mesh[i]  # r_i * dr^2
            a[i][i-1] = -cst1*prm_sim.mesh[i]
            a[i][i] = cst2 + cst1*(prm_sim.dr + 2*prm_sim.mesh[i])
            a[i][i+1] = -cst1*(prm_sim.dr + prm_sim.mesh[i])
            b[i] = cst2*(c[i] - prm_sim.dt*prm_prob.s)

        # Resolution du systeme lineaire
        c = np.linalg.solve(a, b)
        tf += prm_sim.dt
        diff = abs(sum(c)-sum_c_prec)/abs(sum_c_prec)
    prm_sim.c = c
    prm_sim.tf = tf


#%% mdf2_rxn_0
def mdf2_rxn_0(prm_prob, prm_sim):
    """
    Fonction qui resout par le probleme transitoire jusqu'a l'atteinte du
    regime permanent par la methode des differences finies (Schemas d'ordre
    globaux 1 en temps et 2 en espace).
        - En r = 0 : Un schema Gear avant est utilise pour approximer le
                     gradient de concentration (ordre 2)
        - Pour les points centraux :
            - Derivee premiere : differentiation centree (ordre 2)
            - Derivee seconde : differentiation centree (ordre 3)
        - En r = R : Une condition de Dirichlet est imposee

    Entrees :
        - prm_prob : Objet qui contient les parametres du probleme
            - c0 : float - Concentrations initiales [mol/m^3]
            - ce : float - Concentration de sel de l'eau salee [mol/m^3]
            - r : float - Rayon du pilier cylindrique [m]
            - d_eff : float - Coefficient de diffusion effectif de sel dans
                      le beton [m^2/s]
            - ordre_de_rxn : int - Ordre de la cinetique de reaction du
                             terme source (0 ou 1) []
            - s : float - Terme source constant (reaction d'ordre 0) [mol/m^3/s]
            - k : float - Constante de reaction pour la reaction d'ordre 1 [s^{-1}]
        - prm_sim : Objet qui contient les parametres de simulation
            - n_noeuds : int - Nombre de noeuds dans le maillage [noeud]
            - dr : float - Pas en espace des differents maillages [m]
            - dt : float - Pas de temps des differents maillages [s]
            - mesh : array de floats - Vecteur conteant les noeuds (r_i)
                     du probleme 1D [m]
            - tol : float - Tolerance relative pour l'atteinte du regime
                    permanent []
            - c : array de float - Solution une fois l'atteinte du regime
                  permanent [mol/m^3]
            - tf : float - Temps de fin de la simulation [s]
            - mdf : int - Ordre global en espace de la methode des differences
                    finies utilisee []
            - ordre_de_rxn : int - Ordre de la cinetique de reaction du
                             terme source []
    Sortie : aucune
    """
    tf = 0
    diff = prm_sim.tol+1
    n = prm_sim.n_noeuds
    a = np.zeros((n, n))
    b = np.zeros(n)

    # Condition initiale
    c = np.full(n, prm_prob.c0)
    c[-1] = prm_prob.ce

    while diff > prm_sim.tol:
        sum_c_prec = sum(c)

        # Conditions frontieres
        appliquer_conditions_frontieres(a, b, 0., prm_prob.ce)

        # Points centraux
        cst1 = prm_sim.dt*prm_prob.d_eff
        for i in range(1, n-1):
            cst2 = 2 * prm_sim.dr**2 * prm_sim.mesh[i]  # 2 * r_i * dr^2
            a[i][i-1] = cst1*(prm_sim.dr - 2*prm_sim.mesh[i])
            a[i][i] = cst2 + 4*cst1*prm_sim.mesh[i]
            a[i][i+1] = -cst1*(prm_sim.dr + 2*prm_sim.mesh[i])
            b[i] = cst2*(c[i] - prm_sim.dt*prm_prob.s)

        # Resolution du systeme lineaire
        a_sparse = csc_matrix(a)
        c = spsolve(a_sparse, b)
        tf += prm_sim.dt
        diff = abs(sum(c)-sum_c_prec)/abs(sum_c_prec)

    prm_sim.c = c
    prm_sim.tf = tf


#%% mdf2_rxn_1
def mdf2_rxn_1(prm_prob, prm_sim):
    """
    Fonction qui resout par le probleme transitoire pour une reaction d'ordre 1
    jusqu'a l'atteinte du regime permanent par la methode des differences finies
    (Schemas d'ordre globaux 1 en temps et 2 en espace).
        - En r = 0 : Un schema Gear avant est utilise pour approximer le
                     gradient de concentration (ordre 2)
        - Pour les points centraux :
            - Derivee premiere : differentiation centree (ordre 2)
            - Derivee seconde : differentiation centree (ordre 3)
        - En r = R : Une condition de Dirichlet est imposee

    Entrees :
        - prm_prob : Objet qui contient les parametres du probleme
            - c0 : float - Concentrations initiales [mol/m^3]
            - ce : float - Concentration de sel de l'eau salee [mol/m^3]
            - r : float - Rayon du pilier cylindrique [m]
            - d_eff : float - Coefficient de diffusion effectif de sel dans
                      le beton [m^2/s]
            - ordre_de_rxn : int - Ordre de la cinetique de reaction du
                             terme source (0 ou 1) []
            - s : float - Terme source constant (reaction d'ordre 0) [mol/m^3/s]
            - k : float - Constante de reaction pour la reaction d'ordre 1 [s^{-1}]
        - prm_sim : Objet qui contient les parametres de simulation
            - n_noeuds : int - Nombre de noeuds dans le maillage [noeud]
            - dr : float - Pas en espace des differents maillages [m]
            - dt : float - Pas de temps des differents maillages [s]
            - mesh : array de floats - Vecteur conteant les noeuds (r_i)
                     du probleme 1D [m]
            - tol : float - Tolerance relative pour l'atteinte du regime
                    permanent []
            - c : array de float - Solution une fois l'atteinte du regime
                  permanent [mol/m^3]
            - tf : float - Temps de fin de la simulation [s]
            - mdf : int - Ordre global en espace de la methode des differences
                    finies utilisee []
            - ordre_de_rxn : int - Ordre de la cinetique de reaction du
                             terme source []
            - t : array de float - Vecteur des temps des solutions de la simulation [s]
    Sortie : aucune
    """
    t = 0
    n = prm_sim.n_noeuds
    a = np.zeros((n, n))
    b = np.zeros(n)

    # Condition initiale
    c = np.full(n, prm_prob.c0)
    c[-1] = prm_prob.ce

    while t < prm_sim.tf:

        # Conditions frontieres
        appliquer_conditions_frontieres(a, b, 0., prm_prob.ce)

        # Points centraux
        cst1 = prm_sim.dt*prm_prob.d_eff
        for i in range(1, n-1):
            cst2 = 2 * prm_sim.dr**2 * prm_sim.mesh[i]  # 2 * r_i * dr^2
            a[i][i-1] = cst1*(prm_sim.dr - 2*prm_sim.mesh[i])
            a[i][i] = cst2 + 4*cst1*prm_sim.mesh[i] + cst2*prm_sim.dt*prm_prob.k
            a[i][i+1] = -cst1*(prm_sim.dr + 2*prm_sim.mesh[i])
            b[i] = cst2*c[i]

        # Resolution du systeme lineaire
        a_sparse = csc_matrix(a)
        c = spsolve(a_sparse, b)
        t += prm_sim.dt

        # c_results=np.transpose([c])

        # Stockage des resultats
        prm_sim.c = np.append(prm_sim.c, [c], axis=0)
        prm_sim.t = np.append(prm_sim.t, t)



def mdf2_rxn_1MMS(prm_prob, prm_sim):
    """
     Fonction qui resout par le probleme transitoire pour une reaction d'ordre 1
    jusqu'a l'atteinte du regime permanent par la methode des differences finies
    (Schemas d'ordre globaux 1 en temps et 2 en espace), en s'appuyant sur
    la MMS
    Parameters
    ----------
    prm_prob : ParametresProb
        Objet contenant tous les paramètres du problème et les données MMS
    prm_sim : ParametresSim
        Objet contenant tous les paramètres de la simulation
    Returns
    -------
    None.
    """
    t = 0
    n = prm_sim.n_noeuds
    a = np.zeros((n, n))
    b = np.zeros(n)
    obj_MMS = prm_prob.MMS

    # Condition initiale
    c = np.array([0.0 for i in range(n)])
    for i in range(prm_sim.n_noeuds):
        c[i] =  obj_MMS.evaluate_f((prm_sim.mesh[i],t))

    while t < prm_sim.tf:

        # Conditions frontieres
        neumann = obj_MMS.evaluate_df((prm_sim.mesh[-1],t))
        dirichlet = obj_MMS.evaluate_f((prm_sim.mesh[-1],t))
        appliquer_conditions_frontieres(a, b, neumann, dirichlet)

        # Points centraux
        cst1 = prm_sim.dt*prm_prob.d_eff
        for i in range(1, n-1):
            cst2 = 2 * prm_sim.dr**2 * prm_sim.mesh[i]  # 2 * r_i * dr^2
            a[i][i-1] = cst1*(prm_sim.dr - 2*prm_sim.mesh[i])
            a[i][i] = cst2 + 4*cst1*prm_sim.mesh[i] + cst2*prm_sim.dt*prm_prob.k
            a[i][i+1] = -cst1*(prm_sim.dr + 2*prm_sim.mesh[i])
            t_source = obj_MMS.evaluate_s((prm_sim.mesh[i], t))
            b[i] = c[i] + prm_sim.dt * t_source
            b[i] = cst2 * b[i]

        # Resolution du systeme lineaire
        a_sparse = csc_matrix(a)
        c = spsolve(a_sparse, b)
        t += prm_sim.dt

        # Stockage des resultats
        prm_sim.c = np.append(prm_sim.c, [c], axis=0)
        prm_sim.t = np.append(prm_sim.t, t)


#%% appliquer_conditions_frontieres

def appliquer_conditions_frontieres(a, b, neumann, dirichlet):
    """
    Fonction qui ajoute les conditions frontieres dans le systeme lineaire
        - En r = 0 : Un schema de Gear avant est utilise pour approximer
                     le gradient de concentration (ordre 2) et imposer une
                     condition de symetrie
        - En r = R : Une condition de Dirichlet est imposee

    Entrees :
        - a : array n x n - Matrice des coefficients du systeme lineaire
        - b : array n - Vecteur membre de droite du systeme lineaire
        - dirichlet : float - Condition de Dirichlet imposee en r = R

    Sortie : aucune
    """
    # Gear avant en r = 0
    a[0][0] = -3.
    a[0][1] = 4.
    a[0][2] = -1.
    b[0] = neumann

    # Dirichlet en r = r
    a[-1][-1] = 1.
    b[-1] = dirichlet


#%% analytique
def analytique(prm_prob, mesh, method = "Classic", tools =()):
    """
    Fonction qui calcule la solution analytique aux points du maillage.

    Entrees :
        - prm_prob : Objet qui contient les parametres du probleme
            - c0 : float - Concentrations initiales [mol/m^3]
            - ce : float - Concentration de sel de l'eau salee [mol/m^3]
            - r : float - Rayon du pilier cylindrique [m]
            - d_eff : float - Coefficient de diffusion effectif de sel dans le
                      beton [m^2/s]
            - ordre_de_rxn : int - Ordre de la cinetique de reaction du terme
                             source (0 ou 1) []
            - s : float - Terme source constant (reaction d'ordre 0) [mol/m^3/s]
            - k : float - Constante de réaction pour la reaction
                  d'ordre 1 [s^{-1}]
        - mesh : array de float - Vecteur conteant les noeuds (r_i) du
                 probleme 1D [m]

    Sortie :
        - c : array de float - Le profil de concentration radial analytique
              au regime permanent [mol/m^3]
    """
    if method == "Classic":
        if (prm_prob.ordre_de_rxn == 0):
            c = [0.25*prm_prob.s/prm_prob.d_eff * prm_prob.r**2 * (r**2/prm_prob.r**2 - 1)
                + prm_prob.ce for r in mesh]
        else:
            c = []
            file = f"../data/comsol_solutions/solutions_COMSOL_N{len(mesh)}.csv"

            with (open(file, 'r') as f):
                first_line = f.readline().strip('\n').split(',')
                first_line = first_line[1:]

            df_comsol = pd.read_csv(file)
            for time in first_line:
                c.append(df_comsol.loc[:, f"{time}"].values)

    elif method == "MMS":
        func = prm_prob.MMS

        if (prm_prob.ordre_de_rxn == 0):
            tf, dt = tools
            # dt n'est pas utilisé mais cela permet d'écrire
            #  de manière générique l'appel de analytique()
            c = [float(func.evaluate_f((r, tf))) for r in mesh]

        else:
            t = 0.0
            tf, dt = tools
            c = np.zeros((1, len(mesh)))

            while t < tf:
                c_t = [float(func.evaluate_f((r, t))) for r in mesh]
                c = np.append(c, [c_t], axis=0)
                t += dt

    else:
        raise Exception("Unidentified Method for Function Discretization")

    return c

#%% MMS Functions

def f_MMS(prm_rxn):
    """
    Fonction définissant la fonction choisie pour la MMS, ainsi que
    sa première dérivée spatiale, et le terme source lié à notre équation
    Parameters
    ----------
    prm_rxn : ParametresProb
        Objet contenant tous les paramètres du problème

    Returns
    -------
    obj_MMS: MMS_Func Object
        Objet contenant toutes les données liées à notre fonction dans
        la MMS
    """
    
    r, t = sp.symbols('r t')
    symbols = [r, t]
    C_MMS = sp.cos(sp.pi * r / (2 * prm_rxn.r) + sp.pi/2) * sp.exp(- sp.pi * sp.Pow(t,1))
    if prm_rxn.ordre_de_rxn == 0:
        source = sp.diff(C_MMS, t) - prm_rxn.d_eff * sp.diff(sp.diff(C_MMS,r), r) + prm_rxn.s
    elif prm_rxn.ordre_de_rxn == 1:
        source = sp.diff(C_MMS, t) - prm_rxn.d_eff * sp.diff(sp.diff(C_MMS,r), r) + prm_rxn.k * C_MMS
    
    obj_MMS = MMS_Func(C_MMS, source, r)
    obj_MMS.lambdify(symbols)
    
    return obj_MMS

def plot_MMS(prm_rxn, path_save = ''):
    """
    Fonction premettant de tracer en 2D la solution MMs choisie, selon la 
    position spatialle et temporelle
    Parameters
    ----------
    prm_rxn : ParametresProb
        Objet contenant tous les paramètres du problème
    path_save: 
        Desired Directory When Saving The Graph

    Returns
    -------
    aucun
    """
    r, t = sp.symbols('r t')
    C = sp.cos(sp.pi * r / (2 * prm_rxn.r) + sp.pi/2) * sp.exp(-sp.pi * t)
    
    # Convert the SymPy expression to a NumPy-compatible function
    f_func = sp.lambdify((r, t), C, modules='numpy')

    # Generate x and y values using numpy
    x_vals = np.linspace(0, 0.5, 100)
    y_vals = np.linspace(0, 1.0, 100)

    # Create a meshgrid from x and y
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute function values for each point in the meshgrid
    Z = f_func(X, Y)

    # Plot the 2D function
    plt.figure()
    plt.contourf(X, Y, Z, cmap='viridis')  # Adjust the colormap as needed
    plt.colorbar()                         # Add a colorbar for reference
    plt.xlabel('Radial position r (m)')
    plt.ylabel('Time t (s)')
    title = "2D Plot of MMS solution C(r, t)"
    plt.title(title)
    plt.grid(True)
    
    # Save the figure in data folder
    if path_save != '':
        os.chdir(path_save)
        if title != '':
            plt.savefig(title+".png", dpi=600)
        else:
            plt.savefig("Solution C_mms.png", dpi=600)
    
    plt.show()
    
#%% erreur_l1
def erreur_l1(c_num, c_analytique):
    """
    Fonction qui calcule la valeur de l'erreur L1 de la solution numerique obtenue

    Entree :
        - c_num : array de float - Solution numerique du probleme 1D [mol/m^3]
        - c_analytique : array de float - Solution analytique du probleme 1D [mol/m^3]

    Sortie :
        - erreur : float - Norme de l'erreur L1 de la solution numerique [mol/m^3]
    """

    erreur = sum(abs(ci_num - ci_analytique)
                  for ci_num, ci_analytique in zip(c_num, c_analytique))
    erreur *= 1/len(c_num)
    return erreur


#%% erreur_l2
def erreur_l2(c_num, c_analytique):
    """
    Fonction qui calcule la valeur de l'erreur L2 de la solution numerique obtenue

    Entree :
        - c_num : array de float - Solution numerique du probleme 1D [mol/m^3]
        - c_analytique : array de float - Solution analytique du probleme 1D [mol/m^3]

    Sortie :
        - erreur : float - Norme de l'erreur L2 de la solution numerique [mol/m^3]
    """
    erreur = sum(abs(ci_num - ci_analytique)**2
                  for ci_num, ci_analytique in zip(c_num, c_analytique))
    erreur *= 1/len(c_num)
    erreur = np.sqrt(erreur)
    return erreur


#%% erreur_linfty
def erreur_linfty(c_num, c_analytique):
    """
    Fonction qui calcule la valeur de l'erreur L_infty de la solution numerique obtenue

    Entree :
        - c_num : array de float - Solution numerique du probleme 1D [mol/m^3]
        - c_analytique : array de float - Solution analytique du probleme 1D [mol/m^3]

    Sortie :
        - erreur : float - Norme de l'erreur L_infty de la solution numerique [mol/m^3]
    """
    erreur = max(abs(ci_num - ci_analytique)
                  for ci_num, ci_analytique in zip(c_num, c_analytique))
    return erreur

#%% Getting Directories:
    
def get_path_results(main_path, file_sep_str, folder):
    """
    Fonction qui trouve ou cree le chemin demande pour le stockage des resultats
    
    Entree:
        - main_path: STR - Chemin d'acces au code source
        - file_sep_str: STR - Separateurs de fichier dans le chemin utilise selon de systeme d'exploitation
        - folder: STR - Dossier desire
    
    Sortie:
        - path_results: STR Chemin d'acces au dossier de resultats concerne
    """
    
    general_folder, cur_dir = os.path.split(main_path)

    if os.path.exists(general_folder+file_sep_str+str(folder)):

        path_results = general_folder+file_sep_str+str(folder)

    # Le dossier desire n'existe pas
    else:

        os.mkdir(general_folder+file_sep_str+str(folder))
        path_results = general_folder+file_sep_str+str(folder)
    
    return path_results
