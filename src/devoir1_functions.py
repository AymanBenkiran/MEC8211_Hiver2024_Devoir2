"""
MEC8211 - Devoir 1 : Verification de code
Fichier : devoir1_functions.py
Description : Fichier secondaire contenant les fonctions pour le devoir 1
              (a utiliser conjointement avec devoir1_main.py)
Auteur.e.s : Amishga Alphonius (2030051), Ayman Benkiran (1984509) et Maxence Farin (2310129)
Date de creation du fichier : 5 février 2024
"""

#%% Importation des modules
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
import os

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
        appliquer_conditions_frontieres(a, b, prm_prob.ce)

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
        appliquer_conditions_frontieres(a, b, prm_prob.ce)

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


#%% appliquer_conditions_frontieres

def appliquer_conditions_frontieres(a, b, dirichlet):
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
    b[0] = 0.

    # Dirichlet en r = r
    a[-1][-1] = 1.
    b[-1] = dirichlet


#%% analytique
def analytique(prm_prob, mesh):
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
    c = [0.25*prm_prob.s/prm_prob.d_eff * prm_prob.r**2 * (r**2/prm_prob.r**2 - 1)
         + prm_prob.ce for r in mesh]
    return c


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
