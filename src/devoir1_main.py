'''
MEC8211 - Devoir 1 : Verification de code

Fichier : devoir1_main.py
Description : Fichier principal pour le devoir 1
Lancer avec :  python3 devoir1_main.py
Auteur.e.s : Amishga Alphonius (2030051), Ayman Benkiran (1984509) et Maxence Farin (2310129)
Date de creation du fichier : 5 fevrier 2024
'''

#%% Importation des modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Importation des fonctions
try:
    from devoir1_functions import (mdf1_rxn_0, mdf2_rxn_0, analytique, erreur_l1,
                                   erreur_l2, erreur_linfty, get_path_results)
except ImportError:
    print("ERREUR ! Il y a une erreur fatale dans le fichier devoir1_functions.py")

try:
    from devoir1_postresults import (plot_stationnary_compar, convergence_compar,
                                     ordre_convergence)
except ImportError:
    print("ERREUR ! Il y a une erreur fatale dans le fichier devoir1_postresults.py")

#%% Donnees du probleme et definition des classes

class ParametresProb:
    """
    Classe qui contient les differents parametres du probleme selon le type de terme source

    Entree:
        - ordre_de_rxn : int - Ordre de la cinetique de reaction du terme source (0 ou 1) []

    Attributs:
        - c0 : float - Concentrations initiales [mol/m^3]
        - ce : float - Concentration de sel de l'eau salee [mol/m^3]
        - r : float - Rayon du pilier cylindrique [m]
        - d_eff : float - Coefficient de diffusion effectif de sel dans le
          beton [m^2/s]
        - ordre_de_rxn : int - Ordre de la cinetique de reaction du terme source (0 ou 1) []
        - s : float - Terme source constant (reaction d'ordre 0) [mol/m^3/s]
        - k : float - Constante de réaction pour la reaction d'ordre 1 [s^{-1}]
    """
    def __init__(self, ordre_de_rxn: int):
        self.c0 = 0.0
        self.ce = 12.0
        self.r = 1*0.5
        self.d_eff = 1e-10
        self.ordre_de_rxn = ordre_de_rxn
        if self.ordre_de_rxn == 0:
            self.s = 8e-9
        elif self.ordre_de_rxn == 1:
            self.k = 4e-9
            raise ValueError("Cet ordre n'est pas encore implememte.")
        else:
            raise ValueError("L'ordre de reaction doit etre de 0 ou 1.")


class ParametresSim:
    """
    Classe qui contient les parametres de simulation pour une simulation donnee

    Entrees:
        - prm_rxn : Objet contenant les donnees du probleme
            - r : Rayon du pilier cylindrique [m^3]
            - ordre_de_rxn : int - Ordre de la cinetique de reaction du terme
              source (0 ou 1) []
        - p_n_noeuds : int - Nombre de noeuds dans le maillage [noeud]
        - p_mdf : int - Ordre global en espace de la methode des differences finies utilisee []

    Attributs :
        - n_noeuds : int - Nombre de noeuds dans le maillage [noeud]
        - dr : float - Pas en espace des differents maillages [m]
        - dt : float - Pas de temps des differents maillages [s]
        - mesh : array of floats - Vecteur conteant les noeuds (r_i) du
          probleme 1D [m]
        - tol : float - Tolerance relative pour l'atteinte du regime
          permanent []
        - c : array of floats - Solution une fois l'atteinte du regime
          permanent [mol/m^3]
        - tf : float - Temps de fin de la simulation [s]
        - mdf : int - Ordre global en espace de la methode des differences finies utilisee []
        - ordre_de_rxn : int - Ordre de la cinetique de reaction du terme source []
    """
    def __init__(self, prm_rxn: ParametresProb, p_n_noeuds: int, p_mdf: int):
        self.n_noeuds = p_n_noeuds
        self.dr = prm_rxn.r/(self.n_noeuds-1)
        self.dt = 0.5*1e-2**2/prm_rxn.d_eff
        self.mesh = np.linspace(0, prm_rxn.r, self.n_noeuds)
        self.tol = 1e-14
        self.c = np.zeros(self.n_noeuds)
        self.tf = 0
        self.mdf = p_mdf
        self.ordre_de_rxn = prm_rxn.ordre_de_rxn


#%% Initialisation des objects contenants les parametres du probleme pour une reaction d'ordre 0
#   et 1

prm_rxn_0 = ParametresProb(ordre_de_rxn=0)
# TODO
# prm_rxn_1 = ParametresProb(ordre_de_rxn=1) # pas encore implemente


#%% Discretisation pour la reaction d'ordre 0

n_noeuds_liste = [10, 20, 40, 80]   # Liste de nombre de noeuds pour les differents
                                    # maillages [noeud]

# Initialisation des differents maillages a l'etude
prm_simulations_mdf1_rxn0 = []
prm_simulations_mdf2_rxn0 = []
for i, n_noeuds in enumerate(n_noeuds_liste):
    prm_simulations_mdf1_rxn0.append(ParametresSim(prm_rxn_0, n_noeuds, 1))
    prm_simulations_mdf2_rxn0.append(ParametresSim(prm_rxn_0, n_noeuds, 2))

#%% Création des Répertoires de Solution

actual_path = os.getcwd()
if os.name == "nt":
    file_sep_str = "\\"
else:
    file_sep_str = "/"

path_donnees = get_path_results(actual_path, file_sep_str, 'data')
path_analyse = get_path_results(actual_path, file_sep_str, 'results')

#%% Resolution du probleme

for prm_simulation in [prm_simulations_mdf1_rxn0, prm_simulations_mdf2_rxn0]:
    print("****************************************************************************")
    for prm_sim in prm_simulation:
        mdf_i = prm_sim.mdf
        ordre_de_rxn = prm_sim.ordre_de_rxn

        # Calcul de la concentration au regime permanent
        # pylint: disable-next=exec-used
        exec(f"mdf{mdf_i}_rxn_{ordre_de_rxn}(prm_rxn_0, prm_sim)")

        # Exportation des solutions dans des fichiers csv
        exported_data = pd.DataFrame({'r': prm_sim.mesh, 'C(r)': prm_sim.c})

        exported_data.to_csv(f"{path_donnees}/mdf{mdf_i}_rxn{ordre_de_rxn}_noeuds_"
                             f"{str(prm_sim.n_noeuds).zfill(3)}.csv", index=False)

        # Affichage au terminal
        print("****************************************************************************")
        print(f"Ordre de la methode des differences finies : {mdf_i}")
        print(f"Ordre de la reaction du terme source : {ordre_de_rxn}")
        print(f"Nombre de noeuds : {prm_sim.n_noeuds} noeuds")
        print(f"dr = {prm_sim.dr} m")
        print(f"dt = {prm_sim.dt} s")
        print(f"tf = {prm_sim.tf} s")
print("****************************************************************************")

#%% Etude de l'erreur

# Initialisation de listes pour entreposer les donnees pour l'affichage graphique
dr = []                     # Pas en espace [m]
liste_erreur_l1 = []        # Erreur L1 [mol/m^3]
liste_erreur_l2 = []        # Erreur L2
liste_erreur_linfty = []    # Erreur Linfty

# Calcul des erreurs pour les differentes simulations
for prm_simulation in [prm_simulations_mdf1_rxn0, prm_simulations_mdf2_rxn0]:
    dr = []
    liste_erreur_l1 = []
    liste_erreur_l2 = []
    liste_erreur_linfty = []
    for prm_sim in prm_simulation:
        mdf_i = prm_sim.mdf
        ordre_de_rxn = prm_sim.ordre_de_rxn
        dr.append(prm_sim.dr)

        # Solution analytique [mol/m^3]
        c_analytique = analytique(prm_rxn_0, prm_sim.mesh)

        liste_erreur_l1.append(erreur_l1(prm_sim.c, c_analytique))
        liste_erreur_l2.append(erreur_l2(prm_sim.c, c_analytique))
        liste_erreur_linfty.append(erreur_linfty(prm_sim.c, c_analytique))
        
        # Comparaison graphique des solutions
        n_noeuds = prm_sim.n_noeuds
        title_analytique = f"Comparaison_Analytique_mdf{mdf_i}_noeuds{n_noeuds}"
        plot_stationnary_compar(prm_sim.mesh, c_analytique, prm_sim.c,
                                plotting = False,
                                path_save = path_analyse,
                                title = title_analytique,
                                num_label = f"Solution par Différences Finies (mdf{mdf_i}, n={n_noeuds})")
        
    # Exportation des valeurs d'erreur dans un fichier csv
    exported_data = pd.DataFrame({'dr': dr, 'L1_error': liste_erreur_l1,
                                  'L2_error': liste_erreur_l2,
                                  'Linfty_error': liste_erreur_linfty})
    
    exported_data.to_csv(f"{path_analyse}/erreurs_mdf{mdf_i}_rxn{ordre_de_rxn}.csv", index=False)


    # Affichage graphique
    title_errors = f"erreurs_mdf{mdf_i}"
    errors_l = [('Erreur $L^1$',liste_erreur_l1), 
                ('Erreur $L^2$',liste_erreur_l2), 
                ('Erreur $L^\inf$',liste_erreur_linfty)]
    convergence_compar(errors_l, dr, 
                           typAnalyse = "Spatial", 
                           path_save = path_analyse,
                           title = title_errors)

    # Verification des ordres numeriques
    for name_error, norm in errors_l:
        name_norm = name_error.split()[-1]
        mdf_i = prm_simulation[0].mdf
        ordre = ordre_convergence(dr, norm)
        print(f"L'ordre observe du schema numerique est {ordre} pour la norme"
              f" {name_norm}")
        print(f"L'ordre théorique du schéma est {mdf_i}")
