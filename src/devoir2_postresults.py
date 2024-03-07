"""
MEC8211 - Devoir 2 : Verification de code - MMS
Fichier : devoir2_post_results.py
Description : Fichier secondaire 
              (a utiliser conjointement avec devoir2_main.py)
Auteur.e.s : Amishga Alphonius (2030051), Ayman Benkiran (1984509) et Maxence Farin (2310129)
Date de creation du fichier : 10 février 2024
"""

#%% Importation des modules
import numpy as np
import matplotlib.pyplot as plt
import os
from math import ceil
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


""" Post Results """

#%% Solution

def plot_stationnary_compar(r_l, st_sol, sim_sol,
                            plotting = False,
                            path_save = '',
                            title = '',
                            num_label = ''):
    """ Plot the spatial distribution of salt concentration of
    the stationnary solution and the finite differences solution
    Entrees:
        - r_l: ARRAY of Spatial Nodes
        - st_sol: ARRAY of Stationnary Solution Values at Nodes
        - sim_sol: ARRAY of Simulation Solution Values at Nodes
        - plotting: BOOL to Determine if We Want to Plot Here the Graph
        - path_save: STR Desired Directory When Saving The Graph
        - title: STR Desired Title of the Graph When Saving The File
        - num_label: STR numeric solution label

    Sortie:
        - FIGURE Graphique de la concentration en sel dans le cylindre
        """

    plt.figure()
    plt.plot(r_l, st_sol, label = 'Solution Analytique')
    plt.plot(r_l, sim_sol, "--", label = num_label)
    plt.xlabel("Rayon du Cylindre (m)")
    plt.ylabel(r"Concentration en sel (mol/m$^3$)")
    plt.title("Comparaison des solutions analytique et numérique")
    plt.legend()
    plt.grid(linestyle = '--')
    if path_save != '':
        os.chdir(path_save)
        if title != '':
            plt.savefig(title+".png", dpi=600)
        else:
            plt.savefig("ComparaisonSolAnalytique.png", dpi=600)

    if plotting is True:
        plt.show()

def plot_transient_compar(r_l, comsol_sols, sim_sols,
                            plotting = False,
                            path_save = '',
                            title = '',
                            num_label = ''):
    """ Plot the spatial distribution of salt concentration of
    the stationnary solution and the finite differences solution
    Entrees:
        - r_l: ARRAY of Spatial Nodes
        - st_sol: ARRAY of Stationnary Solution Values at Nodes
        - sim_sol: ARRAY of Simulation Solution Values at Nodes
        - plotting: BOOL to Determine if We Want to Plot Here the Graph
        - path_save: STR Desired Directory When Saving The Graph
        - title: STR Desired Title of the Graph When Saving The File
        - num_label: STR numeric solution label

    Sortie:
        - FIGURE Graphique de la concentration en sel dans le cylindre
        """

    plt.figure()
    i = 0
    for c_n, c_a in zip(sim_sols, comsol_sols):
        if i == 0:
            plt.plot(r_l, c_a, "r", label = 'Solution Analytique')
            plt.plot(r_l, c_n, "b--", label = num_label)
            i += 1
        else:
            plt.plot(r_l, c_a, "r")
            plt.plot(r_l, c_n, "b--")
    plt.xlabel("Rayon du Cylindre (m)")
    plt.ylabel(r"Concentration en sel (mol/m$^3$)")
    plt.title("Comparaison des solutions analytique et numérique")
    plt.legend()
    plt.grid(linestyle = '--')
    if path_save != '':
        os.chdir(path_save)
        if title != '':
            plt.savefig(title+".png", dpi=600)
        else:
            plt.savefig("ComparaisonSolAnalytique.png", dpi=600)

    if plotting is True:
        plt.show()

#%% Video

# def make_film (r, y, c, )

#%% Erreurs

def convergence_compar(norm_l, n_l,
                       typAnalyse = "Spatial",
                       n_fit = -1,
                       path_save = '',
                       title = ''):
    """ Construit et affiche un graphe de convergences des erreurs selon
    les différentes normes utilisées et enregistre le graphique dans un
    dossier spécifié
    Entrees:
        - norm_l: ARRAY of Tuples with the norm's name'
        - n_l: ARRAY of Abscisses for the Convergence Graph
        - typAnalyse: STR Type of Convergence Analysis
        - path_save: STR Desired Directory When Saving The Graph
        - title: STR Desired Title of the Graph When Saving The File

    Sortie:
        - FIGURE Graphique de la concentration en sel dans le cylindre
        """

    plt.figure()
    for name_norm, norm_l in norm_l:
        if n_fit != -1:
            n_lfit, norm_lfit = n_l[:n_fit], norm_l[:n_fit]
        else:
            n_lfit, norm_lfit = n_l, norm_l
        
        ordre, cste = np.polyfit(np.log10(n_lfit), np.log10(norm_lfit), 1)
        ordre, cste = format((ceil(100*ordre)/100), '.2f'), format((ceil(100*cste)/100), '.2f')

        # if typAnalyse == "Spatial":

        label_norm = name_norm + '   ' + f'$log(\epsilon) = {ordre} log(\Delta r) + {cste}$'
        plt.loglog(n_l, norm_l, "s-", label = label_norm)

        # else:
        #     plt.loglog(n_l, norm_l, "s-", label = name_norm)

    if typAnalyse == "Spatial":

        plt.title("Convergence Spatiale: Évolution des Erreurs dans le cylindre")
        plt.xlabel(r"$\Delta r$ [m]")

    if typAnalyse == "Temporal":

        plt.title("Convergence Temporelle: Évolution des Erreurs dans le cylindre")
        plt.xlabel(r"$\Delta t$ [s]")

    plt.ylabel(r"Erreur [mol/m$^3$]")
    plt.grid(linestyle = '-')
    plt.legend()
    if path_save != '':
        os.chdir(path_save)
        if title != '':
            plt.savefig(title+".png", dpi=600)
        else:
            plt.savefig(typAnalyse + "_Convergence.png", dpi=600)
    plt.show()

def ordre_convergence(dstep_l, error_l, n_fit = -1):
    """
    Utilise la bibliotheque Numpy pour determiner l'ordre de convergence d'un
    schema numerique
    Entrees:
        - dstep_l: ARRAY ou LIST, liste contenant les points de la discretisation
        spatiale ou temporelle
        - error_l: ARRAY ou LIST, liste contenant les erreurs du schema numerique
        etant donne une norme et une discretisation temporelle ou spatiale
    Sorties:
        - m: FLOAT Ordre de convergence du schéma
    """
    if n_fit != -1:
        dstep_lfit, error_lfit = dstep_l[:n_fit], error_l[:n_fit]
    else:
        dstep_lfit, error_lfit = dstep_l, error_l
    ordre, b = np.polyfit(np.log10(dstep_lfit), np.log10(error_lfit), 1)

    return ordre
