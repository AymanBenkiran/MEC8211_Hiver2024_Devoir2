"""
MEC8211 - Devoir 1 : Verification de code
Fichier : devoir1_tests_unitaires.py
Description : Fichier tertiaire contenant les fonctions de tests unitaires pour le devoir 1
              (a utiliser separement)
Lancer avec :  python3 devoir1_tests_unitaires.py
Auteur.e.s : Amishga Alphonius (2030051), Ayman Benkiran (1984509) et Maxence Farin (2310129)
Date de creation du fichier : 5 février 2024
"""

#%% Importation des modules
import numpy as np

# Importation des fonctions
try:
    from devoir1_functions import (mdf1_rxn_0, mdf2_rxn_0, analytique, erreur_l1,
                                   erreur_l2, erreur_linfty)
    from devoir1_main import (ParametresProb)
    from devoir1_postresults import (ordre_convergence)
except ImportError:
    print("ERREUR ! Il y a une erreur fatale dans le fichier devoir1_functions.py ou devoir1_main")
    
#%% test_errors
def test_errors():
    """
    This function verifies the L1, L2 and L_inf errors between two vectors
    """
    
    # Vectors and scalars of inputs
    x = [1, 2, 3, 4, 5]
    vecteur_x = np.array(x)
    
    y = [1.01, 2.1, 3.2, 4.1, 5.11]
    vecteur_y = np.array(y)
    
    dx = 0.1
    L  = 1.0
    
    # Outputs (execution)
    L1 = erreur_l1(vecteur_x, vecteur_y)
    L2 = erreur_l2(vecteur_x, vecteur_y)
    L_inf = erreur_linfty(vecteur_x, vecteur_y)
    
    # Verification with expected results
    test1 = 0
    test2 = 0
    test3 = 0
    
    # Conditions
    if (abs(L1 - 0.104) < 0.001) :
        test1 = 1
    if (abs(L2 - 0.120167) < 0.000001):
        test2 = 1
    if (abs(L_inf - 0.2) < 0.000001):
        test3 = 1
    
    # Print test results
    if (test1 == 1):
        print("L'erreur L1 est verifiee")
    elif (test1 == 0):
        print("Il y a une erreur dans l'erreur L1")
        print("La solution trouvee est :")
        print(L1)
        
    if (test2 == 1):
        print("L'erreur L2 est verifiee")

    elif (test2 == 0):
        print("Il y a une erreur dans l'erreur L2")
        print("La solution trouvee est :")
        print(L2)
        
    if (test3 == 1):
        print("L'erreur Linf est verifiee")
    elif (test3 == 0):
        print("Il y a une erreur dans l'erreur Linf")
        print("La solution trouvee est :")
        print(L_inf)

#%% test_solution_analytique
def test_solution_analytique():
    
    # Initialisation du cas d'input
    prm_rxn_0 = ParametresProb(0)
    n_noeuds = 100
    R    = 0.5
    mesh    = np.linspace(0, R, n_noeuds)
    
    # Calcul de la solution (output a tester)
    concentration_analytique = analytique(prm_rxn_0, mesh)
    
    # Verification with expected results
    test = 0
    
    # Conditions
    if (abs(concentration_analytique[0] - 7) < 0.00001) and  (abs(concentration_analytique[-1] - 12) < 0.00001):
        test = 1
    
    # Print test results
    if (test == 1):
        print("La solution analytique en r = 0 et r = R est verifiee")

    elif (test == 0):
        print("Il y a une erreur dans la solution analytique")
        print("La solution trouvee est :")
        print("*en r = 0")
        print(concentration_analytique[0])
        print("*en r = R")
        print(concentration_analytique[-1])
    
def test_order_convergence():
    """
    This function verifies the evaluation of order of convergence
    """
    
    # Inputs
    dr_l = [100, 0.1]
    error_l = [0.01, 0.001]
    
    # Output (execution)
    ordre = ordre_convergence(dr_l, error_l)

    # Verification with expected results
    test = 0
    
        # Conditions
    if (abs(ordre - 0.3333) < 0.0001) :
        test = 1
    
        # Print test results
    if (test == 1):
        print("L'ordre de convergence est verifie")

    elif (test == 0):
        print("Il y a une erreur dans l'ordre de convergence")
        print("La solution trouvee pour l'ordre est :")
        print(ordre)
        
#%% Execution des tests unitaires
print("########################")
print("###Verification erreurs")
test_errors()

print("####################################")
print("###Verification solution analytique")
test_solution_analytique()

print("####################################")
print("###Verification ordre de convergence")
test_order_convergence()