#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour tester les guardrails avec le modèle Mistral.
"""

import time
import json
import argparse
import pandas as pd
from pathlib import Path
import nltk
import os
import colorama
from colorama import Fore, Style

# Initialiser colorama pour une sortie colorée
colorama.init()

# Télécharger les ressources NLTK nécessaires
try:
    nltk.download('punkt')
except Exception as e:
    print(f"Erreur lors du téléchargement des ressources NLTK, mais continuons quand même: {e}")

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from guardrails import Guard
from guardrails.hub import ToxicLanguage


def setup_model(model_name="mistralai/Mistral-7B-Instruct-v0.3", use_8bit=True):
    """
    Configuration du modèle de génération de texte.
    
    Args:
        model_name (str): Nom du modèle Hugging Face à utiliser
        use_8bit (bool): Utiliser la quantification 8-bit pour économiser de la mémoire
        
    Returns:
        tokenizer, model: Tokenizer et modèle configurés
    """
    print(f"{Fore.YELLOW}Chargement du modèle {model_name}...{Style.RESET_ALL}")
    
    # Chargement du modèle avec quantification 8-bit si demandé
    if use_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return tokenizer, model


def setup_guardrail(threshold=0.5, validation_method="sentence", on_fail="fix"):
    """
    Configuration du guardrail anti-toxicité.
    
    Args:
        threshold (float): Seuil de confiance pour la toxicité.
        validation_method (str): Méthode de validation ('sentence' ou 'full').
        on_fail (str): Action à prendre en cas d'échec.
        
    Returns:
        Guard: L'objet Guard configuré.
    """
    print(f"{Fore.YELLOW}Configuration du guardrail avec threshold={threshold}, method={validation_method}{Style.RESET_ALL}")
    guard = Guard().use(
        ToxicLanguage, 
        threshold=threshold, 
        validation_method=validation_method, 
        on_fail=on_fail
    )
    return guard


def generate_response(tokenizer, model, prompt, max_length=512):
    """
    Génère une réponse du modèle en utilisant le format de chat approprié.
    
    Args:
        tokenizer: Tokenizer pour le modèle.
        model: Modèle de génération de texte.
        prompt (str): Le prompt pour le modèle.
        max_length (int): Longueur maximale de la réponse.
        
    Returns:
        str: La réponse générée.
    """
    # Construction du message au format chat
    messages = [{"role": "user", "content": prompt}]
    
    # Formatage pour Mistral, qui utilise un format spécifique pour les chats
    chat_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenisation et génération
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids, 
        max_new_tokens=max_length, 
        do_sample=True, 
        temperature=0.7,
        top_p=0.9,
    )
    
    # Décodage de la réponse générée
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraction de la réponse du modèle (après le prompt)
    # Cette partie peut nécessiter des ajustements selon le format exact de sortie
    response = full_response.split("[/INST]")[-1].strip()
    
    return response


def evaluate_prompt(tokenizer, model, guard, prompt, history):
    """
    Évalue un prompt avec le guardrail.
    
    Args:
        tokenizer: Tokenizer pour le modèle.
        model: Modèle de génération de texte.
        guard: Guardrail configuré.
        prompt (str): Le prompt à tester.
        history (list): Historique des résultats précédents.
        
    Returns:
        dict: Résultat de l'évaluation.
    """
    try:
        print(f"{Fore.CYAN}Génération de la réponse...{Style.RESET_ALL}")
        
        # Génération de la réponse
        start_time_generation = time.time()
        model_response = generate_response(tokenizer, model, prompt)
        generation_time = time.time() - start_time_generation
        
        print(f"{Fore.GREEN}Réponse brute du modèle :{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{model_response}{Style.RESET_ALL}")
        print("\n" + "-" * 80 + "\n")
        
        # Application du guardrail
        print(f"{Fore.CYAN}Application du guardrail...{Style.RESET_ALL}")
        start_time_validation = time.time()
        
        # Utiliser try-except pour gérer différentes versions de guardrails-ai
        try:
            # Nouvelle version de guardrails-ai (retourne un ValidationOutcome)
            outcome = guard.parse(model_response)
            # Extraire les valeurs de l'objet ValidationOutcome
            raw_output = model_response
            # Conserver l'objet ValidationOutcome pour l'affichage, mais utiliser validated_output pour les comparaisons
            validated_output_str = str(outcome.validated_output) if hasattr(outcome, 'validated_output') else str(outcome)
            validation_info = {'errors': outcome.validation_summaries} if hasattr(outcome, 'validation_summaries') else None
            print(f"{Fore.CYAN}Utilisation de la nouvelle API guardrails-ai{Style.RESET_ALL}")
        except (AttributeError, TypeError):
            # Ancienne version (retourne un tuple)
            try:
                parse_result = guard.parse(model_response)
                if isinstance(parse_result, tuple):
                    if len(parse_result) == 3:
                        raw_output, validated_output, validation_info = parse_result
                    else:
                        raw_output, validated_output = parse_result
                        validation_info = None
                else:
                    # Si ni tuple ni ValidationOutcome, utiliser comme validated_output
                    raw_output = model_response
                    validated_output = parse_result
                    validated_output_str = str(validated_output)
                    validation_info = None
                print(f"{Fore.CYAN}Utilisation de l'ancienne API guardrails-ai{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Erreur lors du parsing: {str(e)}{Style.RESET_ALL}")
                raise
        
        validation_time = time.time() - start_time_validation
        
        # Vérifier si le guardrail a été activé (contenu modifié)
        is_activated = raw_output != validated_output_str
        activation_reason = None
        
        if is_activated and validation_info and isinstance(validation_info, dict) and 'errors' in validation_info:
            activation_reason = validation_info['errors']
        
        # Pour l'historique, on utilise l'objet complet outcome pour l'affichage, 
        # mais aussi une version string pour les comparaisons et la sérialisation
        result = {
            'prompt': prompt,
            'model_response': model_response,
            'validated_response': outcome if 'outcome' in locals() else validated_output,
            'validated_response_str': validated_output_str if 'validated_output_str' in locals() else str(validated_output),
            'is_activated': is_activated,
            'activation_reason': str(activation_reason) if activation_reason else None,
            'generation_time': generation_time,
            'validation_time': validation_time,
            'total_time': generation_time + validation_time
        }
        
        # Ajouter le résultat à l'historique
        history.append(result)
        
        # Afficher les résultats - ici on peut afficher l'objet ValidationOutcome complet
        print(f"{Fore.GREEN}Réponse validée :{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{result['validated_response']}{Style.RESET_ALL}")
        print("\n" + "-" * 80 + "\n")
        
        print(f"{Fore.YELLOW}Métriques :{Style.RESET_ALL}")
        print(f"- Guardrail activé: {Fore.RED if is_activated else Fore.GREEN}{is_activated}{Style.RESET_ALL}")
        if is_activated and activation_reason:
            print(f"- Raison: {Fore.RED}{activation_reason}{Style.RESET_ALL}")
        print(f"- Temps de génération: {Fore.CYAN}{generation_time:.4f}{Style.RESET_ALL} secondes")
        print(f"- Temps de validation: {Fore.CYAN}{validation_time:.4f}{Style.RESET_ALL} secondes")
        print(f"- Temps total: {Fore.CYAN}{(generation_time + validation_time):.4f}{Style.RESET_ALL} secondes")
        
        return result
        
    except Exception as e:
        print(f"{Fore.RED}Erreur lors du traitement du prompt: {prompt}{Style.RESET_ALL}")
        print(f"{Fore.RED}Type d'erreur: {type(e).__name__}{Style.RESET_ALL}")
        print(f"{Fore.RED}Exception: {str(e)}{Style.RESET_ALL}")
        
        # Afficher plus de détails pour le débogage
        import traceback
        print(f"{Fore.RED}Traceback: {traceback.format_exc()}{Style.RESET_ALL}")
        
        result = {
            'prompt': prompt,
            'model_response': "ERROR",
            'validated_response': "ERROR",
            'validated_response_str': "ERROR",
            'is_activated': None,
            'activation_reason': f"{type(e).__name__}: {str(e)}",
            'generation_time': None,
            'validation_time': None,
            'total_time': None
        }
        
        # Ajouter le résultat à l'historique
        history.append(result)
        
        return result


def save_history(history, output_file):
    """
    Sauvegarde l'historique des tests.
    
    Args:
        history (list): Liste des résultats des tests.
        output_file (str): Chemin du fichier de sortie.
    """
    # Créer le répertoire si nécessaire
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Convertir l'historique en DataFrame
    serializable_history = []
    for item in history:
        serializable_item = {}
        for key, value in item.items():
            # Convertir ValidationOutcome en string
            if hasattr(value, '__class__') and value.__class__.__name__ == 'ValidationOutcome':
                if hasattr(value, 'validated_output'):
                    serializable_item[key] = str(value.validated_output)
                else:
                    serializable_item[key] = str(value)
            else:
                serializable_item[key] = value
        serializable_history.append(serializable_item)
    
    # Convertir en DataFrame pour CSV
    df = pd.DataFrame(serializable_history)
    
    # Enregistrer les résultats
    df.to_csv(output_file, index=False)
    
    # Sauvegarder également au format JSON pour une analyse plus détaillée
    json_output = output_file.replace('.csv', '.json')
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(serializable_history, f, ensure_ascii=False, indent=4)
    
    print(f"{Fore.GREEN}Historique sauvegardé dans {output_file} et {json_output}{Style.RESET_ALL}")


def main():
    parser = argparse.ArgumentParser(description='Test interactif de guardrails avec Mistral')
    parser.add_argument('--output', type=str, default='data/mistral_results.csv',
                       help='Fichier de sortie pour les résultats')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Seuil de confiance pour la toxicité')
    parser.add_argument('--method', type=str, default='sentence', choices=['sentence', 'full'],
                       help='Méthode de validation')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3',
                       help='Modèle à utiliser')
    parser.add_argument('--no-8bit', action='store_true',
                       help='Désactiver la quantification 8-bit')
    args = parser.parse_args()
    
    # Configurer le modèle et le guardrail
    tokenizer, model = setup_model(model_name=args.model, use_8bit=not args.no_8bit)
    guard = setup_guardrail(threshold=args.threshold, validation_method=args.method)
    
    # Historique des tests
    history = []
    
    # Boucle interactive
    while True:
        print("\n" + "=" * 80)
        print(f"{Fore.YELLOW}Mode interactif avec {args.model} - Entrez un prompt pour tester le guardrail{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Commandes spéciales:{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}!quit{Style.RESET_ALL} - Quitter le programme")
        print(f"  {Fore.CYAN}!stats{Style.RESET_ALL} - Afficher les statistiques actuelles")
        print(f"  {Fore.CYAN}!save{Style.RESET_ALL} - Sauvegarder l'historique")
        print(f"  {Fore.CYAN}!clear{Style.RESET_ALL} - Effacer l'écran")
        print("=" * 80 + "\n")
        
        prompt = input(f"{Fore.GREEN}Prompt > {Style.RESET_ALL}")
        
        if prompt.lower() == "!quit":
            # Sauvegarder automatiquement l'historique avant de quitter
            if history:
                save_history(history, args.output)
            print(f"{Fore.YELLOW}Au revoir !{Style.RESET_ALL}")
            break
            
        elif prompt.lower() == "!stats":
            if history:
                from src.interactive_guardrails import analyze_history, display_metrics
                metrics = analyze_history(history)
                display_metrics(metrics)
            else:
                print(f"{Fore.RED}Aucun historique disponible.{Style.RESET_ALL}")
                
        elif prompt.lower() == "!save":
            if history:
                save_history(history, args.output)
            else:
                print(f"{Fore.RED}Aucun historique à sauvegarder.{Style.RESET_ALL}")
                
        elif prompt.lower() == "!clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            
        elif prompt.strip() == "":
            print(f"{Fore.RED}Veuillez entrer un prompt non vide.{Style.RESET_ALL}")
            
        else:
            # Évaluer le prompt
            evaluate_prompt(tokenizer, model, guard, prompt, history)


if __name__ == "__main__":
    main()
