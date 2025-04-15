#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour tester interactivement les guardrails anti-toxicité.
Ce script permet d'entrer des prompts manuellement et d'obtenir les métriques en temps réel.
Supporte à la fois les modèles TinyLlama et Mistral.
"""

import time
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import nltk
import os
import colorama
from colorama import Fore, Style

# Initialiser colorama pour une sortie colorée
colorama.init()

# Télécharger les ressources NLTK nécessaires
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    print(f"Erreur lors du téléchargement des ressources NLTK, mais continuons quand même: {e}")

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from guardrails import Guard
from guardrails.hub import ToxicLanguage


def setup_pipeline_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Configuration du modèle de génération de texte avec pipeline.
    
    Args:
        model_name (str): Nom du modèle à utiliser
        
    Returns:
        pipeline: Pipeline de génération de texte.
    """
    print(f"{Fore.YELLOW}Chargement du modèle {model_name} avec pipeline...{Style.RESET_ALL}")
    # Utilisation du CPU (device=-1) pour éviter les problèmes de mémoire GPU
    pipe = pipeline("text-generation", model=model_name, device=-1)
    return pipe


def setup_advanced_model(model_name="mistralai/Mistral-7B-Instruct-v0.3", use_8bit=True):
    """
    Configuration avancée du modèle de génération de texte.
    
    Args:
        model_name (str): Nom du modèle Hugging Face à utiliser
        use_8bit (bool): Utiliser la quantification 8-bit pour économiser de la mémoire
        
    Returns:
        tokenizer, model: Tokenizer et modèle configurés
    """
    print(f"{Fore.YELLOW}Chargement du modèle avancé {model_name}...{Style.RESET_ALL}")
    
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


def generate_pipeline_response(pipe, prompt, max_new_tokens=200):
    """
    Génère une réponse du modèle via pipeline.
    
    Args:
        pipe: Pipeline de génération de texte.
        prompt (str): Le prompt pour le modèle.
        max_new_tokens (int): Nombre maximum de tokens à générer.
        
    Returns:
        str: La réponse générée.
    """
    response = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return response[0]['generated_text'][len(prompt):]


def generate_advanced_response(tokenizer, model, prompt, max_length=512):
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
    
    # Formatage pour le modèle, qui utilise un format spécifique pour les chats
    try:
        chat_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print(f"{Fore.RED}Erreur lors de l'application du template de chat: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Utilisation du prompt direct{Style.RESET_ALL}")
        chat_text = prompt
    
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
    try:
        response = full_response.split("[/INST]")[-1].strip()
    except:
        # Si le format n'est pas standard, retourner la réponse complète moins le prompt
        response = full_response[len(chat_text):].strip()
        if not response:
            response = full_response
    
    return response


def evaluate_prompt_pipeline(pipe, guard, prompt, history):
    """
    Évalue un prompt avec le guardrail en utilisant la pipeline.
    
    Args:
        pipe: Pipeline de génération de texte.
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
        model_response = generate_pipeline_response(pipe, prompt)
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


def evaluate_prompt_advanced(tokenizer, model, guard, prompt, history):
    """
    Évalue un prompt avec le guardrail en utilisant le modèle avancé.
    
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
        model_response = generate_advanced_response(tokenizer, model, prompt)
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


def analyze_history(history):
    """
    Analyse l'historique des tests.
    
    Args:
        history (list): Liste des résultats des tests.
        
    Returns:
        dict: Métriques d'analyse.
    """
    # Convertir l'historique en DataFrame
    df = pd.DataFrame(history)
    
    metrics = {
        'total_prompts': len(df),
        'activation_rate': df['is_activated'].mean() if 'is_activated' in df and df['is_activated'].notna().any() else None,
        'avg_generation_time': df['generation_time'].mean() if 'generation_time' in df and df['generation_time'].notna().any() else None,
        'avg_validation_time': df['validation_time'].mean() if 'validation_time' in df and df['validation_time'].notna().any() else None,
        'avg_total_time': df['total_time'].mean() if 'total_time' in df and df['total_time'].notna().any() else None,
    }
    
    # Raisons d'activation
    if 'activation_reason' in df:
        filtered_df = df.loc[df['is_activated'] == True]
        if not filtered_df.empty:
            reason_counts = filtered_df['activation_reason'].value_counts().to_dict()
            metrics['activation_reasons'] = reason_counts
    
    return metrics


def display_metrics(metrics):
    """
    Affiche les métriques d'analyse.
    
    Args:
        metrics (dict): Métriques à afficher.
    """
    print(f"\n{Fore.YELLOW}=== Résultats de l'évaluation ==={Style.RESET_ALL}")
    print(f"Nombre total de prompts: {Fore.CYAN}{metrics['total_prompts']}{Style.RESET_ALL}")
    
    if metrics['activation_rate'] is not None:
        print(f"Taux d'activation: {Fore.CYAN}{metrics['activation_rate']:.2%}{Style.RESET_ALL}")
    else:
        print(f"Taux d'activation: {Fore.RED}N/A{Style.RESET_ALL}")
        
    if metrics['avg_generation_time'] is not None:
        print(f"Temps moyen de génération: {Fore.CYAN}{metrics['avg_generation_time']:.4f}{Style.RESET_ALL} secondes")
    else:
        print(f"Temps moyen de génération: {Fore.RED}N/A{Style.RESET_ALL}")
        
    if metrics['avg_validation_time'] is not None:
        print(f"Temps moyen de validation: {Fore.CYAN}{metrics['avg_validation_time']:.4f}{Style.RESET_ALL} secondes")
    else:
        print(f"Temps moyen de validation: {Fore.RED}N/A{Style.RESET_ALL}")
        
    if metrics['avg_total_time'] is not None:
        print(f"Temps total moyen: {Fore.CYAN}{metrics['avg_total_time']:.4f}{Style.RESET_ALL} secondes")
    else:
        print(f"Temps total moyen: {Fore.RED}N/A{Style.RESET_ALL}")
    
    if 'activation_reasons' in metrics:
        print(f"\n{Fore.YELLOW}Raisons d'activation:{Style.RESET_ALL}")
        for reason, count in metrics['activation_reasons'].items():
            print(f"- {reason}: {Fore.CYAN}{count}{Style.RESET_ALL}")


def main():
    parser = argparse.ArgumentParser(description='Test interactif de guardrails anti-toxicité')
    parser.add_argument('--output', type=str, default='data/interactive_results.csv',
                       help='Fichier de sortie pour les résultats')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Seuil de confiance pour la toxicité')
    parser.add_argument('--method', type=str, default='sentence', choices=['sentence', 'full'],
                       help='Méthode de validation')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='Modèle à utiliser')
    parser.add_argument('--advanced', action='store_true',
                       help='Utiliser le mode avancé pour les grands modèles comme Mistral')
    parser.add_argument('--no-8bit', action='store_true',
                       help='Désactiver la quantification 8-bit (mode avancé uniquement)')
    args = parser.parse_args()
    
    # Configurer le guardrail
    guard = setup_guardrail(threshold=args.threshold, validation_method=args.method)
    
    # Configurer le modèle
    if args.advanced:
        print(f"{Fore.YELLOW}Utilisation du mode avancé pour le modèle {args.model}{Style.RESET_ALL}")
        tokenizer, model = setup_advanced_model(model_name=args.model, use_8bit=not args.no_8bit)
    else:
        print(f"{Fore.YELLOW}Utilisation du mode pipeline pour le modèle {args.model}{Style.RESET_ALL}")
        pipe = setup_pipeline_model(model_name=args.model)
    
    # Historique des tests
    history = []
    
    # Boucle interactive
    while True:
        print("\n" + "=" * 80)
        model_desc = args.model.split('/')[-1] if '/' in args.model else args.model
        print(f"{Fore.YELLOW}Mode interactif avec {model_desc} - Entrez un prompt pour tester le guardrail{Style.RESET_ALL}")
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
            # Évaluer le prompt avec le mode approprié
            if args.advanced:
                evaluate_prompt_advanced(tokenizer, model, guard, prompt, history)
            else:
                evaluate_prompt_pipeline(pipe, guard, prompt, history)


if __name__ == "__main__":
    main() 