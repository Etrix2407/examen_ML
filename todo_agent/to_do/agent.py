#!/usr/bin/env python
# coding: utf-8

# # Devoir 5 – Agent IA basique : Assistant de gestion de tâches (Todo List)
# 
# **Basé sur l'atelier de M. Suire — Henallux, Avril 2026**
# 
# ## Concept de l'agent
# 
# Cet agent implémente la **boucle agentique ReAct** (Reason + Act) :
# 
# | Étape | Description |
# |-------|-------------|
# | **Observation** | L'agent reçoit un message utilisateur |
# | **Réflexion** | Le LLM raisonne et décide quelle action mener |
# | **Action** | L'agent appelle un outil Python |
# | **Retour** | Le résultat est réinjecté dans le contexte |
# | **Réitération** | Le cycle continue jusqu'à la réponse finale |
# 
# ## Technologies utilisées (atelier M. Suire)
# - **Google ADK** : framework agentique (orchestrateur)
# - **LiteLLM** : couche d'abstraction pour les LLMs
# - **Ollama** : exécution locale d'un LLM (`qwen3:4b`)
# - **Python** : logique des outils (CRUD + analyse + suggestions)
# 
# ## Structure du notebook
# 1. Installation et imports
# 2. Persistance des données (JSON)
# 3. Outils CRUD (`add_todo`, `list_todos`, `complete_todo`, `delete_todo`, + bonus)
# 4. Outils d'analyse (`list_todos_by_priority`, `group_todos_by_tag`)
# 5. Définition de l'agent ADK avec system prompt
# 6. Démonstration des outils
# 7. Tests unitaires des outils
# 8. Discussion : tests comportementaux
# 9. Mini-rapport
# 
# ---
# > **Prérequis :** Ollama installé et lancé, modèle `qwen3:4b` disponible (`ollama pull qwen3:4b`)
# > **Installation :** `pip install google-adk litellm`
# 

# ## 1. Installation et imports

# In[ ]:


# Installation des dépendances (à exécuter une seule fois)
# pip install google-adk litellm

import json
import os


# ## 2. Persistance des données
# 
# Les tâches sont stockées dans un fichier `todos.json`.
# Chaque tâche suit la structure :
# 
# ```json
# {
#   "id": 1,
#   "title": "Preparer la reunion",
#   "tag": "travail",
#   "done": false
# }
# ```
# 
# Les tags valides : `"perso"`, `"travail"`, `"urgent"`
# 

# In[ ]:


# PERSISTANCE : lecture / ecriture du fichier JSON
# Fonctions utilitaires internes (prefixe _ = usage interne uniquement)

TODO_FILE = "todos.json"
VALID_TAGS = ["perso", "travail", "urgent"]


def _load_todos() -> list:
    """Charge la liste des taches depuis le fichier JSON."""
    if not os.path.exists(TODO_FILE):
        return []
    with open(TODO_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_todos(todos: list) -> None:
    """Sauvegarde la liste des taches dans le fichier JSON."""
    with open(TODO_FILE, "w", encoding="utf-8") as f:
        json.dump(todos, f, indent=2, ensure_ascii=False)


def _next_id(todos: list) -> int:
    """Retourne le prochain ID disponible."""
    if not todos:
        return 1
    return max(t["id"] for t in todos) + 1


print("Fonctions de persistance definies")
print(f"   Fichier de donnees : {TODO_FILE}")
print(f"   Tags valides : {VALID_TAGS}")


# ## 3. Outils CRUD
# 
# Ce sont les **outils que l'agent peut appeler**. Google ADK lit automatiquement :
# - Le **nom** de la fonction → identifiant de l'outil
# - Les **parametres** → ce que l'agent doit fournir
# - La **docstring** → description transmise au LLM pour qu'il sache quand utiliser l'outil
# 
# > La docstring est **critique** : c'est elle qui guide le LLM dans le choix de l'outil.
# 

# In[ ]:


def add_todo(title: str, tag: str = "perso") -> str:
    """Ajoute une nouvelle tache a la liste.

    Args:
        title: Le titre ou la description de la tache a ajouter.
        tag: La categorie de la tache. Valeurs possibles : perso, travail, urgent.
             Par defaut : perso.

    Returns:
        Un message de confirmation avec l'ID de la tache creee,
        ou une erreur si le tag est invalide.
    """
    if tag not in VALID_TAGS:
        return f"Erreur : tag '{tag}' invalide. Valeurs acceptees : {', '.join(VALID_TAGS)}."
    todos = _load_todos()
    new_todo = {"id": _next_id(todos), "title": title, "tag": tag, "done": False}
    todos.append(new_todo)
    _save_todos(todos)
    return f"Tache ajoutee : '{title}' [tag: {tag}] (id: {new_todo['id']})"


print("add_todo defini")


# In[ ]:


def list_todos(tag: str = "all") -> str:
    """Liste les taches existantes, avec filtre optionnel par tag.

    Args:
        tag: Tag de filtrage. Valeurs possibles : all (toutes), perso, travail,
             urgent, pending (non terminees), done (terminees).

    Returns:
        La liste des taches formatee en texte lisible, ou un message si aucune tache.
    """
    todos = _load_todos()
    if not todos:
        return "Aucune tache enregistree."
    if tag == "pending":
        todos = [t for t in todos if not t["done"]]
    elif tag == "done":
        todos = [t for t in todos if t["done"]]
    elif tag in VALID_TAGS:
        todos = [t for t in todos if t["tag"] == tag]
    if not todos:
        return f"Aucune tache pour le filtre '{tag}'."
    lines = []
    for t in todos:
        state = "[OK]" if t["done"] else "[ ]"
        lines.append(f"{state} [{t['id']}] {t['title']} (tag: {t['tag']})")
    return "\n".join(lines)


print("list_todos defini")


# In[ ]:


def complete_todo(todo_id: int) -> str:
    """Marque une tache comme terminee.

    Args:
        todo_id: L'identifiant numerique de la tache a terminer.

    Returns:
        Un message de confirmation ou d'erreur si la tache n'existe pas.
    """
    todos = _load_todos()
    for t in todos:
        if t["id"] == todo_id:
            t["done"] = True
            _save_todos(todos)
            return f"Tache '{t['title']}' (id: {todo_id}) marquee comme terminee."
    return f"Erreur : aucune tache trouvee avec l'id {todo_id}."


def delete_todo(todo_id: int) -> str:
    """Supprime definitivement une tache.

    Args:
        todo_id: L'identifiant numerique de la tache a supprimer.

    Returns:
        Un message de confirmation ou d'erreur si la tache n'existe pas.
    """
    todos = _load_todos()
    for i, t in enumerate(todos):
        if t["id"] == todo_id:
            removed = todos.pop(i)
            _save_todos(todos)
            return f"Tache '{removed['title']}' (id: {todo_id}) supprimee."
    return f"Erreur : aucune tache trouvee avec l'id {todo_id}."


print("complete_todo et delete_todo definis")


# ### Outils CRUD supplementaires (prolongations niveau 1 de l'atelier)
# 
# `rename_todo` et `clear_done` sont proposes dans les prolongations de l'atelier sans code fourni.
# 

# In[ ]:


def rename_todo(todo_id: int, new_title: str) -> str:
    """Modifie le titre d'une tache existante sans la supprimer.

    Args:
        todo_id: L'identifiant numerique de la tache a renommer.
        new_title: Le nouveau titre a attribuer a la tache.

    Returns:
        Un message de confirmation ou d'erreur si la tache n'existe pas.
    """
    todos = _load_todos()
    for t in todos:
        if t["id"] == todo_id:
            old_title = t["title"]
            t["title"] = new_title
            _save_todos(todos)
            return f"Tache {todo_id} renommee : '{old_title}' vers '{new_title}'"
    return f"Erreur : aucune tache trouvee avec l'id {todo_id}."


def clear_done() -> str:
    """Supprime toutes les taches marquees comme terminees en une seule operation.

    Returns:
        Un message indiquant le nombre de taches supprimees.
    """
    todos = _load_todos()
    avant = len(todos)
    todos = [t for t in todos if not t["done"]]
    _save_todos(todos)
    supprimes = avant - len(todos)
    if supprimes == 0:
        return "Aucune tache terminee a supprimer."
    return f"{supprimes} tache(s) terminee(s) supprimee(s)."


print("rename_todo et clear_done definis")


# ## 4. Outils d'analyse (logique Python deterministe)
# 
# Ces outils n'utilisent **aucun LLM** — ce sont des algorithmes deterministes.
# Meme entree = meme sortie, toujours.
# 
# > **Pourquoi les encapsuler comme outils ?**
# > Parce que l'agent decide *lui-meme* de les appeler selon le contexte de la conversation.
# 

# In[ ]:


PRIORITY_ORDER = {"urgent": 0, "travail": 1, "perso": 2}


def list_todos_by_priority() -> str:
    """Retourne toutes les taches non terminees, triees par priorite decroissante.

    L'ordre de priorite est : urgent > travail > perso.

    Returns:
        La liste triee des taches en attente, ou un message si aucune tache.
    """
    todos = _load_todos()
    pending = [t for t in todos if not t["done"]]
    if not pending:
        return "Aucune tache en attente."
    sorted_todos = sorted(pending, key=lambda t: PRIORITY_ORDER.get(t["tag"], 99))
    lines = [f"[{t['tag'].upper()}] [{t['id']}] {t['title']}" for t in sorted_todos]
    return "\n".join(lines)


def group_todos_by_tag() -> str:
    """Retourne toutes les taches non terminees, regroupees par tag.

    Returns:
        Les taches organisees par categorie (urgent, travail, perso).
    """
    todos = _load_todos()
    pending = [t for t in todos if not t["done"]]
    if not pending:
        return "Aucune tache en attente."
    groups = {tag: [] for tag in VALID_TAGS}
    for t in pending:
        tag = t.get("tag", "perso")
        if tag in groups:
            groups[tag].append(f"  [{t['id']}] {t['title']}")
    lines = []
    for tag in sorted(VALID_TAGS, key=lambda x: PRIORITY_ORDER[x]):
        if groups[tag]:
            lines.append(f"### {tag.capitalize()}")
            lines.extend(groups[tag])
    return "\n".join(lines) if lines else "Aucune tache en attente."


print("list_todos_by_priority et group_todos_by_tag definis")


# ## 5. Definition de l'agent ADK
# 
# L'agent est compose de :
# - Un **LLM** (Ollama local via LiteLLM)
# - Un **system prompt** : role, regles metier, instructions par outil, suggestions
# - La **liste des outils** disponibles
# 
# ### Note sur les suggestions automatiques (Partie 8 de l'atelier)
# 
# La regle de suggestion est **non deterministe** : elle repose sur la creativite du LLM,
# contrairement aux outils de tri/regroupement qui sont purement algorithmiques.
# C'est l'illustration directe du point de discussion de l'atelier :
# *quand utiliser le LLM vs un algorithme ?*
# 

# In[ ]:


from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# CONFIGURATION DU MODELE
# Option A : modele local Ollama
MODEL_ID = "ollama_chat/qwen3:4b"

# Option B : endpoint distant (decommenter et renseigner)
# MODEL_ID     = "openai/<model_name>"
# API_BASE_URL = "http://<adresse>:<port>/v1"
# API_KEY      = "votre_cle"

SYSTEM_PROMPT = """Tu es un assistant de gestion de taches (todo list).
Tu es courtois, professionnel et concis.
Chaque tache a un titre et un tag parmi : perso, travail, urgent.

## Regles d'utilisation des outils

- Utilise TOUJOURS l'outil approprie pour repondre aux demandes liees aux taches.
- Quand l'utilisateur demande a voir ses taches, utilise list_todos.
- Quand l'utilisateur veut ajouter une tache, utilise add_todo.
  Si le tag n'est pas precise, infere-le depuis le contexte. Si tu n'es pas sur, demande.
- Quand l'utilisateur a termine une tache, utilise complete_todo.
- Quand l'utilisateur veut supprimer une tache, utilise delete_todo.
  Avant de supprimer, demande TOUJOURS une confirmation explicite.
- Quand l'utilisateur veut renommer une tache, utilise rename_todo.
- Quand l'utilisateur veut supprimer toutes les taches terminees, utilise clear_done.
- Quand l'utilisateur demande une priorisation, utilise list_todos_by_priority.
- Quand l'utilisateur veut ses taches par theme, utilise group_todos_by_tag.
- Ne fabrique jamais de donnees : utilise toujours les outils.

## Suggestions automatiques

Apres chaque ajout de tache reussi :
1. Appelle list_todos pour voir le contexte actuel.
2. Propose 1 ou 2 taches complementaires pertinentes en lien avec la tache ajoutee.
   Formule comme une question : "Souhaitez-vous aussi ajouter [suggestion] ?"
3. Ne propose pas de suggestions pour les suppressions ou complet ions.

## Hors perimetre

Si l'utilisateur demande quelque chose hors perimetre (meteo, sport, poemes, maths...),
indique poliment que tu ne peux pas l'aider, et rappelle ce que tu peux faire.
"""

root_agent = Agent(
    name="todo_agent",
    model=LiteLlm(model=MODEL_ID),
    description="Agent de gestion de taches avec tags, priorisation et suggestions",
    instruction=SYSTEM_PROMPT,
    tools=[
        add_todo,
        list_todos,
        complete_todo,
        delete_todo,
        rename_todo,
        clear_done,
        list_todos_by_priority,
        group_todos_by_tag,
    ],
)
if "name" == "main":

    print(f"Agent 'todo_agent' cree avec {len(root_agent.tools)} outils")
    for tool in root_agent.tools:
        print(f"  - {tool.__name__}")


# ## 6. Lancement et demonstration
# 
# ### Lancement de l'interface web ADK
# 
# ```bash
# # Depuis le dossier racine du projet
# uv run adk web
# # puis ouvrir : http://localhost:8000
# ```
# 
# L'interface web affiche les logs des appels d'outils — la boucle ReAct est visible en temps reel.
# 
# ### Scenarios de test suggeres (tires de l'atelier)
# 
# | Requete | Outil(s) attendu(s) |
# |---------|---------------------|
# | "Ajoute une tache : reviser ADK, tag travail" | `add_todo` + `list_todos` (suggestion) |
# | "Qu'est-ce que j'ai a faire ?" | `list_todos` |
# | "Par quoi je dois commencer ?" | `list_todos_by_priority` |
# | "Montre mes taches par categorie" | `group_todos_by_tag` |
# | "J'ai fini la tache 1" | `complete_todo` |
# | "Nettoie les taches terminees" | `clear_done` |
# | "Quel temps fait-il ?" | *Aucun outil — refus poli* |
# 
# ### Demo des outils (sans LLM)
# 

# In[ ]:


# Demonstration des outils independamment du LLM
    if os.path.exists(TODO_FILE):
        os.remove(TODO_FILE)

    print("=== Demonstration des outils ===")
    print(add_todo("Reviser le cours ADK", "travail"))
    print(add_todo("Faire les courses", "perso"))
    print(add_todo("Appeler le medecin", "urgent"))
    print(add_todo("Preparer la reunion", "travail"))
    print(add_todo("Envoyer le rapport", "urgent"))

    print("\n--- Toutes les taches ---")
    print(list_todos())

    print("\n--- Par priorite ---")
    print(list_todos_by_priority())

    print("\n--- Par categorie ---")
    print(group_todos_by_tag())

    print("\n--- Completion tache 3 ---")
    print(complete_todo(3))

    print("\n--- Renommage tache 1 ---")
    print(rename_todo(1, "Reviser ADK + faire les exercices"))

    print("\n--- Nettoyage des terminees ---")
    print(clear_done())

    print("\n--- Taches restantes ---")
    print(list_todos())

    if os.path.exists(TODO_FILE):
        os.remove(TODO_FILE)


# ## 7. Tests unitaires des outils
# 
# Les outils sont des fonctions Python **deterministes** : meme entree = meme sortie.
# On peut et on doit les tester independamment du LLM.
# 
# | Critere | Tests outils | Tests comportementaux (section 8) |
# |---------|-------------|-----------------------------------|
# | Vitesse | < 1 seconde | 5-15s par test |
# | Determinisme | 100% | Non-deterministe |
# | Dependance LLM | Non | Oui |
# 

# In[ ]:


    def reset():
        if os.path.exists(TODO_FILE):
            os.remove(TODO_FILE)


    def check(nom, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {nom}" + (f" -- {detail}" if detail and not condition else ""))
        return condition


    print("=" * 55)
    print("TESTS UNITAIRES")
    print("=" * 55)
    resultats = []

# add_todo
    reset()
    print("\n[add_todo]")
    r = add_todo("Acheter du lait", "perso")
    resultats.append(check("Ajout valide retourne id", "id: 1" in r))
    resultats.append(check("Titre present dans le retour", "Acheter du lait" in r))
    with open(TODO_FILE) as f:
        todos = json.load(f)
    resultats.append(check("Un seul enregistrement", len(todos) == 1))
    resultats.append(check("done=False par defaut", todos[0]["done"] is False))
    r2 = add_todo("T", "mauvais_tag")
    resultats.append(check("Tag invalide => Erreur", "Erreur" in r2))

# list_todos
    reset()
    print("\n[list_todos]")
    resultats.append(check("Liste vide => Aucune", "Aucune" in list_todos()))
    add_todo("Perso", "perso")
    add_todo("Travail", "travail")
    resultats.append(check("Filtre perso exclut travail", "Travail" not in list_todos("perso")))
    resultats.append(check("Filtre all inclut tout", "Perso" in list_todos("all") and "Travail" in list_todos("all")))

# complete_todo
    reset()
    print("\n[complete_todo]")
    add_todo("A terminer", "travail")
    r  = complete_todo(1)
    resultats.append(check("Confirmation de completion", "terminee" in r))
    resultats.append(check("Apparait dans done", "A terminer" in list_todos("done")))
    resultats.append(check("ID inexistant => Erreur", "Erreur" in complete_todo(999)))

# delete_todo
    reset()
    print("\n[delete_todo]")
    add_todo("A supprimer", "perso")
    r = delete_todo(1)
    resultats.append(check("Confirmation de suppression", "supprimee" in r))
    resultats.append(check("Liste vide apres suppression", "Aucune" in list_todos()))
    resultats.append(check("ID inexistant => Erreur", "Erreur" in delete_todo(999)))

# rename_todo
    reset()
    print("\n[rename_todo]")
    add_todo("Ancien titre", "perso")
    r = rename_todo(1, "Nouveau titre")
    resultats.append(check("Confirmation renommage", "Nouveau titre" in r))
    resultats.append(check("Nouveau titre dans la liste", "Nouveau titre" in list_todos()))
    resultats.append(check("ID inexistant => Erreur", "Erreur" in rename_todo(999, "x")))

# clear_done
    reset()
    print("\n[clear_done]")
    add_todo("T1", "perso"); add_todo("T2", "travail")
    complete_todo(1)
    r = clear_done()
    resultats.append(check("1 tache supprimee", "1 tache" in r))
    resultats.append(check("Tache active preservee", "T2" in list_todos()))
    resultats.append(check("Deuxieme appel => Aucune", "Aucune" in clear_done()))

# list_todos_by_priority
    reset()
    print("\n[list_todos_by_priority]")
    add_todo("P", "perso"); add_todo("U", "urgent"); add_todo("T", "travail")
    r = list_todos_by_priority()
    lines = r.splitlines()
    resultats.append(check("Premier = URGENT", "URGENT" in lines[0]))
    resultats.append(check("Dernier = PERSO", "PERSO" in lines[-1]))
    complete_todo(2)
    resultats.append(check("Terminees exclues", "U" not in list_todos_by_priority()))

# group_todos_by_tag
    reset()
    print("\n[group_todos_by_tag]")
    add_todo("P", "perso"); add_todo("T", "travail"); add_todo("U", "urgent")
    r = group_todos_by_tag()
    resultats.append(check(
        "Urgent avant Travail avant Perso",
        r.index("Urgent") < r.index("Travail") < r.index("Perso")
    ))

    print()
    print("=" * 55)
    nb_pass = sum(resultats)
    print(f"RESULTAT : {nb_pass}/{len(resultats)} tests reussis")
    print("=" * 55)
    reset()


# ## 8. Tests comportementaux (Partie 10 de l'atelier)
# 
# Les tests unitaires (section 7) testent les **outils** — Python pur, deterministe.
# Ici on teste le **comportement de l'agent** face a des requetes en langage naturel.
# 
# ### Strategie (tiree de l'atelier)
# 
# - **Check fiable** : verifier qu'aucun outil n'est appele pour des requetes hors-perimetre
# - **Check secondaire** : presence de mots-cles de refus dans la reponse
# 
# Ces tests sont dans un fichier separe car ils dependent d'Ollama en cours d'execution.
# 
# **Pour executer :**
# ```bash
# uv add --dev pytest pytest-asyncio
# uv run pytest tests/test_agent_behavior.py -v
# ```
# 

# In[ ]:


# Code des tests comportementaux
# A placer dans tests/test_agent_behavior.py

    code = '''
    # tests/test_agent_behavior.py
    # Necessite : Ollama lance + qwen3:4b disponible + pytest-asyncio

    import pytest
    from todo_agents.agent import root_agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types


    async def ask_agent(message):
        session_service = InMemorySessionService()
        runner = Runner(agent=root_agent, app_name="test", session_service=session_service)
        session = await session_service.create_session(app_name="test", user_id="tester")
        content = types.Content(role="user", parts=[types.Part(text=message)])

        response_text = ""
        tool_was_called = False

        async for event in runner.run_async(user_id="tester", session_id=session.id, new_message=content):
            if event.content:
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        tool_was_called = True
                    if hasattr(part, "text") and part.text:
                        response_text += part.text

        print(f"Reponse : {response_text[:200]}")
        return response_text, tool_was_called


    # Tests de REFUS (aucun outil ne doit etre appele)
    async def test_refuses_weather():
        response, called = await ask_agent("Quel temps fait-il a Namur ?")
        assert not called

    async def test_refuses_sport():
        response, called = await ask_agent("Qui a gagne le Tour de France ?")
        assert not called

    async def test_refuses_poem():
        response, called = await ask_agent("Ecris-moi un poeme sur la pluie.")
        assert not called

    async def test_refuses_math():
        response, called = await ask_agent("Combien font 357 multiplie par 48 ?")
        assert not called


    # Tests POSITIFS (l'agent DOIT appeler un outil)
    async def test_add_uses_tool():
        response, called = await ask_agent("Ajoute une tache : envoyer le rapport, tag travail.")
        assert called

    async def test_list_uses_tool():
        response, called = await ask_agent("Montre-moi mes taches.")
        assert called

    async def test_priority_uses_tool():
        response, called = await ask_agent("Par quoi dois-je commencer ?")
        assert called

    async def test_group_uses_tool():
        response, called = await ask_agent("Montre mes taches par categorie.")
        assert called
    '''

    print(code)
    print()
    print("-> Placer dans tests/test_agent_behavior.py")
    print("-> Lancer : uv run pytest tests/test_agent_behavior.py -v")


# ---
# 
# ## 9. Mini-rapport
# 
# ### A. Facon dont l'IA a ete utilisee
# 
# Pour realiser ce devoir, j'ai utilise **Claude (Anthropic)** a plusieurs etapes :
# 
# **1. Lecture et synthese de l'atelier**
# J'ai fourni le lien GitHub du depot `csuire01/henallux_agent_workshop` a Claude,
# qui a lu le fichier `student_instructions.md` et extrait les elements essentiels :
# outils a implementer, structure du projet, patterns de test, points de discussion.
# 
# **2. Adaptation au format notebook**
# L'atelier est concu pour un projet Python multi-fichiers (`agent.py`, `tools.py`, `tests/`).
# J'ai demande a Claude d'adapter cette structure en notebook Jupyter autonome,
# avec les outils definis directement dans les cellules.
# 
# **3. Implementation des outils de prolongation**
# Les outils `rename_todo` et `clear_done` sont proposes dans les prolongations de l'atelier
# sans code fourni. Claude les a implementes en respectant les conventions etablies
# (docstrings precisent les parametres et valeurs de retour, gestion explicite des erreurs).
# 
# **4. Redaction du system prompt**
# Le system prompt de l'atelier contenait un placeholder `# Reflechissez...` pour
# la section suggestions automatiques. Claude a redige cette regle de maniere precise,
# en specifiant le comportement attendu (appel `list_todos` + formulation interrogative).
# 
# ---
# 
# ### B. Analyse reflexive
# 
# **Points positifs :**
# 
# - **Gain de temps sur la structure** : adapter un projet multi-fichiers en notebook
#   aurait necessaire beaucoup de manipulation manuelle.
# - **Coherence stylistique** : les conventions sont respectees uniformement (nommage,
#   docstrings, style) car un seul "redacteur" a produit le code.
# - **Aide a la comprehension** : Claude a explique la distinction entre
#   `ollama_chat/` et `openai/` dans LiteLLM, et clarifie pourquoi la normalisation L2
#   dans FAISS revient a maximiser la similarite cosinus.
# 
# **Limites et risques identifies :**
# 
# - **L'IA ne peut pas tester** : Claude ne peut pas lancer Ollama, verifier que les outils
#   fonctionnent vraiment, ni observer le comportement du LLM en temps reel.
#   La validation reste entierement a la charge de l'etudiant.
# - **Risque de sur-confiance** : un code genere par IA semble correct mais peut contenir
#   des erreurs subtiles (version d'API, parametre manquant). Chaque cellule doit etre
#   lue et comprise avant d'etre executee.
# - **Le prompt engineering reste artisanal** : Claude peut proposer un system prompt,
#   mais l'ajuster en fonction du comportement reel du modele Ollama necessite des tests
#   manuels que l'IA ne peut pas remplacer.
# 
# **Conclusion :**
# L'IA est un excellent outil pour la mise en place de la structure (boilerplate, conventions,
# scaffold), mais ne remplace pas l'experimentation directe. Dans le cadre d'un agent IA,
# le vrai apprentissage se fait en observant la boucle ReAct en action dans les logs ADK,
# en testant des requetes ambigues, et en iterant sur le system prompt.
# 
# ---
# 
# ### C. Organisation des tests
# 
# Deux niveaux de tests distincts, conformement a l'approche de l'atelier (Parties 9 et 10) :
# 
# **Niveau 1 — Tests unitaires des outils (Section 7 du notebook)**
# 
# | Test | Outil | Verification |
# |------|-------|-------------|
# | Ajout valide | `add_todo` | Retour + fichier JSON |
# | Tag invalide | `add_todo` | Message d'erreur |
# | Liste vide | `list_todos` | Message "Aucune" |
# | Filtre par tag | `list_todos` | Isolation des resultats |
# | Completion | `complete_todo` | done=True dans JSON |
# | ID inexistant | `complete_todo`, `delete_todo`, `rename_todo` | Message "Erreur" |
# | Renommage | `rename_todo` | Nouveau titre dans la liste |
# | Nettoyage groupe | `clear_done` | Comptage + taches actives preservees |
# | Priorisation | `list_todos_by_priority` | Ordre URGENT > TRAVAIL > PERSO |
# | Regroupement | `group_todos_by_tag` | Ordre des sections dans la sortie |
# 
# **Niveau 2 — Tests comportementaux (Section 8, fichier separe)**
# 
# - 4 **tests de refus** : requetes hors-perimetre → aucun outil ne doit etre appele
# - 4 **tests positifs** : requetes legitimes → au moins un outil doit etre appele
# 
# **Choix de conception :** les tests comportementaux sont dans un fichier separe
# (`test_agent_behavior.py`) car ils dependent d'Ollama en cours d'execution
# et ne peuvent pas etre integres a une CI/CD sans infrastructure dediee.
# Ce sont des tests **non-deterministes** par nature — ils peuvent echouer occasionnellement
# meme pour un agent bien concu, ce qui est precisement leur interet pedagogique.
# 
