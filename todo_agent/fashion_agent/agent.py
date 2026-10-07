#!/usr/bin/env python
# coding: utf-8

"""
Agent IA - Bot Conseil Mode : Assistant de gestion de garde-robe et conseil vestimentaire
Basé sur la structure de l'atelier ReAct (Google ADK + LiteLLM + Ollama qwen3:4b).

Fonctionnalités :
  1. CRUD des vêtements (ajout, liste, modification, suppression)
  2. Organisation par saison et par occasion
  3. Suggestions de tenues (association type + couleur + saison/occasion)

Prérequis : Ollama installé et lancé, modèle qwen3:4b disponible (`ollama pull qwen3:4b`, "ollama run qwen3:4b")
Installation : pip install google-adk litellm (adk web)
"""

import json
import os
from itertools import product

# ============================================================
# 1. Persistance des données
# ============================================================
# Chaque vetement suit la structure :
# {
#   "id": 1,
#   "name": "Chemise bleue",
#   "type": "haut",
#   "color": "bleu",
#   "season": "mi_saison",
#   "occasion": "travail"
# }

CLOSET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "closet.json")
VALID_TYPES = ["haut", "bas", "robe", "veste", "chaussures", "accessoire"]
VALID_SEASONS = ["ete", "hiver", "mi_saison", "toutes"]
VALID_OCCASIONS = ["casual", "travail", "sport", "soiree", "toutes"]
NEUTRAL_COLORS = {"noir", "blanc", "gris", "beige", "marine"}

# Position de chaque couleur sur le cercle chromatique (0-360 degres).
# Sert a determiner les relations d'harmonie : couleurs analogues (proches
# sur le cercle) et couleurs complementaires (opposees sur le cercle).
# Les couleurs neutres ne figurent pas ici : elles sont gerees a part
# (elles se marient avec tout).
COLOR_WHEEL = {
    "rouge": 0, "bordeaux": 355, "brique": 15,
    "corail": 20, "terracotta": 25,
    "orange": 60, "abricot": 50, "camel": 55, "marron": 50,
    "jaune": 120, "moutarde": 95, "ocre": 90, "or": 100,
    "kaki": 150, "olive": 155,
    "vert": 180, "menthe": 190, "emeraude": 185,
    "turquoise": 210, "cyan": 215,
    "bleu": 240, "ciel": 235, "denim": 245,
    "indigo": 270,
    "violet": 300, "lavande": 295, "mauve": 305,
    "magenta": 320,
    "rose": 335, "rose fluo": 340, "fuchsia": 325, "corail rose": 330,
}

# Ecart maximal (en degres) pour considerer deux couleurs comme "analogues"
# (voisines sur le cercle chromatique -- jusqu'a 2 teintes d'ecart --,
# associations douces et harmonieuses).
ANALOGOUS_MAX_DIFF = 65
# Plage d'ecart (en degres) autour de 180 pour considerer deux couleurs
# comme "complementaires" (opposees sur le cercle, associations contrastees
# mais harmonieuses).
COMPLEMENTARY_RANGE = (150, 210)


def _hue_diff(deg1: float, deg2: float) -> float:
    """Ecart angulaire minimal entre deux teintes sur le cercle chromatique (0-180)."""
    diff = abs(deg1 - deg2) % 360
    return min(diff, 360 - diff)


def _load_items() -> list:
    """Charge la garde-robe depuis le fichier JSON."""
    if not os.path.exists(CLOSET_FILE):
        return []
    with open(CLOSET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_items(items: list) -> None:
    """Sauvegarde la garde-robe dans le fichier JSON."""
    with open(CLOSET_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def _next_id(items: list) -> int:
    """Retourne le prochain ID disponible."""
    if not items:
        return 1
    return max(i["id"] for i in items) + 1


def _color_relation(c1: str, c2: str) -> str | None:
    """Determine la relation d'harmonie entre deux couleurs, ou None si aucune.

    Relations reconnues (par ordre de priorite) :
      - "identiques"     : memes couleurs
      - "neutre"         : l'une des deux (ou les deux) est neutre
      - "complementaires": opposees sur le cercle chromatique (contraste harmonieux)
      - "analogues"      : voisines sur le cercle chromatique (accord tout en douceur)
    Si l'une des couleurs est inconnue du cercle chromatique, seules les
    regles "identiques" et "neutre" s'appliquent (comportement de repli).
    """
    c1, c2 = c1.lower().strip(), c2.lower().strip()
    if c1 == c2:
        return "identiques"
    if c1 in NEUTRAL_COLORS or c2 in NEUTRAL_COLORS:
        return "neutre"
    if c1 not in COLOR_WHEEL or c2 not in COLOR_WHEEL:
        return None
    diff = _hue_diff(COLOR_WHEEL[c1], COLOR_WHEEL[c2])
    if COMPLEMENTARY_RANGE[0] <= diff <= COMPLEMENTARY_RANGE[1]:
        return "complementaires"
    if diff <= ANALOGOUS_MAX_DIFF:
        return "analogues"
    return None


def _colors_compatible(c1: str, c2: str) -> bool:
    """Deux couleurs sont compatibles si elles sont identiques, si l'une est
    neutre, ou si elles forment un accord analogue ou complementaire sur le
    cercle chromatique."""
    return _color_relation(c1, c2) is not None


print("Fonctions de persistance definies")
print(f"   Fichier de donnees : {CLOSET_FILE}")
print(f"   Types valides : {VALID_TYPES}")
print(f"   Saisons valides : {VALID_SEASONS}")
print(f"   Occasions valides : {VALID_OCCASIONS}")


# ============================================================
# 2. Outils CRUD
# ============================================================

def add_clothing(name: str, type: str, color: str,
                  season: str = "toutes", occasion: str = "toutes") -> str:
    """Ajoute un vetement a la garde-robe.

    Args:
        name: Le nom/description du vetement (ex: "Chemise bleue").
        type: Le type de vetement. Valeurs possibles : haut, bas, robe,
            veste, chaussures, accessoire.
        color: La couleur principale du vetement (ex: "bleu", "noir").
        season: La saison associee. Valeurs possibles : ete, hiver,
            mi_saison, toutes. Par defaut : toutes.
        occasion: L'occasion associee. Valeurs possibles : casual, travail,
            sport, soiree, toutes. Par defaut : toutes.

    Returns:
        Un message de confirmation avec l'ID du vetement, ou une erreur si
        le type, la saison ou l'occasion sont invalides.
    """
    if type not in VALID_TYPES:
        return f"Erreur : type '{type}' invalide. Valeurs acceptees : {', '.join(VALID_TYPES)}."
    if season not in VALID_SEASONS:
        return f"Erreur : saison '{season}' invalide. Valeurs acceptees : {', '.join(VALID_SEASONS)}."
    if occasion not in VALID_OCCASIONS:
        return f"Erreur : occasion '{occasion}' invalide. Valeurs acceptees : {', '.join(VALID_OCCASIONS)}."

    items = _load_items()
    new_item = {
        "id": _next_id(items), "name": name, "type": type, "color": color,
        "season": season, "occasion": occasion,
    }
    items.append(new_item)
    _save_items(items)
    return f"Vetement ajoute : '{name}' [{type}, {color}, saison: {season}, occasion: {occasion}] (id: {new_item['id']})"


print("add_clothing defini")


def list_clothing(type: str = "all") -> str:
    """Liste les vetements de la garde-robe, avec filtre optionnel par type.

    Args:
        type: Filtre. Valeurs possibles : all (tous), ou l'un des types
            valides (haut, bas, robe, veste, chaussures, accessoire).

    Returns:
        La liste des vetements formatee en texte lisible, ou un message si
        la garde-robe est vide.
    """
    items = _load_items()
    if not items:
        return "La garde-robe est vide."
    if type != "all":
        if type not in VALID_TYPES:
            return f"Erreur : type '{type}' invalide."
        items = [i for i in items if i["type"] == type]
    if not items:
        return f"Aucun vetement pour le type '{type}'."
    lines = [f"[{i['id']}] {i['name']} - {i['type']}, {i['color']} "
             f"(saison: {i['season']}, occasion: {i['occasion']})" for i in items]
    return "\n".join(lines)


print("list_clothing defini")


def update_clothing(item_id: int, color: str = None, season: str = None, occasion: str = None) -> str:
    """Met a jour un ou plusieurs attributs d'un vetement existant. Seuls les
    champs fournis (non None) sont modifies.

    Args:
        item_id: L'identifiant numerique du vetement.
        color: Nouvelle couleur (optionnel).
        season: Nouvelle saison (optionnel). Valeurs possibles : ete, hiver, mi_saison, toutes.
        occasion: Nouvelle occasion (optionnel). Valeurs possibles : casual, travail, sport, soiree, toutes.

    Returns:
        Un message de confirmation ou d'erreur si le vetement n'existe pas
        ou si une valeur fournie est invalide.
    """
    if season is not None and season not in VALID_SEASONS:
        return f"Erreur : saison '{season}' invalide."
    if occasion is not None and occasion not in VALID_OCCASIONS:
        return f"Erreur : occasion '{occasion}' invalide."

    items = _load_items()
    for it in items:
        if it["id"] == item_id:
            if color is not None:
                it["color"] = color
            if season is not None:
                it["season"] = season
            if occasion is not None:
                it["occasion"] = occasion
            _save_items(items)
            return f"Vetement '{it['name']}' (id: {item_id}) mis a jour : {it['color']}, saison: {it['season']}, occasion: {it['occasion']}."
    return f"Erreur : aucun vetement trouve avec l'id {item_id}."


def remove_clothing(item_id: int) -> str:
    """Supprime definitivement un vetement de la garde-robe.

    Args:
        item_id: L'identifiant numerique du vetement a supprimer.

    Returns:
        Un message de confirmation ou d'erreur si le vetement n'existe pas.
    """
    items = _load_items()
    for i, it in enumerate(items):
        if it["id"] == item_id:
            removed = items.pop(i)
            _save_items(items)
            return f"'{removed['name']}' (id: {item_id}) supprime de la garde-robe."
    return f"Erreur : aucun vetement trouve avec l'id {item_id}."


print("update_clothing et remove_clothing definis")


# ============================================================
# 3. Outils d'organisation et de conseil (logique deterministe)
# ============================================================

def list_by_season(season: str) -> str:
    """Liste les vetements adaptes a une saison donnee (inclut ceux marques 'toutes').

    Args:
        season: La saison recherchee. Valeurs possibles : ete, hiver, mi_saison, toutes.

    Returns:
        La liste des vetements correspondants, ou un message si aucun ne correspond.
    """
    if season not in VALID_SEASONS:
        return f"Erreur : saison '{season}' invalide."
    items = [i for i in _load_items() if i["season"] == season or i["season"] == "toutes"]
    if not items:
        return f"Aucun vetement pour la saison '{season}'."
    lines = [f"[{i['id']}] {i['name']} - {i['type']}, {i['color']} (occasion: {i['occasion']})" for i in items]
    return "\n".join(lines)


def list_by_occasion(occasion: str) -> str:
    """Liste les vetements adaptes a une occasion donnee (inclut ceux marques 'toutes').

    Args:
        occasion: L'occasion recherchee. Valeurs possibles : casual, travail, sport, soiree, toutes.

    Returns:
        La liste des vetements correspondants, ou un message si aucun ne correspond.
    """
    if occasion not in VALID_OCCASIONS:
        return f"Erreur : occasion '{occasion}' invalide."
    items = [i for i in _load_items() if i["occasion"] == occasion or i["occasion"] == "toutes"]
    if not items:
        return f"Aucun vetement pour l'occasion '{occasion}'."
    lines = [f"[{i['id']}] {i['name']} - {i['type']}, {i['color']} (saison: {i['season']})" for i in items]
    return "\n".join(lines)


TYPE_ORDER = {t: i for i, t in enumerate(VALID_TYPES)}


def group_by_type() -> str:
    """Retourne tous les vetements de la garde-robe, regroupes par type.

    Returns:
        Les vetements organises par type (haut, bas, robe, veste, chaussures, accessoire).
    """
    items = _load_items()
    if not items:
        return "La garde-robe est vide."
    groups = {t: [] for t in VALID_TYPES}
    for it in items:
        groups.setdefault(it["type"], []).append(f"  [{it['id']}] {it['name']} - {it['color']}")
    lines = []
    for t in sorted(VALID_TYPES, key=lambda x: TYPE_ORDER[x]):
        if groups[t]:
            lines.append(f"### {t}")
            lines.extend(groups[t])
    return "\n".join(lines) if lines else "La garde-robe est vide."


def suggest_outfit(season: str = "toutes", occasion: str = "toutes") -> str:
    """Suggere une tenue complete (haut + bas ou robe + chaussures, veste en
    option) adaptee a la saison et l'occasion demandees, en verifiant la
    compatibilite des couleurs entre les pieces.

    Args:
        season: La saison visee. Valeurs possibles : ete, hiver, mi_saison, toutes.
        occasion: L'occasion visee. Valeurs possibles : casual, travail, sport, soiree, toutes.

    Returns:
        Une tenue complete si une combinaison compatible existe, sinon un
        message expliquant ce qui manque dans la garde-robe.
    """
    if season not in VALID_SEASONS:
        return f"Erreur : saison '{season}' invalide."
    if occasion not in VALID_OCCASIONS:
        return f"Erreur : occasion '{occasion}' invalide."

    items = [i for i in _load_items()
             if (i["season"] == season or i["season"] == "toutes" or season == "toutes")
             and (i["occasion"] == occasion or i["occasion"] == "toutes" or occasion == "toutes")]

    hauts = [i for i in items if i["type"] == "haut"]
    robes = [i for i in items if i["type"] == "robe"]
    bas = [i for i in items if i["type"] == "bas"]
    chaussures = [i for i in items if i["type"] == "chaussures"]
    vestes = [i for i in items if i["type"] == "veste"]
    accessoires = [i for i in items if i["type"] == "accessoire"]

    if not chaussures:
        return f"Impossible de proposer une tenue (saison: {season}, occasion: {occasion}) : aucune chaussure disponible."
    if not robes and (not hauts or not bas):
        return f"Impossible de proposer une tenue (saison: {season}, occasion: {occasion}) : il manque un haut+bas ou une robe."

    # Bases possibles : soit haut+bas, soit robe seule
    bases = []
    for h, b in product(hauts, bas):
        if _colors_compatible(h["color"], b["color"]):
            bases.append((h, b, None))
    for r in robes:
        bases.append((None, None, r))

    for h, b, r in bases:
        for c in chaussures:
            base_pieces = [h, b] if r is None else [r]
            if all(_colors_compatible(p["color"], c["color"]) for p in base_pieces if p):
                # Base + chaussures compatibles trouvee, on tente d'ajouter une veste
                outfit = [p for p in base_pieces if p] + [c]
                for v in vestes:
                    if all(_colors_compatible(p["color"], v["color"]) for p in outfit):
                        outfit.append(v)
                        break
                if accessoires:
                    for a in accessoires:
                        if all(_colors_compatible(p["color"], a["color"]) for p in outfit):
                            outfit.append(a)
                            break
                lines = [f"Tenue proposee (saison: {season}, occasion: {occasion}) :"]
                lines += [f"  - {p['type']} : {p['name']} ({p['color']})" for p in outfit]
                relations = {
                    _color_relation(p1["color"], p2["color"])
                    for idx, p1 in enumerate(outfit) for p2 in outfit[idx + 1:]
                }
                relations.discard(None)
                relations.discard("identiques")
                if relations:
                    lines.append(f"  Harmonie des couleurs : {', '.join(sorted(relations))}.")
                return "\n".join(lines)

    return (f"Aucune combinaison de couleurs compatible trouvee pour la saison '{season}' "
            f"et l'occasion '{occasion}'. Essaie d'ajouter des pieces neutres (noir, blanc, gris, beige, marine).")


print("list_by_season, list_by_occasion, group_by_type et suggest_outfit definis")


# ============================================================
# 4. Definition de l'agent ADK
# ============================================================

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

MODEL_ID = "ollama_chat/qwen3:4b-instruct"

SYSTEM_PROMPT = """
## Rôle

Tu es un assistant de conseil mode et de gestion de garde-robe.
Tu es courtois, professionnel et concis.
Chaque vetement a un nom, un type, une couleur, une saison et une occasion.

## Premier message 

Lors de ta première réponse dans une nouvelle conversation, commence TOUJOURS par ce message, puis réponds à la demande de l'utilisateur :

"Bonjour, je suis votre assistant mode. Donnez-moi un vêtement, sa couleur et sa saison (ex : un t-shirt noir, été). Pour un style encore plus pointu, n'oubliez pas de me préciser sa nuance exacte (ex : une robe bordeaux, printemps) ! :D"


## Regles d'utilisation des outils

- Utilise TOUJOURS l'outil approprie pour repondre aux demandes liees a la garde-robe.
- Quand l'utilisateur ajoute un vetement, utilise add_clothing.
  Si le type n'est pas precise, infere-le depuis le nom. Si tu n'es pas sur, demande.
- Quand l'utilisateur veut corriger la couleur, la saison ou l'occasion d'un vetement, utilise update_clothing.
- Quand l'utilisateur veut retirer un vetement (donne, jete, perdu), utilise remove_clothing.
  Avant de supprimer, demande TOUJOURS une confirmation explicite.
- Quand l'utilisateur veut voir sa garde-robe, utilise list_clothing.
- Quand l'utilisateur veut ses vetements par type, utilise group_by_type.
- Quand l'utilisateur demande ce qui est adapte a une saison, utilise list_by_season.
- Quand l'utilisateur demande ce qui est adapte a une occasion, utilise list_by_occasion.
- Quand l'utilisateur demande une suggestion de tenue ("que porter", "aide-moi a m'habiller"),
  utilise suggest_outfit en deduisant la saison/occasion depuis le contexte (sinon "toutes").
- Ne fabrique jamais de donnees : utilise toujours les outils.

## Regles general

- Reponds UNIQUEMENT en francais. N'utilise jamais de mots chinois, anglais ou d'autres langues.
- Reprends fidelement le resultat des outils. N'ajoute aucune description, aucun commentaire ni adjectif qui ne figure pas dans le resultat.

## Suggestions automatiques

Si list_by_season ou list_by_occasion renvoie plusieurs vetements compatibles, propose
spontanement d'appeler suggest_outfit pour composer une tenue complete. Formule comme une question.

## Hors perimetre

Si l'utilisateur demande quelque chose hors perimetre (meteo, sport, recettes, maths...),
indique poliment que tu ne peux pas l'aider, et rappelle ce que tu peux faire.
"""

import re


def _strip_thinking_callback(callback_context, llm_response):
    """Retire le raisonnement du contenu affiche a l'utilisateur :
    - supprime les Part marquees thought=True (cas Ollama/LiteLLM : le
      raisonnement arrive comme une Part separee du texte final)
    - nettoie aussi d'eventuelles balises <think>...</think> residuelles
      dans le texte, au cas ou elles seraient melangees au contenu.
    Le modele continue de raisonner en interne ; seul l'affichage est filtre."""
    if not llm_response.content or not llm_response.content.parts:
        return None

    changed = False
    new_parts = []
    for part in llm_response.content.parts:
        if getattr(part, "thought", False):
            changed = True
            continue
        if part.text:
            cleaned = re.sub(r"<think>.*?</think>", "", part.text, flags=re.DOTALL).strip()
            if cleaned != part.text:
                part.text = cleaned
                changed = True
            if not cleaned:
                changed = True
                continue
        new_parts.append(part)

    if not changed:
        return None
    llm_response.content.parts = new_parts
    return llm_response


root_agent = Agent(
    name="fashion_agent",
    model=LiteLlm(model=MODEL_ID, extra_body={"think": False}, temperature = 0.2),
    description="Agent de gestion de garde-robe et conseil vestimentaire",
    instruction=SYSTEM_PROMPT,
    after_model_callback=_strip_thinking_callback,
    tools=[
        add_clothing,
        list_clothing,
        update_clothing,
        remove_clothing,
        list_by_season,
        list_by_occasion,
        group_by_type,
        suggest_outfit,
    ],
)

""""
def _demo():
    ""Demonstration des outils independamment du LLM.""
    if os.path.exists(CLOSET_FILE):
        os.remove(CLOSET_FILE)

    print("=== Demonstration des outils ===")
    print(add_clothing("Chemise blanche", "haut", "blanc", "toutes", "travail"))
    print(add_clothing("Pantalon noir", "bas", "noir", "toutes", "travail"))
    print(add_clothing("Derbies noires", "chaussures", "noir", "toutes", "travail"))
    print(add_clothing("Veste beige", "veste", "beige", "mi_saison", "travail"))
    print(add_clothing("T-shirt blanc", "haut", "blanc", "ete", "casual"))
    print(add_clothing("Short bleu", "bas", "bleu", "ete", "casual"))
    print(add_clothing("Baskets blanches", "chaussures", "blanc", "toutes", "casual"))

    print("\n--- Garde-robe complete ---")
    print(list_clothing())

    print("\n--- Par type ---")
    print(group_by_type())

    print("\n--- Suggestion de tenue (travail) ---")
    print(suggest_outfit(occasion="travail"))

    print("\n--- Suggestion de tenue (casual, ete) ---")
    print(suggest_outfit(season="ete", occasion="casual"))

    if os.path.exists(CLOSET_FILE):
        os.remove(CLOSET_FILE)


def _run_unit_tests():
    ""Tests unitaires des outils (deterministes, sans LLM).""

    def reset():
        if os.path.exists(CLOSET_FILE):
            os.remove(CLOSET_FILE)

    def check(nom, condition):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {nom}")
        return condition

    print("=" * 55)
    print("TESTS UNITAIRES")
    print("=" * 55)
    resultats = []

    reset()
    print("\n[add_clothing]")
    r = add_clothing("Pull gris", "haut", "gris", "hiver", "casual")
    resultats.append(check("Ajout valide retourne id", "id: 1" in r))
    resultats.append(check("Type invalide => Erreur", "Erreur" in add_clothing("X", "chapeau", "noir")))
    resultats.append(check("Saison invalide => Erreur", "Erreur" in add_clothing("X", "haut", "noir", season="printemps")))
    resultats.append(check("Occasion invalide => Erreur", "Erreur" in add_clothing("X", "haut", "noir", occasion="ski")))

    reset()
    print("\n[list_clothing]")
    resultats.append(check("Garde-robe vide => message", "vide" in list_clothing()))
    add_clothing("Jean", "bas", "bleu", "toutes", "casual")
    add_clothing("Robe noire", "robe", "noir", "toutes", "soiree")
    resultats.append(check("Filtre par type exclut le reste", "Robe" not in list_clothing("bas")))

    reset()
    print("\n[update_clothing]")
    add_clothing("Manteau", "veste", "marron", "hiver", "casual")
    r = update_clothing(1, color="camel", occasion="travail")
    resultats.append(check("Champs mis a jour", "camel" in r and "travail" in r))
    resultats.append(check("ID inexistant => Erreur", "Erreur" in update_clothing(999, color="noir")))
    resultats.append(check("Saison invalide => Erreur", "Erreur" in update_clothing(1, season="ski")))

    reset()
    print("\n[remove_clothing]")
    add_clothing("Casquette", "accessoire", "noir", "ete", "casual")
    r = remove_clothing(1)
    resultats.append(check("Confirmation suppression", "supprime" in r))
    resultats.append(check("Garde-robe vide apres suppression", "vide" in list_clothing()))

    reset()
    print("\n[list_by_season / list_by_occasion]")
    add_clothing("Maillot de bain", "bas", "bleu", "ete", "sport")
    add_clothing("Echarpe", "accessoire", "gris", "toutes", "toutes")
    add_clothing("Bonnet", "accessoire", "noir", "hiver", "casual")
    r = list_by_season("ete")
    resultats.append(check("Item saison ete + toutes inclus", "Maillot" in r and "Echarpe" in r))
    resultats.append(check("Item hiver exclu de la saison ete", "Bonnet" not in r))
    r2 = list_by_occasion("sport")
    resultats.append(check("Occasion sport trouvee", "Maillot" in r2))

    reset()
    print("\n[group_by_type]")
    add_clothing("Baskets", "chaussures", "blanc", "toutes", "toutes")
    add_clothing("T-shirt", "haut", "blanc", "ete", "casual")
    r = group_by_type()
    resultats.append(check("Types tries et presents", "haut" in r and "chaussures" in r))

    reset()
    print("\n[suggest_outfit]")
    resultats.append(check("Garde-robe vide => pas de chaussures", "Impossible" in suggest_outfit()))
    add_clothing("Chemise blanche", "haut", "blanc", "toutes", "travail")
    add_clothing("Pantalon noir", "bas", "noir", "toutes", "travail")
    add_clothing("Derbies noires", "chaussures", "noir", "toutes", "travail")
    r = suggest_outfit(occasion="travail")
    resultats.append(check("Tenue complete proposee", "Tenue proposee" in r))
    add_clothing("Pantalon rose fluo", "bas", "rose fluo", "toutes", "soiree")
    add_clothing("Escarpins violets", "chaussures", "violet", "toutes", "soiree")
    r2 = suggest_outfit(occasion="soiree")
    resultats.append(check("Combinaison incompatible => message clair", "Impossible" in r2 or "Aucune combinaison" in r2))

    print()
    print("=" * 55)
    nb_pass = sum(resultats)
    print(f"RESULTAT : {nb_pass}/{len(resultats)} tests reussis")
    print("=" * 55)
    reset()


if __name__ == "__main__":
    print(f"Agent 'fashion_agent' cree avec {len(root_agent.tools)} outils")
    for tool in root_agent.tools:
        print(f"  - {tool.__name__}")
    print()
    _demo()
    print()
    _run_unit_tests()
"""
# ------------------------------------------------------------
# Lancement de l'interface web ADK :
#   uv run adk web
#   puis ouvrir : http://localhost:8000
# ------------------------------------------------------------
