# llm_utils.py
"""
Hilfsfunktionen für die Interaktion mit dem Google Gemini Large Language Model (LLM).

Dieses Modul kapselt die Logik für die Kommunikation mit der Google Gemini API
und stellt Funktionen bereit für:
1.  Extraktion strukturierter Filterkriterien aus natürlichsprachlichen
    Benutzereingaben (`get_filters_from_gemini`).
2.  Auswahl von Aktivitätsvorschlägen aus einer Kandidatenliste und Generierung
    einer Begründung basierend auf der Nutzereingabe (`get_selection_and_justification`).
3.  Sichere Aktualisierung der LLM-bezogenen Zustandsvariablen im Streamlit
    Session State (`update_llm_state`).

Requirement 5: Dieses Modul implementiert einen Teil der ML-Anforderung durch
die Nutzung eines Large Language Models (Gemini).
"""

import streamlit as st
import google.generativeai as genai
import json
import traceback # Für detaillierte Fehlermeldungen im Log
from typing import Dict, Any, Optional, List, Tuple # Für Type Hints

# Importiere Konfigurationen (API-Schlüssel-Status, erlaubte LLM-Werte, State Keys)
try:
    import config
except ImportError:
    st.error("Fehler: config.py konnte nicht importiert werden (llm_utils.py).")
    # Setze Dummy-Werte, um Abstürze zu vermeiden
    config = type('obj', (object,), {
        'LLM_POSSIBLE_ARTEN': [], 'LLM_POSSIBLE_PERSONEN_KAT': [],
        'LLM_POSSIBLE_INDOOR_OUTDOOR': [], 'LLM_POSSIBLE_WETTER_PREF': [],
        'LLM_POSSIBLE_ZIELGRUPPEN': [], 'STATE_LLM_FILTERS': 'llm_filters',
        'STATE_LLM_SUGGESTION_IDS': 'llm_suggestion_ids',
        'STATE_LLM_JUSTIFICATION': 'llm_justification',
        'STATE_SHOW_LLM_RESULTS': 'show_llm_results',
        'STATE_NLP_QUERY_SUBMITTED': 'nlp_query_submitted'
    })()


# Die eigentliche Konfiguration von genai (genai.configure) erfolgt in app.py
# um sicherzustellen, dass sie nur einmal beim App-Start passiert.

@st.cache_data(show_spinner="Analysiere Wunsch...") # Cache Ergebnis, um API-Kosten/Zeit zu sparen
def get_filters_from_gemini(
    user_query: str,
    google_api_configured: bool
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extrahiert strukturierte Filter aus einer natürlichsprachlichen Anfrage via Gemini.

    Sendet die `user_query` an das Gemini-Modell (`gemini-1.5-flash`).
    Der Prompt weist das LLM an, relevante Filter (Art, Personen, Preis, Ort, Wetter,
    Zielgruppe) zu identifizieren und *ausschließlich* als valides JSON zurückzugeben.
    Erlaubte Kategorien und Werte werden im Prompt vorgegeben (aus `config.py`).

    Args:
        user_query (str): Die natürlichsprachliche Eingabe des Benutzers.
        google_api_configured (bool): Flag, ob die Google AI API konfiguriert ist.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str]]:
        - filter_dict (Optional[Dict]): Dictionary mit extrahierten Filtern
          (z.B. {'Art': ['Kultur'], 'Preis_Max': 20}) oder None bei Fehler.
        - error_message (Optional[str]): Fehlermeldung für den Benutzer bei
          Problemen (API nicht konfiguriert, ungültige Antwort, etc.), sonst None.
    """
    if not google_api_configured:
        return None, "Google AI API Key nicht korrekt konfiguriert oder fehlt."
    if not user_query:
        return None, "Bitte gib deine Wunsch-Aktivität ein."

    # --- Prompt Design für Filterextraktion ---
    # Definiert die Anweisungen für das LLM.
    # Nutzt f-string, um erlaubte Werte dynamisch aus config.py einzufügen.
    prompt = f"""
    Du bist ein hilfreicher Assistent für die explore-it App, der Nutzereingaben analysiert, um passende Filterkriterien für Aktivitäten zu extrahieren.
    Deine Aufgabe ist es, aus der folgenden Nutzeranfrage die relevanten Kriterien zu identifizieren und sie **NUR als gültiges JSON-Objekt** zurückzugeben.

    Verfügbare Filter und ihre erwarteten Typen/Werte im JSON:
    - "Art": Eine LISTE von einer oder mehreren Aktivitätenarten. Erlaubte Werte: {config.LLM_POSSIBLE_ARTEN}
    - "Preis_Max": Eine ZAHL (Integer oder Float) als Preisobergrenze.
    - "Indoor_Outdoor": EIN String für die Ortspräferenz. Erlaubte Werte: {config.LLM_POSSIBLE_INDOOR_OUTDOOR}
    - "Wetter_Praeferenz": EIN String für die Wetterpräferenz. Erlaubte Werte: {config.LLM_POSSIBLE_WETTER_PREF}
    - "Zielgruppe": Eine LISTE von einer oder mehreren Zielgruppen. Erlaubte Werte: {config.LLM_POSSIBLE_ZIELGRUPPEN}

    WICHTIG:
    1. Gib das Ergebnis **AUSSCHLIESSLICH als valides JSON-Objekt** zurück. KEIN einleitender Text, keine Erklärungen, keine Markdown-Formatierung wie ```json ... ```.
    2. Wenn ein Kriterium aus der Anfrage nicht eindeutig hervorgeht oder keinem erlaubten Wert entspricht, lasse den entsprechenden Schlüssel im JSON weg.
    3. Bei "Art" und "Zielgruppe" gib immer eine Liste von Strings zurück (z.B. ["Sport"] oder ["Familie", "Kinder"]).
    4. Bei "Preis_Max" gib nur die Zahl zurück (z.B. 50 oder 25.5).
    5. Halte dich bei den Werten für "Art", "Indoor_Outdoor", "Wetter_Praeferenz" und "Zielgruppe" strikt an die oben genannten erlaubten Werte.

    Nutzeranfrage: "{user_query}"

    JSON-Ergebnis:
    """
    # print(f"DEBUG: Sende Filter-Prompt an Gemini:\n{prompt}") # Zum Debuggen des Prompts

    try:
        # Initialisiere das Modell (Modellname könnte auch aus config kommen)
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        # Generiere die Antwort vom LLM
        response = model.generate_content(prompt)

        # Bereinige die Antwort: Entferne mögliche Markdown-Code-Blöcke und Whitespace
        # Wichtig, falls das LLM die Anweisung "nur JSON" nicht perfekt befolgt.
        cleaned_response = response.text.strip().lstrip('```json').rstrip('```').strip()

        # Versuche, die bereinigte Antwort als JSON zu parsen
        filter_dict = json.loads(cleaned_response)

        # --- Optionale Validierung der extrahierten Werte ---
        # Prüfe, ob die vom LLM zurückgegebenen Werte auch wirklich erlaubt sind (gemäß config.py).
        # Entferne ungültige Werte oder ganze Schlüssel, wenn sie nicht den Vorgaben entsprechen.
        # Beispiel für 'Art':
        if 'Art' in filter_dict:
            if isinstance(filter_dict['Art'], list):
                # Behalte nur die Arten, die in unserer erlaubten Liste stehen
                valid_arten = [item for item in filter_dict['Art'] if item in config.LLM_POSSIBLE_ARTEN]
                if not valid_arten: del filter_dict['Art'] # Entfernen, wenn keine gültige Art übrig bleibt
                else: filter_dict['Art'] = valid_arten
            else:
                del filter_dict['Art'] # Entfernen, wenn es keine Liste ist

        # (Hier könnten analog Validierungen für Indoor_Outdoor, Wetter_Praeferenz, Zielgruppe folgen)

        # Beispiel für 'Preis_Max':
        if 'Preis_Max' in filter_dict:
            if not isinstance(filter_dict['Preis_Max'], (int, float)):
                del filter_dict['Preis_Max'] # Entfernen, wenn es keine Zahl ist

        # print(f"DEBUG: Von Gemini extrahierte und validierte Filter: {filter_dict}")
        return filter_dict, None # Erfolg: Gib das validierte Dictionary zurück

    except json.JSONDecodeError:
        # Fehler, falls die Antwort des LLM kein gültiges JSON war.
        # print(f"FEHLER: LLM (Filter) gab kein gültiges JSON zurück. Antwort:\n{cleaned_response}")
        return None, "Die Antwort der KI (Filter) war nicht im erwarteten Format. Versuche es anders zu formulieren."
    except Exception as e:
        # Fange andere Fehler ab (z.B. API nicht erreichbar, Kontingent überschritten).
        # print(f"FEHLER: Unerwarteter Fehler bei LLM (Filter) Kommunikation: {e}")
        # traceback.print_exc() # Nützlich für Entwickler zur Fehlersuche (gibt Details in Konsole aus)
        return None, f"Ein Fehler ist bei der Kommunikation mit der KI (Filter) aufgetreten: {type(e).__name__}"


# Kein Caching hier (@st.cache_data), da die Kandidatenliste (candidate_info_string) sich ständig ändert.
def get_selection_and_justification(
    user_query: str,
    candidate_info_string: str,
    google_api_configured: bool,
    num_suggestions: int = 5
    ) -> Tuple[Optional[List[int]], Optional[str], Optional[str]]:
    """
    Wählt passende Aktivitäten aus Kandidaten aus und begründet die Auswahl via Gemini.

    Sendet die ursprüngliche Nutzeranfrage und eine formatierte Liste von Kandidaten-
    aktivitäten an das Gemini-Modell. Das LLM soll die besten `num_suggestions`
    Kandidaten anhand ihrer IDs auswählen und eine textuelle Begründung liefern.
    Die Antwort wird als spezifisches JSON-Objekt erwartet ({ "suggestion_ids": [...], "justification": "..." }).

    Args:
        user_query (str): Die ursprüngliche natürlichsprachliche Anfrage des Benutzers.
        candidate_info_string (str): Ein String, der die Kandidatenaktivitäten
            beschreibt (mit ID, Name, Art, Preis etc.).
        google_api_configured (bool): Flag, ob die Google AI API konfiguriert ist.
        num_suggestions (int): Maximale Anzahl der vorzuschlagenden Aktivitäts-IDs.

    Returns:
        Tuple[Optional[List[int]], Optional[str], Optional[str]]:
        - suggestion_ids (Optional[List[int]]): Liste der IDs (als Zahlen) der vom LLM
          ausgewählten Aktivitäten. Kann leer sein ([]). Gibt None bei einem API-Fehler zurück.
        - justification (Optional[str]): Die vom LLM generierte Begründung. Gibt None bei einem API-Fehler zurück.
        - error_message (Optional[str]): Fehlermeldung für den Benutzer bei Problemen, sonst None.
    """
    if not google_api_configured:
        return None, None, "Google AI API Key nicht korrekt konfiguriert oder fehlt."
    if not user_query:
        user_query = "deinem Wunsch" # Generischer Text, falls die Originalanfrage fehlt
    if not candidate_info_string:
        # Wenn keine Kandidaten da sind, kann das LLM nichts auswählen.
        # Das ist kein technischer Fehler, daher geben wir leere Liste und passende Begründung zurück.
        return [], "Es wurden keine Aktivitäten gefunden, die den Filterkriterien entsprechen, um Vorschläge zu machen.", None

    # --- Prompt Design für Vorschlagsgenerierung ---
    # Weist das LLM an, aus der Kandidatenliste auszuwählen und die Antwort
    # in einem exakten JSON-Format zu strukturieren.
    prompt = f"""
    Du bist ein hilfreicher Assistent für die explore-it App. Deine Aufgabe ist es, aus einer Liste potenzieller Aktivitäten die besten Vorschläge basierend auf dem Nutzerwunsch auszuwählen und diese Auswahl zu begründen.

    Nutzerwunsch: "{user_query}"

    Potenzielle Kandidaten (Aktivitäten):
    ---
    {candidate_info_string}
    ---

    Aufgaben:
    1. Wähle bis zu {num_suggestions} Aktivitäten aus der KANDIDATEN-Liste aus, die am besten zum Nutzerwunsch passen. Konzentriere dich auf die IDs der Aktivitäten. Gib eine leere Liste zurück, wenn keine passenden Kandidaten gefunden werden.
    2. Schreibe eine kurze, freundliche Begründung (1-3 Sätze), warum diese Auswahl gut zum Nutzerwunsch passt ODER warum keine passenden Vorschläge gemacht werden konnten.
    3. Gib das Ergebnis **NUR als gültiges JSON-Objekt** zurück. Das JSON muss genau die folgenden zwei Schlüssel enthalten:
       - "suggestion_ids": Eine LISTE von ZAHLEN (Integer) der ausgewählten IDs. Muss eine Liste sein, auch wenn leer: [].
       - "justification": Ein STRING mit deiner Begründung.

    Beispiel JSON-Format (mit Vorschlägen):
    {{
      "suggestion_ids": [1, 8, 17],
      "justification": "Diese Vorschläge passen gut, da sie kulturelle Erlebnisse in St. Gallen bieten und auch eine Option für Naturfreunde dabei ist, wie im Wunsch erwähnt."
    }}

    Beispiel JSON-Format (ohne Vorschläge):
     {{
       "suggestion_ids": [],
       "justification": "Leider konnten keine Aktivitäten gefunden werden, die sowohl kulturell sind als auch für grosse Gruppen unter 10 CHF geeignet sind."
     }}

    WICHTIG: Stelle sicher, dass "suggestion_ids" eine Liste von Zahlen ist (kann leer sein []) und "justification" ein Text (String). Gib **KEINEN** anderen Text oder Formatierungen ausserhalb des JSON-Objekts zurück.

    JSON-Ergebnis:
    """
    # print(f"DEBUG: Sende Vorschlags-Prompt an Gemini...") # Debug-Ausgabe (Prompt ist oft sehr lang)

    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        response = model.generate_content(prompt)
        # Bereinige die Antwort wieder von möglichem Markdown etc.
        cleaned_response = response.text.strip().lstrip('```json').rstrip('```').strip()

        # Parse JSON
        result_dict = json.loads(cleaned_response)

        # Validiere die Struktur und Typen der Antwort
        suggestion_ids = result_dict.get("suggestion_ids")
        justification = result_dict.get("justification")

        # Prüfe, ob beide Schlüssel vorhanden sind und die Typen stimmen (Liste von Zahlen, String).
        if isinstance(suggestion_ids, list) and \
           all(isinstance(i, int) for i in suggestion_ids) and \
           isinstance(justification, str):
            # Erfolg: Korrektes Format erhalten.
            # Begrenze die Anzahl der IDs auf num_suggestions, falls das LLM mehr zurückgibt.
            # print(f"DEBUG: Von Gemini empfangene Vorschläge: IDs={suggestion_ids[:num_suggestions]}, Begründung='{justification}'")
            return suggestion_ids[:num_suggestions], justification, None # Kein Fehler
        else:
            # Fehler: Unerwartetes Format im JSON. Erstelle detaillierte Fehlermeldung.
            error_details = []
            if not isinstance(suggestion_ids, list): error_details.append("'suggestion_ids' ist keine Liste")
            elif not all(isinstance(i, int) for i in suggestion_ids): error_details.append("'suggestion_ids' enthält nicht nur Zahlen")
            if not isinstance(justification, str): error_details.append("'justification' ist kein Text")
            error_msg_details = "; ".join(error_details) if error_details else "Unbekanntes Formatproblem"
            # print(f"FEHLER: LLM (Vorschlag) gab unerwartetes Format zurück. Details: {error_msg_details}. Antwort:\n{cleaned_response}")
            return None, None, f"Die KI-Antwort (Vorschlag) hatte nicht das erwartete Format ({error_msg_details})."

    except json.JSONDecodeError:
        # Fehler, wenn die Antwort kein gültiges JSON ist.
        # print(f"FEHLER: LLM (Vorschlag) gab kein gültiges JSON zurück. Antwort:\n{cleaned_response}")
        return None, None, "Die Antwort der KI (Vorschlag) konnte nicht verarbeitet werden (ungültiges JSON)."
    except Exception as e:
        # Fange andere Fehler ab (API-Fehler etc.).
        # print(f"FEHLER: Unerwarteter Fehler bei LLM (Vorschlag) Kommunikation: {e}")
        # traceback.print_exc()
        return None, None, f"Ein Fehler ist bei der Kommunikation mit der KI (Vorschlag) aufgetreten: {type(e).__name__}"

def update_llm_state(**kwargs: Any) -> None:
    """
    Aktualisiert Variablen im Streamlit Session State, die mit dem LLM zu tun haben.

    Diese Hilfsfunktion dient dazu, den Code in `app.py` sauberer zu halten.
    Sie nimmt die Ergebnisse der LLM-Aufrufe (Filter, Vorschläge, Begründung)
    entgegen und speichert sie sicher im Session State unter den in `config.py`
    definierten Schlüsseln.

    Mögliche Argumente (übergeben als Schlüsselwortargumente, z.B. filters=mein_filter_dict):
        filters (Optional[Dict]): Das extrahierte Filter-Dictionary.
        suggestion_ids (Optional[List[int]]): Liste der vorgeschlagenen IDs.
        justification (Optional[str]): Die textuelle Begründung.
        show_results (bool): Flag, ob die LLM-Ergebnisse angezeigt werden sollen.
        query (Optional[str]): Die zuletzt übermittelte Nutzeranfrage.
        reset_suggestions (bool): Wenn True, werden Vorschläge/Begründung zusätzlich
                                  explizit auf None gesetzt (nützlich bei neuer Anfrage).
    """
    # Mapping von Funktionsargument-Namen zu Session State Keys aus config.py
    allowed_keys_map = {
        'filters': config.STATE_LLM_FILTERS,
        'suggestion_ids': config.STATE_LLM_SUGGESTION_IDS,
        'justification': config.STATE_LLM_JUSTIFICATION,
        'show_results': config.STATE_SHOW_LLM_RESULTS,
        'query': config.STATE_NLP_QUERY_SUBMITTED
    }

    # Verarbeite 'reset_suggestions' zuerst
    reset_suggestions = kwargs.get('reset_suggestions', False)
    if reset_suggestions:
        st.session_state[config.STATE_LLM_SUGGESTION_IDS] = None
        st.session_state[config.STATE_LLM_JUSTIFICATION] = None
        # print(f"DEBUG: Session State '{config.STATE_LLM_SUGGESTION_IDS}' und '{config.STATE_LLM_JUSTIFICATION}' zurückgesetzt.")

    # Iteriere durch die erwarteten Schlüssel und aktualisiere den State, wenn der Schlüssel übergeben wurde.
    for key, state_key in allowed_keys_map.items():
        if key in kwargs:
            st.session_state[state_key] = kwargs[key]
            # print(f"DEBUG: Session State '{state_key}' aktualisiert mit Wert: {kwargs[key]}")