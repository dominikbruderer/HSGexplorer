# app.py
"""
Hauptanwendung für HSGexplorer - Ein intelligenter Aktivitätsfinder für St. Gallen und Umgebung.

Dieses Skript initialisiert und startet die Streamlit-Webanwendung HSGexplorer.
Es dient als zentraler Orchestrator und löst folgendes Problem (Requirement 1):
Nutzern dabei zu helfen, passende Freizeitaktivitäten in St. Gallen und Umgebung zu finden,
basierend auf manuellen Filtern, Wetterdaten und optionalen, KI-gestützten Vorschlägen
sowie personalisierten Empfehlungen basierend auf Nutzerbewertungen.

Hauptaufgaben:
- Seitenkonfiguration und Initialisierung des Session State ("Gedächtnis" der App).
- Laden und Vorverarbeiten der Aktivitätsdaten (via data_utils).
- Berechnung von Features für das ML-Empfehlungssystem (via recommender).
- Aufbau der Benutzeroberfläche (Titel, KI-Eingabe, Sidebar, Karte/Wetter,
  Aktivitätenliste, Personalisierungsbereich) (via ui_components).
- Verarbeitung von Benutzereingaben (Filter, KI-Anfrage, Likes/Dislikes).
- Interaktion mit dem LLM (Google Gemini) (via llm_utils).
- Anwendung der Filterlogik (Basis- und Wetterfilter) (via logic).
- Abruf von Wetterdaten via API (via weather_utils).
- Verwaltung des Anwendungszustands (KI-Modus vs. manueller Modus, etc.).
- Darstellung der Ergebnisse und Visualisierungen (via ui_components).
- Implementierung personalisierter Empfehlungen (via recommender).
"""

# --- Importe ---
import streamlit as st
import pandas as pd
import datetime
import os # Wird indirekt von Modulen verwendet -> Kann hier weg, wenn nicht direkt genutzt
import traceback # Für detailliertere Fehlermeldungen -> Entfernen für finale Version
import random # Für Exploration und Shuffle bei Empfehlungen
from typing import List, Dict, Any, Optional, Set, Tuple, Union #Für Type Hints

# --- Eigene Module importieren ---
# Lade alle benötigten Funktionen und Konstanten aus unseren anderen .py Dateien.
try:
    import config # Globale Konfigurationen und Konstanten (config.py)
    from data_utils import load_data # Daten laden/bereinigen (data_utils.py)
    from weather_utils import get_weather_forecast_for_day # Wetter-API Abruf (weather_utils.py)
    from llm_utils import get_filters_from_gemini, get_selection_and_justification, update_llm_state # LLM Interaktion (llm_utils.py)
    from logic import apply_base_filters, apply_weather_filter # Filterlogik (logic.py)
    from ui_components import ( # UI-Elemente (ui_components.py)
        display_sidebar, display_map, display_weather_overview,
        display_activity_details, display_recommendation_card,
        display_preference_visualization
    )
    from recommender import ( # ML-Funktionen für Empfehlungen (recommender.py)
        preprocess_features, calculate_user_profile, get_profile_recommendations,
        calculate_preference_scores, generate_profile_label,
        calculate_top_target_groups, get_liked_prices
    )
except ImportError as e:
    st.error(f"Fataler Fehler beim Importieren von Modulen: '{e}'. Stellen Sie sicher, dass alle .py-Dateien im selben Verzeichnis liegen und alle Abhängigkeiten installiert sind (siehe requirements.txt).")
    st.stop() # App kann ohne Module nicht sinnvoll starten

# --- Globale Konfiguration und Initialisierung ---

# 1. Seitenkonfiguration (Titel im Browser-Tab, Seitenlayout, Icon)
# Muss als erstes Streamlit-Kommando aufgerufen werden.
st.set_page_config(page_title="HSGexplorer", layout="wide", page_icon="🗺️")

# 2. Session State initialisieren (Das "Gedächtnis" der App)
# Geht alle Standardwerte aus config.py durch und legt sie im Session State an,
# falls sie dort noch nicht existieren (passiert nur beim allerersten Start der Session).
# print("DEBUG: Initializing session state...") # Debug
for key, default_value in config.DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
# print("DEBUG: Session state initialized.") # Debug

# 3. Google AI (Gemini) sicher konfigurieren
# Dies geschieht nur einmal pro Session, wenn der API-Key gültig ist.
google_api_error_handled = False # Hilfsvariable für Fehlermeldung
# Prüfe, ob Key in config.py als gültig erkannt wurde UND ob die Konfiguration in dieser Session noch nicht lief
if config.GOOGLE_API_CONFIGURED and not st.session_state.get(config.STATE_GOOGLE_AI_CONFIGURED, False):
    try:
        import google.generativeai as genai
        # Konfiguriere die Gemini-Bibliothek mit dem API-Key aus config.py
        genai.configure(api_key=config.GOOGLE_API_KEY)
        # Merke im Session State, dass die Konfiguration erfolgreich war
        st.session_state[config.STATE_GOOGLE_AI_CONFIGURED] = True
        # print("INFO: Google AI API erfolgreich konfiguriert.") # Debug
    except ImportError:
        # Fehler, falls das google-generativeai Paket fehlt
        st.error("Fehler: Das Paket 'google-generativeai' fehlt. Installation: pip install google-generativeai")
        st.session_state[config.STATE_GOOGLE_AI_CONFIGURED] = False
        google_api_error_handled = True
    except Exception as e:
        # Fange andere Fehler bei der Konfiguration ab
        st.error(f"Fehler bei der Konfiguration der Google AI API: {e}")
        st.session_state[config.STATE_GOOGLE_AI_CONFIGURED] = False
        google_api_error_handled = True
elif not config.GOOGLE_API_CONFIGURED:
    # Fall: Key wurde schon in config.py als fehlend/ungültig erkannt
    st.session_state[config.STATE_GOOGLE_AI_CONFIGURED] = False


# --- API Key Warnungen (nur einmal pro Session anzeigen) ---
# Überprüfe den finalen Konfigurationsstatus aus dem Session State
is_google_ai_really_configured = st.session_state.get(config.STATE_GOOGLE_AI_CONFIGURED, False)
# Zeige Warnung, wenn Google AI nicht konfiguriert ist UND die Warnung noch nicht gezeigt wurde
if not is_google_ai_really_configured and not google_api_error_handled and not st.session_state.get(config.STATE_GOOGLE_KEY_WARNING_SHOWN, False):
    st.warning("Google AI API Key nicht korrekt konfiguriert oder Konfiguration fehlgeschlagen. KI-Funktionen (Suche, Vorschläge) sind deaktiviert.", icon="🔑")
    st.session_state[config.STATE_GOOGLE_KEY_WARNING_SHOWN] = True # Merken, dass Warnung gezeigt wurde

# Zeige Warnung, wenn OpenWeatherMap Key nicht konfiguriert ist UND die Warnung noch nicht gezeigt wurde
if not config.OPENWEATHERMAP_API_CONFIGURED and not st.session_state.get(config.STATE_OWM_KEY_WARNING_SHOWN, False):
    st.warning("OpenWeatherMap API Key nicht korrekt konfiguriert. Wetterfunktionen sind eingeschränkt.", icon="🌦️")
    st.session_state[config.STATE_OWM_KEY_WARNING_SHOWN] = True

# --- Daten laden und ML-Vorbereitung ---

# Lade Aktivitätsdaten mit der Funktion aus data_utils.py
CSV_PATH_NEU = "aktivitaeten_neu.csv" # Pfad zur Datenquelle
df_activities = load_data(CSV_PATH_NEU) # Enthält jetzt die bereinigten Daten

# Berechne die Feature-Matrix für ML-Empfehlungen (nur einmal pro Session, wenn noch nicht vorhanden)
# Diese Matrix wird für die Nutzerprofilierung und Empfehlungen benötigt (von recommender.py)
if config.STATE_FEATURES_MATRIX not in st.session_state or st.session_state[config.STATE_FEATURES_MATRIX] is None:
    if not df_activities.empty:
        # print("INFO: Berechne Feature-Matrix für Empfehlungen...") # Debug
        with st.spinner('Analysiere Aktivitäten für Empfehlungen...'): # Zeige Spinner während Berechnung
            # Rufe die Funktion aus recommender.py auf
            _, features = preprocess_features(df_activities)
        # Speichere die berechnete Matrix im Session State
        st.session_state[config.STATE_FEATURES_MATRIX] = features
        if features is None or features.shape[1] == 0 :
            # print("WARNUNG: Keine Features für Empfehlungen extrahiert.") # Debug
            st.warning("Konnte keine Merkmale für Empfehlungen extrahieren.", icon="⚠️")
        # else: # Debug
             # print(f"INFO: Feature-Matrix ({features.shape}) berechnet und im State gespeichert.")
    else:
        # Stelle sicher, dass der State auch bei leeren Daten None ist
        st.session_state[config.STATE_FEATURES_MATRIX] = None
        # print("INFO: Keine Aktivitätsdaten, Feature-Matrix-Berechnung übersprungen.") # Debug

# Prüfe, ob Daten erfolgreich geladen wurden. Wenn nicht, kann die App kaum sinnvoll laufen.
if df_activities.empty:
    st.error("Fataler Fehler: Keine Aktivitätsdaten gefunden oder geladen. Die App kann nicht richtig funktionieren.")
    # Ggf. st.stop() hier, wenn die App ohne Daten gar keinen Sinn macht.


# Konstante: Mindestanzahl an Likes, bevor das detaillierte Nutzerprofil angezeigt wird.
MIN_LIKES_FOR_PROFILE = 5 # Kann angepasst werden 

# --- Callback Funktion für Like/Dislike Buttons ---
# Diese Funktion wird aufgerufen, wenn ein Nutzer auf 👍 oder 👎 in der Vorschlagskarte klickt.
def update_recommendations(clicked_activity_id: int, rating: int) -> None:
    """
    Aktualisiert die Nutzerbewertungen (Likes/Dislikes) im Session State,
    berechnet das Nutzerprofil neu, holt neue Empfehlungen und aktualisiert
    die Daten für die Präferenzvisualisierung.

    Args:
        clicked_activity_id (int): Die ID der Aktivität, die bewertet wurde.
        rating (int): Die Bewertung (1 für Like, -1 für Dislike).
    """
    # print(f"Callback ausgelöst: ID={clicked_activity_id}, Rating={rating}") # Debug
    if clicked_activity_id is None or clicked_activity_id == -1: # Sicherheitscheck
        # print("WARNUNG (Callback): Ungültige activity_id empfangen.") # Debug
        return

    # 1. Listen der Likes/Dislikes im Session State aktualisieren
    # Hole aktuelle Listen (oder leere Listen, falls noch nicht vorhanden)
    liked_ids_list = st.session_state.get(config.STATE_LIKED_IDS, [])
    disliked_ids_list = st.session_state.get(config.STATE_DISLIKED_IDS, [])
    try: # Stelle sicher, dass ID eine Zahl ist
        clicked_activity_id = int(clicked_activity_id)
    except (ValueError, TypeError):
        # print(f"WARNUNG (Callback): Konnte ID '{clicked_activity_id}' nicht in int umwandeln.") # Debug
        return

    # Füge ID zur entsprechenden Liste hinzu und entferne sie ggf. aus der anderen Liste
    if rating == 1: # Like
        if clicked_activity_id not in liked_ids_list: liked_ids_list.append(clicked_activity_id)
        if clicked_activity_id in disliked_ids_list: disliked_ids_list.remove(clicked_activity_id)
    elif rating == -1: # Dislike
        if clicked_activity_id not in disliked_ids_list: disliked_ids_list.append(clicked_activity_id)
        if clicked_activity_id in liked_ids_list: liked_ids_list.remove(clicked_activity_id)
    # Speichere die aktualisierten Listen zurück im Session State
    st.session_state[config.STATE_LIKED_IDS] = liked_ids_list
    st.session_state[config.STATE_DISLIKED_IDS] = disliked_ids_list
    # print(f"DEBUG (Callback): State aktualisiert - Likes: {st.session_state[config.STATE_LIKED_IDS]}, Dislikes: {st.session_state[config.STATE_DISLIKED_IDS]}") # Debug

    # 2. Nutzerprofil und Empfehlungen neu berechnen (nur wenn Features vorhanden)
    features_matrix = st.session_state.get(config.STATE_FEATURES_MATRIX)
    if features_matrix is not None and not df_activities.empty:
        # Berechne den neuen Profil-Vektor mit der Funktion aus recommender.py
        user_profile = calculate_user_profile(
            liked_ids=st.session_state[config.STATE_LIKED_IDS],
            disliked_ids=st.session_state[config.STATE_DISLIKED_IDS], # Derzeit nicht verwendet
            features_matrix=features_matrix,
            df=df_activities
        )
        # Speichere das neue Profil im State
        st.session_state[config.STATE_USER_PROFILE] = user_profile

        # Wenn ein Profil berechnet werden konnte, hole neue Empfehlungen
        if user_profile is not None:
            # Definiere alle bereits bewerteten IDs (damit diese nicht erneut vorgeschlagen werden)
            rated_ids: Set[int] = set(st.session_state[config.STATE_LIKED_IDS]) | set(st.session_state[config.STATE_DISLIKED_IDS])
            # Hole neue Empfehlungs-IDs mit der Funktion aus recommender.py
            new_recommendation_ids = get_profile_recommendations(
                user_profile=user_profile,
                features_matrix=features_matrix,
                df=df_activities,
                rated_ids=rated_ids,
                n=5 # Anzahl der Vorschläge, die im Personalisierungs-Expander rotieren sollen
            )
            # Speichere die neuen Vorschlags-IDs im State (diese werden dann angezeigt)
            st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS] = new_recommendation_ids
            # print(f"DEBUG (Callback): Neue Empfehlungs-IDs im State: {st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS]}") # Debug
        else:
            # Kein Profil -> keine Empfehlungen
            st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS] = []
            # print("DEBUG (Callback): Kein User Profile berechnet, Empfehlungsliste geleert.") # Debug

        # 3. Daten für die Präferenz-Visualisierung neu berechnen
        # Berechne, welche Arten / Zielgruppen / Preise der Nutzer bevorzugt (basierend auf Likes)
        pref_scores = calculate_preference_scores(st.session_state[config.STATE_LIKED_IDS], df_activities)
        profile_label = generate_profile_label(pref_scores) # Erzeuge Label wie "Kultur-Fan"
        # Speichere das Label im State
        st.session_state[config.STATE_USER_PROFILE_LABEL] = profile_label
        # print(f"DEBUG (Callback): Profil-Label im State: {st.session_state[config.STATE_USER_PROFILE_LABEL]}") # Debug
    else:
        # Fallback, wenn keine Features/Daten vorhanden sind
        st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS] = []
        st.session_state[config.STATE_USER_PROFILE_LABEL] = None
        # print("DEBUG (Callback): Keine Feature-Matrix oder Daten für Profil/Empfehlungen/Label.") # Debug

    # WICHTIG: KEIN st.rerun() am Ende eines Callbacks! Streamlit führt es automatisch aus.

# --- UI Aufbau ---

# 1. Titel und Einleitung
st.title("HSGexplorer")
st.write("Dein intelligenter Assistent für Aktivitäten in St. Gallen und Umgebung.")

# 2. KI-Eingabezeile
st.subheader("Was möchtest du unternehmen?")
nlp_query = st.text_input("Beschreibe deine Wunsch-Aktivität:",
                        placeholder="z.B. 'Wandern mit Aussicht', 'Museum für Familien', 'Günstiges Date bei Regen'",
                        label_visibility="collapsed", # Versteckt das Label "Beschreibe..." (spart Platz)
                        key="nlp_query_input") # Eindeutiger Schlüssel
# Button für KI-Suche; deaktiviert, wenn KI nicht konfiguriert oder keine Daten geladen wurden
nlp_button_pressed = st.button("KI-Vorschläge finden", key="nlp_button_main", type="primary",
                            disabled=not is_google_ai_really_configured or df_activities.empty)

# Platzhalter für dynamische Texte (werden später gefüllt)
justification_placeholder = st.empty() # Für die Begründung des LLM
extracted_filters_placeholder = st.empty() # Für die Anzeige der extrahierten Filter

# --- Logik: Initiale Empfehlungen für Personalisierungs-Expander laden ---
# Füllt die Vorschlagsliste beim ersten Laden der Seite oder nach einem Reset
# mit zufälligen, noch nicht bewerteten Aktivitäten, damit der Nutzer etwas zum Klicken hat.
if not st.session_state.get(config.STATE_RECOMMENDATIONS_TO_SHOW_IDS):
    # print("DEBUG: Keine Vorschläge im State, lade initiale Empfehlungen...") # Debug
    if not df_activities.empty:
        # Finde alle IDs, die noch nicht bewertet wurden
        rated_ids = set(st.session_state.get(config.STATE_LIKED_IDS, [])) | set(st.session_state.get(config.STATE_DISLIKED_IDS, []))
        initial_candidates = df_activities[config.COL_ID].dropna().unique().tolist()
        valid_unrated_ids = [int(id_) for id_ in initial_candidates if isinstance(id_, (int, float)) and pd.notna(id_) and int(id_) not in rated_ids and int(id_) != -1]
        if valid_unrated_ids: random.shuffle(valid_unrated_ids) # Mische die Kandidaten
        # Speichere die ersten 5 (oder weniger) im State für die Anzeige
        st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS] = valid_unrated_ids[:5]
    else:
        st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS] = []
    # print(f"DEBUG: Initiale/aktualisierte Empfehlungs-IDs geladen: {st.session_state.get(config.STATE_RECOMMENDATIONS_TO_SHOW_IDS)}") # Debug


# 3. Personalisierungs-Expander (Vorschlagskarte & Präferenz-Visualisierung)
st.markdown("---") # Trennlinie
with st.expander("✨ Personalisierte Vorschläge (Beta)", expanded=True):
    # Hole die IDs, die aktuell angezeigt werden sollen (entweder initiale oder neu berechnete)
    recommendation_ids_for_card = st.session_state.get(config.STATE_RECOMMENDATIONS_TO_SHOW_IDS, [])

    # Layout: Zwei Spalten für Karte und Visualisierung nebeneinander
    col_card, col_viz = st.columns([1, 1]) # Verhältnis 1:1

    # Linke Spalte: Vorschlagskarte anzeigen
    with col_card:
        st.markdown("**Aktueller Vorschlag:**")
        if not recommendation_ids_for_card:
             st.caption("Bewerte Aktivitäten 👍 / 👎, um hier passende Vorschläge zu sehen.")
        elif df_activities.empty:
             st.warning("Keine Aktivitätsdaten zum Anzeigen des Vorschlags.")
        else:
            # Zeige immer die *erste* Aktivität aus der aktuellen Vorschlagsliste an
            single_suggestion_id = recommendation_ids_for_card[0]
            try:
                activity_id_int = int(single_suggestion_id)
                # Finde die Datenzeile für diese Aktivität im Haupt-DataFrame
                card_row_df = df_activities[df_activities[config.COL_ID] == activity_id_int]
                if not card_row_df.empty:
                     card_row = card_row_df.iloc[0]
                     # Rufe die Funktion aus ui_components.py auf, um die Karte anzuzeigen
                     # Wichtig: Übergabe der Callback-Funktionen für die Buttons!
                     display_recommendation_card(
                         activity_row=card_row,
                         card_key_suffix=f"single_rec_{activity_id_int}", # Eindeutiger Key-Teil
                         on_like_callback=update_recommendations, # Name der Callback-Funktion
                         on_dislike_callback=update_recommendations # Name der Callback-Funktion
                         # Die Argumente (ID, Rating) werden in display_recommendation_card im Button definiert
                     )
                else:
                     # Fall: Aktivität aus Vorschlagsliste nicht mehr in Daten gefunden (sollte selten sein)
                     st.warning(f"Vorgeschlagene Aktivität ID {activity_id_int} nicht gefunden.")
                     # Entferne die ungültige ID aus der Liste und lade neu, um die nächste anzuzeigen
                     if activity_id_int in st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS]:
                          st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS].pop(0)
                          st.rerun() # Lade UI neu
            except Exception as e:
                 st.error(f"Fehler bei Vorschlagskarte für ID '{single_suggestion_id}': {e}")

    # Rechte Spalte: Präferenzvisualisierung anzeigen
    with col_viz:
        st.markdown("**Deine Präferenzen (gelernt):**")
        current_likes_count = len(st.session_state.get(config.STATE_LIKED_IDS, []))

        # Zeige Visualisierung erst, wenn eine Mindestanzahl an Likes erreicht ist
        if current_likes_count >= MIN_LIKES_FOR_PROFILE:
            # Hole oder berechne die Daten für die Visualisierungen
            current_profile_label = st.session_state.get(config.STATE_USER_PROFILE_LABEL) # Label aus State holen
            liked_ids_for_viz = st.session_state.get(config.STATE_LIKED_IDS, [])
            # Berechne Scores/Listen mit Funktionen aus recommender.py
            pref_scores_art = calculate_preference_scores(liked_ids_for_viz, df_activities)
            top_groups = calculate_top_target_groups(liked_ids_for_viz, df_activities, top_n=5)
            price_list = get_liked_prices(liked_ids_for_viz, df_activities, include_free=False) # Hier z.B. ohne kostenlose

            # Rufe die Funktion aus ui_components.py auf, um die Diagramme anzuzeigen
            display_preference_visualization(
                profile_label=current_profile_label,
                preference_scores_art=pref_scores_art,
                top_target_groups=top_groups,
                liked_prices_list=price_list
            )

            # Button, um explizit Empfehlungen basierend auf dem gelernten Profil anzuzeigen
            show_profile_recommendations = st.button("Zeige passende Aktivitäten für mein Profil", key="btn_show_profile_rec")
            if show_profile_recommendations:
                # print("DEBUG: Button 'Zeige passende Aktivitäten für mein Profil' geklickt.") # Debug
                current_user_profile = st.session_state.get(config.STATE_USER_PROFILE)
                features_matrix = st.session_state.get(config.STATE_FEATURES_MATRIX)
                # Prüfe, ob Profil und Features vorhanden sind
                if current_user_profile is not None and features_matrix is not None and not df_activities.empty:
                    rated_ids = set(st.session_state.get(config.STATE_LIKED_IDS, [])) | set(st.session_state.get(config.STATE_DISLIKED_IDS, []))
                    with st.spinner('Suche passende Aktivitäten...'): # Spinner anzeigen
                         # Hole explizite Empfehlungen (ohne Exploration)
                         explicit_ids = get_profile_recommendations(
                             user_profile=current_user_profile, features_matrix=features_matrix,
                             df=df_activities, rated_ids=rated_ids, n=10, exploration_rate=0
                         )
                    # Speichere diese Liste im State; sie wird dann unten statt der normalen Liste angezeigt
                    st.session_state[config.STATE_EXPLICIT_RECOMMENDATIONS] = explicit_ids
                    # print(f"INFO: Explizite Empfehlungs-IDs im State gespeichert: {explicit_ids}") # Debug
                    st.rerun() # Lade App neu, um die Liste anzuzeigen
                elif current_user_profile is None: st.warning("Bitte bewerte zuerst einige Aktivitäten, um ein Profil zu erstellen.")
                else: st.error("Fehler: Benötigte Profildaten nicht verfügbar.")

        # Hinweis anzeigen, wenn noch nicht genug Likes gesammelt wurden
        elif current_likes_count > 0:
            st.caption(f"Bewerte noch {MIN_LIKES_FOR_PROFILE - current_likes_count} weitere Aktivität(en) positiv 👍, um dein detailliertes Profil zu sehen!")
        else: # Noch gar nichts geliked
            st.caption("Bewerte einige Aktivitäten 👍, um hier deine Präferenzen zu sehen!")

        # Reset-Button für die Personalisierung (Likes/Dislikes/Profil löschen)
        if st.button("Personalisierung zurücksetzen", key="btn_reset_prefs"):
             # Setze alle relevanten State-Variablen zurück
             st.session_state[config.STATE_LIKED_IDS] = []
             st.session_state[config.STATE_DISLIKED_IDS] = []
             st.session_state[config.STATE_USER_PROFILE] = None
             st.session_state[config.STATE_USER_PROFILE_LABEL] = None
             st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS] = [] # Leert auch Vorschlagskarte
             st.session_state[config.STATE_EXPLICIT_RECOMMENDATIONS] = None # Leert explizite Liste
             # print("INFO: Personalisierung zurückgesetzt.") # Debug
             st.rerun() # Lade App neu, um Reset anzuzeigen

# 4. Sidebar anzeigen (Funktion aus ui_components.py)
# Diese Funktion gibt die in der Sidebar ausgewählten Filterwerte zurück.
today = datetime.date.today()
datum, consider_weather_sb, aktivitaetsart_sb, personen_anzahl_sb, budget_sb, reset_llm_pressed = display_sidebar(
    df_activities, today, config.OPENWEATHERMAP_API_CONFIGURED
)

# --- Logik: Zustandsverwaltung (KI-Modus vs. Manueller Modus) ---

# Aktionen basierend auf Button-Klicks aus der Sidebar oder der Hauptseite:
# Fall 1: Nutzer klickt in Sidebar auf "Manuelle Filter verwenden"
if reset_llm_pressed:
    # print("INFO: Manuelle Filter aktiviert, LLM State wird zurückgesetzt.") # Debug
    # Setze den LLM-Zustand zurück (Filter, Vorschläge, Begründung, Anzeige-Flag)
    update_llm_state(show_results=False, query=None, filters=None, suggestion_ids=None, justification=None)
    # Setze auch die expliziten Profil-Empfehlungen zurück
    st.session_state[config.STATE_EXPLICIT_RECOMMENDATIONS] = None
    # Leere die dynamischen Text-Platzhalter
    extracted_filters_placeholder.empty()
    justification_placeholder.empty()
    st.rerun() # Lade App neu, um im manuellen Modus zu sein

# Fall 2: Nutzer klickt auf "KI-Vorschläge finden"
if nlp_button_pressed:
    if not df_activities.empty and nlp_query: # Nur wenn Daten da sind und eine Anfrage eingegeben wurde
        # print(f"INFO: NLP Button gedrückt, starte Filterextraktion für: '{nlp_query}'") # Debug
        # Setze explizite Profil-Empfehlungen zurück, wenn neue KI-Suche startet
        st.session_state[config.STATE_EXPLICIT_RECOMMENDATIONS] = None
        # Rufe LLM auf, um Filter aus der Nutzeranfrage zu extrahieren (Funktion aus llm_utils.py)
        filter_dict, error_msg = get_filters_from_gemini(nlp_query, is_google_ai_really_configured)

        # Aktualisiere den Session State basierend auf dem LLM-Ergebnis
        if error_msg: # Fehler bei der LLM-Kommunikation
            update_llm_state(filters=None, justification=f"Fehler: {error_msg}", show_results=False, query=nlp_query, reset_suggestions=True)
        elif not filter_dict: # LLM hat keine Filter erkannt
            update_llm_state(filters=None, justification="Keine Filter aus deiner Anfrage erkannt. Zeige allgemeine Aktivitäten.", show_results=True, query=nlp_query, reset_suggestions=True)
        else: # LLM hat Filter erfolgreich extrahiert
            update_llm_state(filters=filter_dict, justification=None, show_results=True, query=nlp_query, reset_suggestions=True)
        st.rerun() # Lade App neu, um Ergebnisse zu verarbeiten und anzuzeigen
    elif not nlp_query: st.warning("Bitte beschreibe zuerst deine Wunsch-Aktivität.")
    # else: st.error("Keine Aktivitätsdaten zum Analysieren vorhanden.") # Sollte nicht passieren

# --- Logik: Anzuwendende Filter bestimmen ---
# Entscheide, ob die Filter aus der Sidebar oder die vom LLM extrahierten Filter verwendet werden sollen.
aktivitaetsart_filter: str = "Alle"
personen_filter: str = "Alle"
budget_filter: Optional[Union[float, int]] = None
consider_weather_filter: bool = consider_weather_sb # Wetter-Checkbox kommt immer aus Sidebar

# Prüfe, ob der KI-Modus aktiv ist UND ob Filter vom LLM extrahiert wurden
if st.session_state.get(config.STATE_SHOW_LLM_RESULTS) and isinstance(st.session_state.get(config.STATE_LLM_FILTERS), dict):
    # KI-Modus: Verwende die Filter aus dem Session State (die vom LLM kamen)
    extracted_filters = st.session_state.get(config.STATE_LLM_FILTERS, {})
    if extracted_filters:
        # Zeige die vom LLM extrahierten Filter in einem ausklappbaren Bereich an
        with extracted_filters_placeholder.expander("Von KI erkannte Filter (werden angewendet)", expanded=False):
            st.json(extracted_filters) # Zeige Filter als JSON an
    # Überschreibe die Filtervariablen mit den Werten aus dem LLM-Ergebnis (mit Fallbacks)
    aktivitaetsart_filter = extracted_filters.get('Art', ["Alle"])[0] if extracted_filters.get('Art') else "Alle" # Nimm erste Art aus Liste oder "Alle"
    personen_filter = extracted_filters.get('Personen_Anzahl_Kategorie', 'Alle')
    budget_filter = extracted_filters.get('Preis_Max') # Kann None sein
else:
    # Manueller Modus: Verwende die Filter direkt aus der Sidebar
    aktivitaetsart_filter = aktivitaetsart_sb
    personen_filter = personen_anzahl_sb
    budget_filter = budget_sb
    extracted_filters_placeholder.empty() # Stelle sicher, dass der Platzhalter leer ist

# --- Logik: Daten filtern (Basis + Wetter) ---
# Initialisiere leere Ergebnis-DataFrames, um Fehler zu vermeiden, falls Filterung fehlschlägt
columns_to_use = df_activities.columns if not df_activities.empty else config.EXPECTED_COLUMNS
base_filtered_df = pd.DataFrame(columns=columns_to_use) # Nach Basisfiltern
final_filtered_df = pd.DataFrame(columns=columns_to_use) # Nach Basis- UND Wetterfiltern
weather_data_map: Dict[int, Dict[str, Any]] = {} # Wetterinfos pro Original-ID
df_with_weather_cols = pd.DataFrame(columns=list(columns_to_use) + ['weather_note', 'location_temp', 'location_icon', 'location_desc']) # Basisgefiltert + Wetterspalten

# Führe Filterung nur aus, wenn Aktivitätsdaten vorhanden sind und ein Datum ausgewählt wurde
if not df_activities.empty and datum is not None:
    # 1. Wende Basisfilter an (Funktion aus logic.py)
    # print("INFO: Wende Basisfilter an...") # Debug
    base_filtered_df = apply_base_filters(
        df=df_activities, selected_date=datum,
        activity_type_filter=aktivitaetsart_filter,
        people_filter=personen_filter,
        budget_filter=budget_filter
    )
    # print(f"INFO: Nach Basisfilter: {len(base_filtered_df)} Aktivitäten.") # Debug

    # 2. Wende Wetterfilter an (Funktion aus logic.py)
    # Diese Funktion ruft intern die Wetter-API auf (via weather_utils) und reichert Daten an.
    weather_check_status = st.empty() # Platzhalter für "Prüfe Wetter..." Nachricht
    if config.OPENWEATHERMAP_API_CONFIGURED and not base_filtered_df.empty:
        weather_check_status.info("🌦️ Prüfe Wettervorhersagen für gefilterte Aktivitäten...")

    # Rufe die Funktion auf. Sie gibt 3 Ergebnisse zurück (siehe Docstring in logic.py).
    final_filtered_df, weather_data_map, df_with_weather_cols = apply_weather_filter(
        base_filtered_df=base_filtered_df,
        original_df=df_activities, # Wichtig für ID-Mapping
        selected_date=datum,
        consider_weather=consider_weather_filter, # Kommt von Sidebar-Checkbox
        api_key=config.OPENWEATHERMAP_API_KEY,
        api_configured=config.OPENWEATHERMAP_API_CONFIGURED
    )
    # Blende die "Prüfe Wetter..." Nachricht aus, wenn fertig
    if config.OPENWEATHERMAP_API_CONFIGURED and not base_filtered_df.empty:
        weather_check_status.empty()

# --- Logik: KI-Vorschläge generieren (falls im KI-Modus und noch nicht geschehen) ---
# Dies ist der zweite LLM-Aufruf: Er bekommt die gefilterten Kandidaten und soll die besten auswählen.
# Wird nur ausgeführt, wenn:
# 1. Der KI-Modus aktiv ist (show_llm_results = True)
# 2. Filter vom ersten LLM-Aufruf extrahiert wurden (llm_filters is not None)
# 3. Noch keine Vorschläge für diese Anfrage generiert wurden (llm_suggestion_ids is None)
if (st.session_state.get(config.STATE_SHOW_LLM_RESULTS) and
    st.session_state.get(config.STATE_LLM_FILTERS) is not None and
    st.session_state.get(config.STATE_LLM_SUGGESTION_IDS) is None):

    # Die Kandidaten für das LLM sind die basisgefilterten Aktivitäten, angereichert mit Wetterinfos.
    candidate_activities_df = df_with_weather_cols
    # print(f"INFO: Starte LLM Call 2 (Vorschläge) mit {len(candidate_activities_df)} Kandidaten.") # Debug

    if not candidate_activities_df.empty:
        # Bereite die Kandidatenliste für den Prompt vor (max. X Kandidaten, nur relevante Infos)
        candidate_limit = 15 # Begrenze Anzahl Kandidaten für den Prompt (API-Limits, Kosten)
        candidates_df_for_prompt = candidate_activities_df.head(candidate_limit)
        candidate_info_list = []
        # Definiere Spalten, die das LLM zur Auswahl sehen soll
        required_prompt_cols = [config.COL_ID, config.COL_NAME, config.COL_ART, config.COL_BESCHREIBUNG, config.COL_PREIS, config.COL_ORT, 'weather_note'] # Wetterhinweis hinzufügen?
        if all(col in candidates_df_for_prompt.columns for col in required_prompt_cols):
            # Erstelle für jeden Kandidaten einen beschreibenden String
            for idx, row in candidates_df_for_prompt.iterrows():
                preis_val = row.get(config.COL_PREIS); preis_str = f"{preis_val:.0f} CHF" if pd.notna(preis_val) and preis_val > 0 else "Gratis" if pd.notna(preis_val) else "N/A"
                desc_val = row.get(config.COL_BESCHREIBUNG); desc_short = str(desc_val)[:100] + "..." if pd.notna(desc_val) and len(str(desc_val)) > 100 else str(desc_val) if pd.notna(desc_val) else "N/A"
                weather_n = row.get('weather_note'); weather_str = f", Wetterhinweis: {weather_n}" if pd.notna(weather_n) else ""
                info = f"ID: {row.get(config.COL_ID)}, Name: {row.get(config.COL_NAME)}, Art: {row.get(config.COL_ART)}, Ort: {row.get(config.COL_ORT)}, Preis: {preis_str}, Info: {desc_short}{weather_str}"
                candidate_info_list.append(info)
            candidate_info_string = "\n".join(candidate_info_list)

            # Hole die ursprüngliche Nutzeranfrage aus dem State
            original_query = st.session_state.get(config.STATE_NLP_QUERY_SUBMITTED, 'deinem Wunsch')
            # Rufe das LLM auf, um Vorschläge und Begründung zu erhalten (Funktion aus llm_utils.py)
            sugg_ids, justif, err_msg = get_selection_and_justification(original_query, candidate_info_string, is_google_ai_really_configured)

            # Aktualisiere den Session State mit dem Ergebnis
            if err_msg: update_llm_state(suggestion_ids=[], justification=f"Fehler bei KI-Vorschlag: {err_msg}")
            elif sugg_ids is not None: update_llm_state(suggestion_ids=sugg_ids, justification=justif)
            else: update_llm_state(suggestion_ids=[], justification="Fehler: Unerwartete Antwort von KI (Vorschlag).")
        else:
            # Fehler: Wenn benötigte Spalten für den Prompt fehlen
            missing_cols_str = ", ".join([col for col in required_prompt_cols if col not in candidates_df_for_prompt.columns])
            # print(f"FEHLER: Fehlende Spalten für LLM-Prompt: {missing_cols_str}") # Debug
            update_llm_state(suggestion_ids=[], justification=f"Fehler: Benötigte Informationen für KI-Vorschlag fehlen ({missing_cols_str}).")
        st.rerun() # Lade App neu, um die generierten Vorschläge anzuzeigen
    else:
        # Fall: Nach Basis-/Wetterfilterung sind keine Kandidaten mehr übrig
        # print("INFO: Keine Kandidaten für LLM Call 2.") # Debug
        current_justif = st.session_state.get(config.STATE_LLM_JUSTIFICATION)
        # Setze Begründung nur, wenn nicht schon ein Fehler vom ersten LLM-Call da steht
        if not current_justif or "Fehler" not in current_justif:
            update_llm_state(justification="Keine passenden Aktivitäten für deine Anfrage gefunden, um KI-Vorschläge zu machen.")
        update_llm_state(suggestion_ids=[]) # Leere Vorschlagsliste setzen
        st.rerun() # Lade App neu, um die Meldung anzuzeigen

# --- UI: Hauptbereich (Karte, Wetter, Aktivitätenliste) ---

# 1. Layout für Karte und Wetterübersicht nebeneinander
col_map, col_weather = st.columns([2, 1], gap="large") # Karte bekommt 2/3, Wetter 1/3 der Breite

# Linke Spalte: Karte anzeigen
with col_map:
    # Entscheide, WELCHE Aktivitäten auf der Karte angezeigt werden sollen:
    df_map_display = pd.DataFrame() # Leerer DataFrame als Standard
    current_selection_id = st.session_state.get(config.STATE_SELECTED_ACTIVITY_INDEX) # ID der fokussierten Aktivität
    explicit_rec_ids = st.session_state.get(config.STATE_EXPLICIT_RECOMMENDATIONS) # IDs aus expliziter Profil-Anfrage

    # Priorität 1: Explizite Profil-Empfehlungen werden angezeigt
    if explicit_rec_ids is not None:
        if explicit_rec_ids and not df_activities.empty:
            valid_explicit_ids = [int(i) for i in explicit_rec_ids if isinstance(i, (int, float)) and pd.notna(i)]
            # Wähle die entsprechenden Aktivitäten aus dem Haupt-DataFrame aus
            df_map_display = df_activities[df_activities[config.COL_ID].isin(valid_explicit_ids)].copy()
            # Füge Wetterhinweise hinzu (aus der weather_data_map)
            if weather_data_map and config.COL_ID in df_activities.columns:
                 id_to_idx = {row[config.COL_ID]: index for index, row in df_activities.iterrows() if pd.notna(row[config.COL_ID])}
                 df_map_display['weather_note'] = df_map_display[config.COL_ID].map(lambda i: weather_data_map.get(id_to_idx.get(i, -1), {}).get('note'))

    # Priorität 2: KI-Vorschläge werden angezeigt (wenn keine expliziten da sind)
    elif st.session_state.get(config.STATE_SHOW_LLM_RESULTS) and isinstance(st.session_state.get(config.STATE_LLM_SUGGESTION_IDS), list):
        suggestion_ids = st.session_state.get(config.STATE_LLM_SUGGESTION_IDS, [])
        if suggestion_ids and not df_activities.empty:
            valid_suggestion_ids = [int(i) for i in suggestion_ids if isinstance(i, (int, float)) and pd.notna(i)]
            # Wähle entsprechende Aktivitäten aus Haupt-DataFrame
            df_map_display = df_activities[df_activities[config.COL_ID].isin(valid_suggestion_ids)].copy()
            # Füge Wetterhinweise hinzu
            if weather_data_map and config.COL_ID in df_activities.columns:
                 id_to_idx = {row[config.COL_ID]: index for index, row in df_activities.iterrows() if pd.notna(row[config.COL_ID])}
                 df_map_display['weather_note'] = df_map_display[config.COL_ID].map(lambda i: weather_data_map.get(id_to_idx.get(i, -1), {}).get('note'))

    # Priorität 3: Normal gefilterte Ergebnisse anzeigen (Standardfall)
    elif not final_filtered_df.empty:
        df_map_display = final_filtered_df # Enthält bereits Wetterhinweise aus apply_weather_filter

    # Rufe Funktion aus ui_components.py auf, um die Karte anzuzeigen
    display_map(df_map_display, selected_activity_id=current_selection_id)

# Rechte Spalte: Wetterübersicht für St. Gallen anzeigen
with col_weather:
    forecast_list_sg = None
    # Hole Wetterdaten für St. Gallen nur, wenn API konfiguriert und Datum gewählt
    if config.OPENWEATHERMAP_API_CONFIGURED and datum is not None:
        forecast_list_sg = get_weather_forecast_for_day(
            api_key=config.OPENWEATHERMAP_API_KEY,
            lat=config.ST_GALLEN_LAT, # Koordinaten aus config.py
            lon=config.ST_GALLEN_LON,
            target_date=datum
        )
    # Rufe Funktion aus ui_components.py auf, um die Wetterübersicht anzuzeigen
    display_weather_overview(
        location_name="St.Gallen",
        target_date=datum,
        forecast_list=forecast_list_sg,
        api_configured=config.OPENWEATHERMAP_API_CONFIGURED
    )

# 2. Aktivitätenliste anzeigen (unter Karte/Wetter)
st.markdown("---") # Trennlinie

# Hole relevante Status aus dem Session State
explicit_rec_ids = st.session_state.get(config.STATE_EXPLICIT_RECOMMENDATIONS)
llm_suggestion_ids = st.session_state.get(config.STATE_LLM_SUGGESTION_IDS)
show_llm_results = st.session_state.get(config.STATE_SHOW_LLM_RESULTS)

# Hilfsvariable, um zu prüfen, ob irgendeine Liste angezeigt wurde
list_content_shown = False

# --- Fall 1: Explizite Profil-Empfehlungen anzeigen ---
if explicit_rec_ids is not None:
    st.subheader("Passende Aktivitäten für dein Profil")
    list_content_shown = True
    if not df_activities.empty and explicit_rec_ids:
        valid_explicit_ids = [int(i) for i in explicit_rec_ids if isinstance(i, (int, float)) and pd.notna(i)]
        # Wähle die Daten für die Empfehlungen aus dem Haupt-DataFrame
        explicit_recs_df = df_activities[df_activities[config.COL_ID].isin(valid_explicit_ids)].copy()
        # Füge Wetterinformationen hinzu (aus der weather_data_map, die beim Filtern erstellt wurde)
        if weather_data_map and config.COL_ID in df_activities.columns:
            id_to_idx = {row[config.COL_ID]: index for index, row in df_activities.iterrows() if pd.notna(row[config.COL_ID])}
            weather_cols_keys = ['weather_note', 'location_temp', 'location_icon', 'location_desc']
            weather_map_keys = ['note', 'temp', 'icon', 'desc']
            for col_name, detail_key in zip(weather_cols_keys, weather_map_keys):
                explicit_recs_df[col_name] = explicit_recs_df[config.COL_ID].map(lambda i: weather_data_map.get(id_to_idx.get(i, -1), {}).get(detail_key))

        if not explicit_recs_df.empty:
             # Sortiere die Liste in der Reihenfolge, wie sie vom Recommender kam (optional)
             try:
                 explicit_recs_df[config.COL_ID] = pd.Categorical(explicit_recs_df[config.COL_ID], categories=valid_explicit_ids, ordered=True)
                 explicit_recs_df = explicit_recs_df.sort_values(config.COL_ID)
             except Exception as e: pass # Ignoriere Fehler beim Sortieren
             # Zeige jede Aktivität mit der Detail-Komponente an
             current_selection_id = st.session_state.get(config.STATE_SELECTED_ACTIVITY_INDEX)
             for index, row in explicit_recs_df.iterrows():
                 activity_id_display = row[config.COL_ID]
                 is_expanded = (activity_id_display == current_selection_id) # Expander öffnen, wenn auf Karte ausgewählt
                 display_activity_details(activity_row=row, activity_id=activity_id_display, is_expanded=is_expanded, openweathermap_api_configured=config.OPENWEATHERMAP_API_CONFIGURED, key_prefix="explicit") # Eindeutiger Key-Prefix
        else: st.info("Keine gültigen Aktivitäten für Profil-Empfehlungen gefunden.")
    elif not explicit_rec_ids: st.info("Keine Aktivitäten für dein Profil gefunden.")
    else: st.error("Keine Aktivitätsdaten zum Anzeigen der Empfehlungen.")
    # Button, um diese Liste wieder auszublenden und zur normalen Filteransicht zurückzukehren
    if st.button("Profil-Liste ausblenden", key="btn_hide_explicit"):
        st.session_state[config.STATE_EXPLICIT_RECOMMENDATIONS] = None # Setze State zurück
        st.rerun() # Lade neu
    st.markdown("---")

# --- Fall 2: KI-Vorschläge anzeigen (nur wenn keine expliziten Profil-Empfehlungen aktiv sind) ---
elif show_llm_results and isinstance(llm_suggestion_ids, list):
    # Zeige die Begründung des LLM an (im Platzhalter oben)
    current_justification = st.session_state.get(config.STATE_LLM_JUSTIFICATION)
    if current_justification:
        if "Fehler" in current_justification or "konnte nicht" in current_justification:
            justification_placeholder.error(f"KI-Analyse: {current_justification}")
        else:
            justification_placeholder.info(f"KI-Analyse: {current_justification}")
    else: justification_placeholder.empty() # Leere Platzhalter, wenn keine Begründung da

    # Zeige die Liste der vorgeschlagenen Aktivitäten an
    if llm_suggestion_ids:
        st.subheader("KI-Vorschläge ✨")
        list_content_shown = True
        if not df_activities.empty:
            valid_suggestion_ids = [int(i) for i in llm_suggestion_ids if isinstance(i, (int, float)) and pd.notna(i)]
            # Wähle Daten aus Haupt-DataFrame
            suggestions_df_list = df_activities[df_activities[config.COL_ID].isin(valid_suggestion_ids)].copy()
            # Füge Wetterinfos hinzu
            if weather_data_map and config.COL_ID in df_activities.columns:
                id_to_idx = {row[config.COL_ID]: index for index, row in df_activities.iterrows() if pd.notna(row[config.COL_ID])}
                weather_cols_keys = ['weather_note', 'location_temp', 'location_icon', 'location_desc']
                weather_map_keys = ['note', 'temp', 'icon', 'desc']
                for col_name, detail_key in zip(weather_cols_keys, weather_map_keys):
                     suggestions_df_list[col_name] = suggestions_df_list[config.COL_ID].map(lambda i: weather_data_map.get(id_to_idx.get(i, -1), {}).get(detail_key))

            if not suggestions_df_list.empty:
                # Sortiere in der Reihenfolge der LLM-Vorschläge (optional)
                try:
                    suggestions_df_list[config.COL_ID] = pd.Categorical(suggestions_df_list[config.COL_ID], categories=valid_suggestion_ids, ordered=True)
                    suggestions_df_list = suggestions_df_list.sort_values(config.COL_ID)
                except Exception as e: pass # Ignoriere Sortierfehler
                # Zeige Details für jede vorgeschlagene Aktivität
                current_selection_id = st.session_state.get(config.STATE_SELECTED_ACTIVITY_INDEX)
                for index, row in suggestions_df_list.iterrows():
                    is_expanded = (row[config.COL_ID] == current_selection_id)
                    display_activity_details(activity_row=row, activity_id=row[config.COL_ID], is_expanded=is_expanded, openweathermap_api_configured=config.OPENWEATHERMAP_API_CONFIGURED, key_prefix="llm") # Eindeutiger Key-Prefix
            else: st.info("Keine gültigen KI-Vorschläge gefunden (nach Validierung).")
        else: st.error("Keine Aktivitätsdaten zum Anzeigen der Vorschläge.")
    # else: Wenn llm_suggestion_ids leer ist, wird nur die Begründung (oben) angezeigt.
    st.markdown("---")

# --- Fall 3: Normal gefilterte Liste anzeigen (Standardfall, wenn weder explizit noch KI aktiv) ---
else:
    justification_placeholder.empty() # Sicherstellen, dass LLM-Begründung leer ist
    if not final_filtered_df.empty:
        st.subheader(f"Gefilterte Aktivitäten ({len(final_filtered_df)})")
        list_content_shown = True
        # Begrenze Anzahl angezeigter Items zur Übersichtlichkeit (optional)
        MAX_LIST_ITEMS = 50
        display_df_limited = final_filtered_df.head(MAX_LIST_ITEMS)
        # Zeige Details für jede gefilterte Aktivität
        current_selection_id = st.session_state.get(config.STATE_SELECTED_ACTIVITY_INDEX)
        for index, row in display_df_limited.iterrows():
            is_expanded = (row[config.COL_ID] == current_selection_id)
            display_activity_details(activity_row=row, activity_id=row[config.COL_ID], is_expanded=is_expanded, openweathermap_api_configured=config.OPENWEATHERMAP_API_CONFIGURED, key_prefix="filter") # Eindeutiger Key-Prefix
        # Hinweis, wenn nicht alle Ergebnisse angezeigt werden
        if len(final_filtered_df) > MAX_LIST_ITEMS:
            st.info(f"Hinweis: Nur die ersten {MAX_LIST_ITEMS} von {len(final_filtered_df)} Aktivitäten angezeigt.")

# --- Fallback-Meldung, wenn gar keine Aktivitäten angezeigt werden (in keinem Modus) ---
if not list_content_shown:
    # Zeige nur Meldung, wenn nicht gerade der Fall eintrat, dass KI-Modus aktiv war,
    # aber das LLM bewusst eine leere Vorschlagsliste zurückgab (dieser Fall wird durch die
    # LLM-Begründung oben abgedeckt).
    if not (show_llm_results and isinstance(llm_suggestion_ids, list) and not llm_suggestion_ids):
         st.info("Keine Aktivitäten für die gewählten Filter und das Datum gefunden.")

# print("--- App-Durchlauf beendet ---") # Debug
