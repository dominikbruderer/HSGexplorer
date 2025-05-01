# app.py
"""
Hauptanwendung für HSGexplorer - Ein intelligenter Aktivitätsfinder für St. Gallen und Umgebung.

Dieses Skript initialisiert und startet die Streamlit-Webanwendung HSGexplorer.
Es dient als zentraler Orchestrator und löst folgendes Problem (Requirement 1):
Nutzern dabei zu helfen, passende Freizeitaktivitäten in St. Gallen und Umgebung zu finden,
basierend auf manuellen Filtern, Wetterdaten und optionalen, KI-gestützten Vorschlägen
sowie personalisierten Empfehlungen basierend auf Nutzerbewertungen.

Hauptaufgaben:
- Seitenkonfiguration und Initialisierung des Session State.
- Laden und Vorverarbeiten der Aktivitätsdaten (Requirement 2, teilweise).
- Berechnung von Features für das ML-Empfehlungssystem (Requirement 5, teilweise).
- Aufbau der Benutzeroberfläche (Titel, NLP-Eingabe, Sidebar, Karten-/Wetterbereich,
  Aktivitätenliste, Personalisierungs-Expander) (Requirement 4).
- Verarbeitung von Benutzereingaben (manuelle Filter, NLP-Query, Likes/Dislikes).
- Interaktion mit dem LLM (Google Gemini) zur Filterextraktion und Vorschlagsgenerierung (Requirement 2 & 5).
- Anwendung der Filterlogik (Basis- und Wetterfilter).
- Abruf von Wetterdaten via API (Requirement 2).
- Verwaltung des Anwendungszustands (LLM vs. manueller Modus, etc.).
- Darstellung der Ergebnisse und Visualisierungen (Karte, Wetter, Präferenzen) (Requirement 3).
- Implementierung personalisierter Empfehlungen (Requirement 5).
"""

# --- Importe ---
import streamlit as st
import pandas as pd
import datetime
import os # Wird indirekt von Modulen verwendet
import traceback # Für detailliertere Fehlermeldungen
import random # Für Exploration und Shuffle bei Empfehlungen
# KORREKTUR: Fehlende Importe für Type Hints hinzugefügt
from typing import List, Dict, Any, Optional, Set, Tuple, Union

# --- Eigene Module importieren ---
# Versuche, alle benötigten Module zu importieren und fange Fehler ab
try:
    import config # Globale Konfigurationen und Konstanten
    from data_utils import load_data # Daten laden und bereinigen
    from weather_utils import get_weather_forecast_for_day # Wetter-API Abruf
    from llm_utils import get_filters_from_gemini, get_selection_and_justification, update_llm_state # LLM Interaktion
    from logic import apply_base_filters, apply_weather_filter # Filterlogik
    from ui_components import ( # UI-Elemente
        display_sidebar, display_map, display_weather_overview,
        display_activity_details, display_recommendation_card,
        display_preference_visualization
    )
    # Requirement 5: Import der ML-Funktionen für Empfehlungen
    from recommender import (
        preprocess_features, # Feature Engineering
        # calculate_similarity_matrix, # ENTFERNT, da nicht mehr benötigt
        # get_recommendations, # ENTFERNT, da nicht mehr benötigt
        calculate_user_profile, # Nutzerprofilierung
        get_profile_recommendations, # Profil-basierte Empfehlungen
        calculate_preference_scores, # Helfer für Visualisierung
        generate_profile_label, # Helfer für Visualisierung
        calculate_top_target_groups, # Helfer für Visualisierung
        get_liked_prices # Helfer für Visualisierung
    )
except ImportError as e:
    st.error(f"Fataler Fehler beim Importieren von Modulen: '{e}'. Stellen Sie sicher, dass alle .py-Dateien im selben Verzeichnis liegen und alle Abhängigkeiten installiert sind.")
    st.stop() # App kann ohne Module nicht sinnvoll starten

# --- Globale Konfiguration und Initialisierung ---

# Seitenkonfiguration (Titel, Layout, Icon)
st.set_page_config(page_title="HSGexplorer", layout="wide", page_icon="🗺️")

# Initialisiere Session State mit Defaults aus config.py, falls noch nicht vorhanden
# Der Session State speichert den Zustand der App zwischen Interaktionen
# print("DEBUG: Initializing session state...")
for key, default_value in config.DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
# print("DEBUG: Session state initialized.")

# Konfiguriere Google AI sicher (nur einmal pro Session, wenn Key vorhanden)
# Requirement 2 & 5: Vorbereitung für LLM-API Nutzung
google_api_error_handled = False
if config.GOOGLE_API_CONFIGURED and not st.session_state.get(config.STATE_GOOGLE_AI_CONFIGURED, False):
    try:
        import google.generativeai as genai
        genai.configure(api_key=config.GOOGLE_API_KEY)
        st.session_state[config.STATE_GOOGLE_AI_CONFIGURED] = True
        print("INFO: Google AI API erfolgreich konfiguriert.")
    except ImportError:
        st.error("Fehler: Das Paket 'google-generativeai' fehlt. Installation: pip install google-generativeai")
        st.session_state[config.STATE_GOOGLE_AI_CONFIGURED] = False
        google_api_error_handled = True
    except Exception as e:
        st.error(f"Fehler bei der Konfiguration der Google AI API: {e}")
        st.session_state[config.STATE_GOOGLE_AI_CONFIGURED] = False
        google_api_error_handled = True
elif not config.GOOGLE_API_CONFIGURED and not st.session_state.get(config.STATE_GOOGLE_KEY_WARNING_SHOWN, False):
    # Logik, falls Key von Anfang an fehlt
    st.session_state[config.STATE_GOOGLE_AI_CONFIGURED] = False


# --- API Key Warnungen (einmal pro Session anzeigen) ---
is_google_ai_really_configured = st.session_state.get(config.STATE_GOOGLE_AI_CONFIGURED, False)
# Zeige Warnung, wenn Konfiguration nicht erfolgreich ODER Key fehlt UND Warnung noch nicht gezeigt wurde
if not is_google_ai_really_configured and not google_api_error_handled and not st.session_state.get(config.STATE_GOOGLE_KEY_WARNING_SHOWN, False):
    st.warning("Google AI API Key nicht korrekt konfiguriert oder Konfiguration fehlgeschlagen. KI-Funktionen sind deaktiviert.", icon="🔑")
    st.session_state[config.STATE_GOOGLE_KEY_WARNING_SHOWN] = True # Merken, dass Warnung gezeigt wurde

if not config.OPENWEATHERMAP_API_CONFIGURED and not st.session_state.get(config.STATE_OWM_KEY_WARNING_SHOWN, False):
    st.warning("OpenWeatherMap API Key nicht korrekt konfiguriert. Wetterfunktionen sind eingeschränkt.", icon="🌦️")
    st.session_state[config.STATE_OWM_KEY_WARNING_SHOWN] = True

# --- Daten laden und ML-Vorbereitung ---

# Requirement 2: Laden der Aktivitätsdaten aus CSV
CSV_PATH_NEU = "aktivitaeten_neu.csv" # Pfad zur Haupt-Datenquelle
df_activities = load_data(CSV_PATH_NEU)

# Requirement 5: Berechnung der Feature-Matrix für ML-Empfehlungen (im State speichern)
# Diese Matrix wird für die Nutzerprofilierung und die profilbasierten Empfehlungen benötigt.
if config.STATE_FEATURES_MATRIX not in st.session_state or st.session_state[config.STATE_FEATURES_MATRIX] is None:
    if not df_activities.empty:
        print("INFO: Berechne Feature-Matrix für Empfehlungen...")
        with st.spinner('Analysiere Aktivitäten für Empfehlungen...'): # Zeige Spinner während Berechnung
            _, features = preprocess_features(df_activities)
        st.session_state[config.STATE_FEATURES_MATRIX] = features
        if features is None or features.shape[1] == 0 :
            print("WARNUNG: Keine Features für Empfehlungen extrahiert.")
            st.warning("Konnte keine Features für Empfehlungen extrahieren.", icon="⚠️")
        else:
             print(f"INFO: Feature-Matrix ({features.shape}) berechnet und im State gespeichert.")
    else:
        # Stelle sicher, dass der State auch bei leeren Daten initialisiert wird
        st.session_state[config.STATE_FEATURES_MATRIX] = None
        print("INFO: Keine Aktivitätsdaten, Feature-Matrix-Berechnung übersprungen.")

# --- Die Berechnung der Similarity Matrix wurde entfernt ---

# Prüfe, ob Daten erfolgreich geladen wurden
if df_activities.empty:
    st.error("Fataler Fehler: Keine Aktivitätsdaten gefunden oder geladen. Die App kann nicht richtig funktionieren.")
    # Optional: st.stop() um die App hier anzuhalten

# Konstante für die Mindestanzahl an Likes für die Profilanzeige
MIN_LIKES_FOR_PROFILE = 5 # Schwelle anpassen nach Bedarf

# --- Callback Funktion für Like/Dislike Buttons ---
# Requirement 4 & 5: Verarbeitet Nutzerinteraktion (Bewertung) und stößt ML-Neuberechnung an
def update_recommendations(clicked_activity_id: int, rating: int) -> None:
    """
    Aktualisiert Bewertungen, Nutzerprofil, holt neue Empfehlungen und generiert Profil-Label.
    Wird aufgerufen, wenn ein Like-/Dislike-Button geklickt wird.
    """
    # print(f"Callback ausgelöst: ID={clicked_activity_id}, Rating={rating}")
    # Zugriff auf global geladene Daten und im State gespeicherte ML-Daten
    global df_activities

    if clicked_activity_id is None or clicked_activity_id == -1: # ID -1 kommt von ungültiger Konvertierung
        print("WARNUNG (Callback): Ungültige activity_id empfangen.")
        return

    # 1. Listen im State aktualisieren (Likes/Dislikes)
    liked_ids_list = st.session_state.get(config.STATE_LIKED_IDS, [])
    disliked_ids_list = st.session_state.get(config.STATE_DISLIKED_IDS, [])
    # Stelle sicher, dass ID als int behandelt wird
    try: clicked_activity_id = int(clicked_activity_id)
    except (ValueError, TypeError): print(f"WARNUNG (Callback): Konnte ID '{clicked_activity_id}' nicht in int umwandeln."); return

    if rating == 1: # Like
        if clicked_activity_id not in liked_ids_list: liked_ids_list.append(clicked_activity_id)
        if clicked_activity_id in disliked_ids_list: disliked_ids_list.remove(clicked_activity_id)
    elif rating == -1: # Dislike
        if clicked_activity_id not in disliked_ids_list: disliked_ids_list.append(clicked_activity_id)
        if clicked_activity_id in liked_ids_list: liked_ids_list.remove(clicked_activity_id)
    st.session_state[config.STATE_LIKED_IDS] = liked_ids_list
    st.session_state[config.STATE_DISLIKED_IDS] = disliked_ids_list
    # print(f"DEBUG (Callback): State aktualisiert - Likes: {st.session_state[config.STATE_LIKED_IDS]}, Dislikes: {st.session_state[config.STATE_DISLIKED_IDS]}")

    # 2. Profil aktualisieren und neue Empfehlungen holen
    features_matrix = st.session_state.get(config.STATE_FEATURES_MATRIX)
    if features_matrix is not None and not df_activities.empty:
        # Requirement 5: Nutzerprofil berechnen
        user_profile = calculate_user_profile(
            liked_ids=st.session_state[config.STATE_LIKED_IDS],
            disliked_ids=st.session_state[config.STATE_DISLIKED_IDS],
            features_matrix=features_matrix,
            df=df_activities
        )
        st.session_state[config.STATE_USER_PROFILE] = user_profile

        if user_profile is not None:
            # Requirement 5: Profil-basierte Empfehlungen holen
            rated_ids: Set[int] = set(st.session_state[config.STATE_LIKED_IDS]) | set(st.session_state[config.STATE_DISLIKED_IDS])
            new_recommendation_ids = get_profile_recommendations(
                user_profile=user_profile,
                features_matrix=features_matrix,
                df=df_activities,
                rated_ids=rated_ids,
                n=5 # Anzahl der Vorschläge, die im Expander rotieren
            )
            # Store the new list of IDs to be shown
            st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS] = new_recommendation_ids
            # print(f"DEBUG (Callback): Neue Empfehlungs-IDs im State: {st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS]}")
        else:
            # Kein Profil -> keine Empfehlungen
            st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS] = []
            # print("DEBUG (Callback): Kein User Profile berechnet, Empfehlungsliste geleert.")

        # 3. Profil-Label und Visualisierungsdaten (neu) berechnen
        # Requirement 5: Nutzung von Likes zur Profilbeschreibung
        pref_scores = calculate_preference_scores(st.session_state[config.STATE_LIKED_IDS], df_activities)
        profile_label = generate_profile_label(pref_scores)
        st.session_state[config.STATE_USER_PROFILE_LABEL] = profile_label
        # print(f"DEBUG (Callback): Profil-Label im State: {st.session_state[config.STATE_USER_PROFILE_LABEL]}")

    else:
        # Fallback, wenn keine Features/Daten vorhanden
        st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS] = []
        st.session_state[config.STATE_USER_PROFILE_LABEL] = None
        # print("DEBUG (Callback): Keine Feature-Matrix oder Daten für Profil/Empfehlungen/Label.")

    # WICHTIG: KEIN st.rerun() im Callback! Streamlit führt es nach Button-Klick automatisch aus.

# --- UI: Statischer Teil (Titel, Einleitung, NLP-Eingabe) ---
# Requirement 1: Titel verdeutlicht den Zweck
st.title("HSGexplorer")
st.write("Dein intelligenter Assistent für Aktivitäten in St. Gallen und Umgebung.")

# Requirement 4: NLP-Eingabe als Interaktionselement
st.subheader("Was möchtest du unternehmen?")
nlp_query = st.text_input("Beschreibe deine Wunsch-Aktivität:",
                        placeholder="z.B. 'Wandern mit Aussicht', 'Museum für Familien', 'Günstiges Date bei Regen'",
                        label_visibility="collapsed",
                        key="nlp_query_input")
# Button für KI-Suche, deaktiviert wenn KI nicht konfiguriert oder keine Daten
nlp_button_pressed = st.button("KI-Vorschläge finden", key="nlp_button_main", type="primary",
                            disabled=not is_google_ai_really_configured or df_activities.empty)

# Platzhalter für dynamische Inhalte (LLM Begründung, extrahierte Filter)
justification_placeholder = st.empty()
extracted_filters_placeholder = st.empty()

# --- Logik: Initiale Empfehlungen laden (nur wenn Liste im State leer ist) ---
# Füllt die Empfehlungsliste beim ersten Laden oder nach Reset mit zufälligen, unbewerteten Aktivitäten
if not st.session_state.get(config.STATE_RECOMMENDATIONS_TO_SHOW_IDS):
    # print("DEBUG: Keine Vorschläge im State, lade initiale Empfehlungen...")
    if not df_activities.empty:
        rated_ids = set(st.session_state.get(config.STATE_LIKED_IDS, [])) | set(st.session_state.get(config.STATE_DISLIKED_IDS, []))
        initial_candidates = df_activities[config.COL_ID].dropna().unique().tolist()
        # Filtere gültige Integer-IDs und schließe bewertete aus
        valid_unrated_ids = [int(id_) for id_ in initial_candidates if isinstance(id_, (int, float)) and pd.notna(id_) and int(id_) not in rated_ids and int(id_) != -1]
        if valid_unrated_ids: random.shuffle(valid_unrated_ids) # Mische die unbewerteten Kandidaten
        st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS] = valid_unrated_ids[:5] # Nimm die ersten 5 für die Anzeige
    else:
        st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS] = []
    # print(f"DEBUG: Initiale/aktualisierte Empfehlungs-IDs geladen: {st.session_state.get(config.STATE_RECOMMENDATIONS_TO_SHOW_IDS)}")


# --- UI: Personalisierte Vorschläge & Präferenzen (im Expander) ---
st.markdown("---")
# Requirement 4 & 5: Interaktive Empfehlungs- und Lernkomponente
with st.expander("✨ Personalisierte Vorschläge (ausprobieren!)", expanded=True):
    recommendation_ids_for_card = st.session_state.get(config.STATE_RECOMMENDATIONS_TO_SHOW_IDS, [])
    col_card, col_viz = st.columns([1, 1]) # Zwei Spalten Layout

    # --- Linke Spalte: Vorschlagskarte ---
    with col_card:
        st.markdown("**Aktueller Vorschlag:**")
        if not recommendation_ids_for_card:
             st.caption("Bewerte Aktivitäten, um hier Vorschläge zu sehen, oder es gibt keine weiteren.")
        elif df_activities.empty:
             st.warning("Keine Aktivitätsdaten zum Anzeigen des Vorschlags.")
        else:
            # Zeige immer die erste Aktivität aus der Liste an
            single_suggestion_id = recommendation_ids_for_card[0]
            try:
                activity_id_int = int(single_suggestion_id)
                # Finde die Datenzeile für diese Aktivität
                card_row_df = df_activities[df_activities[config.COL_ID] == activity_id_int]
                if not card_row_df.empty:
                     card_row = card_row_df.iloc[0]
                     # Zeige die Empfehlungskarte an, übergebe Callbacks
                     display_recommendation_card(
                         activity_row=card_row,
                         card_key_suffix=f"single_rec_{activity_id_int}",
                         # WICHTIG: Die Callback-Funktion selbst übergeben, nicht das Ergebnis des Aufrufs!
                         # Die Argumente werden dann durch den `args`-Parameter im Button übergeben.
                         on_like_callback=lambda act_id=activity_id_int: update_recommendations(act_id, 1),
                         on_dislike_callback=lambda act_id=activity_id_int: update_recommendations(act_id, -1)
                         # ALTERNATIVE (wenn update_recommendations nur die ID bräuchte):
                         # on_like_callback=update_recommendations, # args=(activity_id_int,) im Button
                     )
                else:
                     # Wenn Aktivität nicht gefunden wird (sollte nicht passieren, wenn Liste aktuell ist)
                     st.warning(f"Vorgeschlagene Aktivität mit ID {activity_id_int} nicht in Daten gefunden.")
                     # Entferne die ungültige ID aus der Liste und lade neu, um die nächste anzuzeigen
                     if activity_id_int in st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS]:
                          st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS].pop(0)
                          st.rerun() # Lade UI neu, um nächste Karte anzuzeigen
            except Exception as e:
                 st.error(f"Fehler bei Vorschlagskarte für ID '{single_suggestion_id}': {e}")
                 # traceback.print_exc()

    # --- Rechte Spalte: Präferenzvisualisierung ---
    # Requirement 3: Visualisierung der gelernten Präferenzen
    with col_viz:
        st.markdown("**Deine Präferenzen (gelernt):**")
        current_likes_count = len(st.session_state.get(config.STATE_LIKED_IDS, []))

        # Zeige Visualisierung nur, wenn genügend Likes vorhanden sind
        if current_likes_count >= MIN_LIKES_FOR_PROFILE:
            # Hole Daten für Visualisierung aus State oder berechne sie
            current_profile_label = st.session_state.get(config.STATE_USER_PROFILE_LABEL)
            liked_ids_for_viz = st.session_state.get(config.STATE_LIKED_IDS, [])
            pref_scores_art = None; top_groups = None; price_list = None
            if not df_activities.empty:
                pref_scores_art = calculate_preference_scores(liked_ids=liked_ids_for_viz, df_all_activities=df_activities)
                top_groups = calculate_top_target_groups(liked_ids=liked_ids_for_viz, df_all_activities=df_activities, top_n=5)
                price_list = get_liked_prices(liked_ids=liked_ids_for_viz, df_all_activities=df_activities, include_free=False) # Hier z.B. ohne kostenlose

            # Zeige die Visualisierungskomponente an
            display_preference_visualization(
                profile_label=current_profile_label,
                preference_scores_art=pref_scores_art,
                top_target_groups=top_groups,
                liked_prices_list=price_list
            )

            # Button, um explizit Empfehlungen für das Profil anzuzeigen
            # Requirement 4: Weitere Nutzerinteraktion
            show_profile_recommendations = st.button("Zeige passende Aktivitäten für mein Profil", key="btn_show_profile_rec")
            if show_profile_recommendations:
                # print("DEBUG: Button 'Zeige passende Aktivitäten für mein Profil' geklickt.")
                current_user_profile = st.session_state.get(config.STATE_USER_PROFILE)
                features_matrix = st.session_state.get(config.STATE_FEATURES_MATRIX)
                if current_user_profile is not None and features_matrix is not None and not df_activities.empty:
                    rated_ids = set(st.session_state.get(config.STATE_LIKED_IDS, [])) | set(st.session_state.get(config.STATE_DISLIKED_IDS, []))
                    with st.spinner('Suche passende Aktivitäten...'): # Spinner anzeigen
                         explicit_ids = get_profile_recommendations(
                             user_profile=current_user_profile, features_matrix=features_matrix,
                             df=df_activities, rated_ids=rated_ids, n=10, exploration_rate=0 # Keine Exploration hier
                         )
                    # Speichere die explizite Liste im State und löse Neuladen aus
                    st.session_state[config.STATE_EXPLICIT_RECOMMENDATIONS] = explicit_ids
                    print(f"INFO: Explizite Empfehlungs-IDs im State gespeichert: {explicit_ids}")
                    st.rerun()
                elif current_user_profile is None: st.warning("Bitte bewerte zuerst einige Aktivitäten, um ein Profil zu erstellen.")
                else: st.error("Fehler: Benötigte Profildaten nicht verfügbar.")

        # Hinweis, wenn noch nicht genug Likes für Profilanzeige vorhanden sind
        elif current_likes_count > 0:
            st.caption(f"Bewerte noch {MIN_LIKES_FOR_PROFILE - current_likes_count} weitere Aktivitäten positiv 👍, um dein detailliertes Profil zu sehen!")
        else:
            st.caption("Bewerte einige Aktivitäten 👍, um hier deine Präferenzen zu sehen!")

        # Reset-Button für Personalisierung
        if st.button("Personalisierung zurücksetzen", key="btn_reset_prefs"):
             st.session_state[config.STATE_LIKED_IDS] = []
             st.session_state[config.STATE_DISLIKED_IDS] = []
             st.session_state[config.STATE_USER_PROFILE] = None
             st.session_state[config.STATE_USER_PROFILE_LABEL] = None
             st.session_state[config.STATE_RECOMMENDATIONS_TO_SHOW_IDS] = [] # Leere auch die Vorschlagsliste
             st.session_state[config.STATE_EXPLICIT_RECOMMENDATIONS] = None # Leere explizite Liste
             print("INFO: Personalisierung zurückgesetzt.")
             st.rerun() # Lade App neu, um Reset anzuzeigen

# --- UI: Sidebar ---
# Requirement 4: Manuelle Filter als Interaktionselement
today = datetime.date.today()
datum, consider_weather_sb, aktivitaetsart_sb, personen_anzahl_sb, budget_sb, reset_llm_pressed = display_sidebar(
    df_activities, today, config.OPENWEATHERMAP_API_CONFIGURED
)

# --- Logik: Zustandsverwaltung (LLM vs. Manuell) & LLM-Filterextraktion ---
# Requirement 5: Nutzung des LLM für Filterextraktion

# Wenn "Manuelle Filter verwenden" geklickt wurde
if reset_llm_pressed:
    print("INFO: Manuelle Filter aktiviert, LLM State wird zurückgesetzt.")
    update_llm_state(show_results=False, query=None, filters=None, suggestion_ids=None, justification=None)
    st.session_state[config.STATE_EXPLICIT_RECOMMENDATIONS] = None # Auch explizite Liste zurücksetzen
    # Leere die dynamischen Platzhalter
    extracted_filters_placeholder.empty()
    justification_placeholder.empty()
    st.rerun() # Neu laden, um manuellen Modus zu aktivieren

# Wenn der NLP-Button gedrückt wurde
if nlp_button_pressed:
    if not df_activities.empty and nlp_query:
        print(f"INFO: NLP Button gedrückt, starte Filterextraktion für: '{nlp_query}'")
        # Setze explizite Liste zurück, wenn neue NLP-Suche gestartet wird
        st.session_state[config.STATE_EXPLICIT_RECOMMENDATIONS] = None
        # Rufe LLM auf, um Filter zu extrahieren
        filter_dict, error_msg = get_filters_from_gemini(nlp_query, is_google_ai_really_configured)
        # Aktualisiere Session State basierend auf LLM-Ergebnis
        if error_msg: update_llm_state(filters=None, justification=f"Fehler: {error_msg}", show_results=False, query=nlp_query, reset_suggestions=True)
        elif not filter_dict: update_llm_state(filters=None, justification="Keine Filter aus deiner Anfrage erkannt.", show_results=True, query=nlp_query, reset_suggestions=True)
        else: update_llm_state(filters=filter_dict, justification=None, show_results=True, query=nlp_query, reset_suggestions=True)
        st.rerun() # Neu laden, um LLM-Ergebnisse zu verarbeiten
    elif not nlp_query: st.warning("Bitte beschreibe zuerst deine Wunsch-Aktivität.")
    else: st.error("Keine Aktivitätsdaten zum Analysieren vorhanden.") # Sollte nicht passieren, wenn App startet

# --- Logik: Anzuwendende Filter bestimmen ---
# Entscheide, ob LLM-Filter oder Sidebar-Filter verwendet werden sollen
aktivitaetsart_filter: str = "Alle"
personen_filter: str = "Alle"
# KORREKTUR: Type Hint hier anpassen
budget_filter: Optional[Union[float, int]] = None
consider_weather_filter: bool = consider_weather_sb # Wetter-Checkbox aus Sidebar

# Wenn LLM-Ergebnisse angezeigt werden sollen und Filter extrahiert wurden:
if st.session_state.get(config.STATE_SHOW_LLM_RESULTS) and isinstance(st.session_state.get(config.STATE_LLM_FILTERS), dict):
    extracted_filters = st.session_state.get(config.STATE_LLM_FILTERS, {})
    if extracted_filters:
        # Zeige die vom LLM extrahierten Filter an (im Expander)
        with extracted_filters_placeholder.expander("Von KI extrahierte Filter (aktiv)", expanded=False):
            st.json(extracted_filters)
    # Überschreibe Filterwerte mit denen vom LLM (mit Fallbacks)
    aktivitaetsart_filter = extracted_filters.get('Art', ["Alle"])[0] if extracted_filters.get('Art') else "Alle"
    personen_filter = extracted_filters.get('Personen_Anzahl_Kategorie', 'Alle')
    budget_filter = extracted_filters.get('Preis_Max')
else:
    # Sonst: Verwende die manuellen Filter aus der Sidebar
    aktivitaetsart_filter = aktivitaetsart_sb
    personen_filter = personen_anzahl_sb
    budget_filter = budget_sb
    extracted_filters_placeholder.empty() # Stelle sicher, dass der Platzhalter leer ist

# --- Logik: Daten filtern (Basis + Wetter) ---
# Initialisiere leere DataFrames für die Ergebnisse
columns_to_use = df_activities.columns if not df_activities.empty else config.EXPECTED_COLUMNS
base_filtered_df = pd.DataFrame(columns=columns_to_use)
final_filtered_df = pd.DataFrame(columns=columns_to_use)
weather_data_map: Dict[int, Dict[str, Any]] = {}
df_with_weather_cols = pd.DataFrame(columns=list(columns_to_use) + ['weather_note', 'location_temp', 'location_icon', 'location_desc'])

# Führe Filterung nur aus, wenn Daten und Datum vorhanden sind
if not df_activities.empty and datum is not None:
    # print("INFO: Wende Basisfilter an...")
    base_filtered_df = apply_base_filters(
        df=df_activities, selected_date=datum,
        activity_type_filter=aktivitaetsart_filter,
        people_filter=personen_filter,
        budget_filter=budget_filter
    )
    # print(f"INFO: Nach Basisfilter: {len(base_filtered_df)} Aktivitäten.")

    # Wende Wetterfilter an (dieser ruft intern die Wetter-API auf)
    # Requirement 2: Nutzung der Wetter-API via logic/weather_utils
    weather_check_status = st.empty()
    if config.OPENWEATHERMAP_API_CONFIGURED and not base_filtered_df.empty:
        weather_check_status.info("🌦️ Prüfe Wettervorhersagen...")

    final_filtered_df, weather_data_map, df_with_weather_cols = apply_weather_filter(
        base_filtered_df=base_filtered_df,
        original_df=df_activities,
        selected_date=datum,
        consider_weather=consider_weather_filter,
        api_key=config.OPENWEATHERMAP_API_KEY,
        api_configured=config.OPENWEATHERMAP_API_CONFIGURED
    )
    if config.OPENWEATHERMAP_API_CONFIGURED and not base_filtered_df.empty:
        weather_check_status.empty()

# --- Logik: LLM Call 2 (Vorschläge generieren, wenn im LLM-Modus und noch keine Vorschläge da) ---
# Requirement 5: Nutzung des LLM zur Auswahl und Begründung von Vorschlägen
if (st.session_state.get(config.STATE_SHOW_LLM_RESULTS) and
    st.session_state.get(config.STATE_LLM_FILTERS) is not None and
    st.session_state.get(config.STATE_LLM_SUGGESTION_IDS) is None):
    candidate_activities_df = df_with_weather_cols
    # print(f"INFO: Starte LLM Call 2 (Vorschläge) mit {len(candidate_activities_df)} Kandidaten.")

    if not candidate_activities_df.empty:
        candidate_limit = 15
        candidates_df_for_prompt = candidate_activities_df.head(candidate_limit)
        candidate_info_list = []
        required_prompt_cols = [config.COL_ID, config.COL_NAME, config.COL_ART, config.COL_BESCHREIBUNG, config.COL_PREIS, config.COL_ORT]
        if all(col in candidates_df_for_prompt.columns for col in required_prompt_cols):
            for idx, row in candidates_df_for_prompt.iterrows():
                preis_val = row.get(config.COL_PREIS); preis_str = f"{preis_val:.0f} CHF" if pd.notna(preis_val) and preis_val > 0 else "Gratis" if pd.notna(preis_val) else "N/A"
                desc_val = row.get(config.COL_BESCHREIBUNG); desc_short = str(desc_val)[:100] + "..." if pd.notna(desc_val) and len(str(desc_val)) > 100 else str(desc_val) if pd.notna(desc_val) else "N/A"
                info = f"ID: {row.get(config.COL_ID)}, Name: {row.get(config.COL_NAME)}, Art: {row.get(config.COL_ART)}, Ort: {row.get(config.COL_ORT)}, Preis: {preis_str}, Info: {desc_short}"
                candidate_info_list.append(info)
            candidate_info_string = "\n".join(candidate_info_list)
            original_query = st.session_state.get(config.STATE_NLP_QUERY_SUBMITTED, 'deinem Wunsch')
            sugg_ids, justif, err_msg = get_selection_and_justification(original_query, candidate_info_string, is_google_ai_really_configured)
            if err_msg: update_llm_state(suggestion_ids=[], justification=f"Fehler: {err_msg}")
            elif sugg_ids is not None: update_llm_state(suggestion_ids=sugg_ids, justification=justif)
            else: update_llm_state(suggestion_ids=[], justification="Fehler: Unerwartete Antwort von KI (Vorschlag).")
        else:
            missing_cols_str = ", ".join([col for col in required_prompt_cols if col not in candidates_df_for_prompt.columns])
            print(f"FEHLER: Fehlende Spalten für LLM-Prompt: {missing_cols_str}")
            update_llm_state(suggestion_ids=[], justification=f"Fehler: Benötigte Informationen für KI-Vorschlag fehlen ({missing_cols_str}).")
        st.rerun()
    else:
        # print("INFO: Keine Kandidaten für LLM Call 2.")
        current_justif = st.session_state.get(config.STATE_LLM_JUSTIFICATION)
        if not current_justif or "Fehler" not in current_justif:
            update_llm_state(justification="Keine passenden Aktivitäten für deine Anfrage gefunden, um KI-Vorschläge zu machen.")
        update_llm_state(suggestion_ids=[])
        st.rerun()

# --- UI: Hauptbereich Layout (Karte und Wetter) ---
# Requirement 3: Visualisierung von Karte und Wetterdaten
#col_map, col_weather = st.columns([2, 1], gap="large")

#with col_map:
 #   df_map_display = pd.DataFrame()
  #  current_selection_id = st.session_state.get(config.STATE_SELECTED_ACTIVITY_INDEX)
   # explicit_rec_ids = st.session_state.get(config.STATE_EXPLICIT_RECOMMENDATIONS)

    #if explicit_rec_ids is not None: # Fall 1: Zeige explizite Profil-Empfehlungen
     #   if explicit_rec_ids and not df_activities.empty:
      #      valid_explicit_ids = [int(i) for i in explicit_rec_ids if isinstance(i, (int, float)) and pd.notna(i)]
       #     df_map_display = df_activities[df_activities[config.COL_ID].isin(valid_explicit_ids)].copy()
        #    if weather_data_map and config.COL_ID in df_activities.columns:
         #        id_to_idx = {row[config.COL_ID]: index for index, row in df_activities.iterrows() if pd.notna(row[config.COL_ID])}
          #       df_map_display['weather_note'] = df_map_display[config.COL_ID].map(lambda i: weather_data_map.get(id_to_idx.get(i, -1), {}).get('note'))

    #elif st.session_state.get(config.STATE_SHOW_LLM_RESULTS) and isinstance(st.session_state.get(config.STATE_LLM_SUGGESTION_IDS), list):
        # Fall 2: Zeige LLM-Vorschläge
     #   suggestion_ids = st.session_state.get(config.STATE_LLM_SUGGESTION_IDS, [])
      #  if suggestion_ids and not df_activities.empty:
       #     valid_suggestion_ids = [int(i) for i in suggestion_ids if isinstance(i, (int, float)) and pd.notna(i)]
        #    df_map_display = df_activities[df_activities[config.COL_ID].isin(valid_suggestion_ids)].copy()
         #   if weather_data_map and config.COL_ID in df_activities.columns:
          #       id_to_idx = {row[config.COL_ID]: index for index, row in df_activities.iterrows() if pd.notna(row[config.COL_ID])}
           #      df_map_display['weather_note'] = df_map_display[config.COL_ID].map(lambda i: weather_data_map.get(id_to_idx.get(i, -1), {}).get('note'))

   # elif not final_filtered_df.empty:
        # Fall 3: Zeige die normal gefilterten Ergebnisse
    #    df_map_display = final_filtered_df

    #display_map(df_map_display, selected_activity_id=current_selection_id)

#with col_weather:
 #   forecast_list_sg = None
  #  if config.OPENWEATHERMAP_API_CONFIGURED and datum is not None:
   #     forecast_list_sg = get_weather_forecast_for_day(
    #        api_key=config.OPENWEATHERMAP_API_KEY, lat=config.ST_GALLEN_LAT, lon=config.ST_GALLEN_LON, target_date=datum
     #   )
    #display_weather_overview(location_name="St. Gallen", target_date=datum, forecast_list=forecast_list_sg, api_configured=config.OPENWEATHERMAP_API_CONFIGURED)

# --- UI: Aktivitätenliste (Dynamischer Inhalt) ---
st.markdown("---")

explicit_rec_ids = st.session_state.get(config.STATE_EXPLICIT_RECOMMENDATIONS)
llm_suggestion_ids = st.session_state.get(config.STATE_LLM_SUGGESTION_IDS)
show_llm_results = st.session_state.get(config.STATE_SHOW_LLM_RESULTS)

list_content_shown = False

# --- Fall 1: Explizite Profil-Empfehlungen anzeigen ---
if explicit_rec_ids is not None:
    st.subheader("Passende Aktivitäten für dein Profil")
    list_content_shown = True
    if not df_activities.empty and explicit_rec_ids:
        valid_explicit_ids = [int(i) for i in explicit_rec_ids if isinstance(i, (int, float)) and pd.notna(i)]
        explicit_recs_df = df_activities[df_activities[config.COL_ID].isin(valid_explicit_ids)].copy()
        if weather_data_map and config.COL_ID in df_activities.columns:
            id_to_idx = {row[config.COL_ID]: index for index, row in df_activities.iterrows() if pd.notna(row[config.COL_ID])}
            weather_cols_keys = ['weather_note', 'location_temp', 'location_icon', 'location_desc']; weather_map_keys = ['note', 'temp', 'icon', 'desc']
            for col_name, detail_key in zip(weather_cols_keys, weather_map_keys): explicit_recs_df[col_name] = explicit_recs_df[config.COL_ID].map(lambda i: weather_data_map.get(id_to_idx.get(i, -1), {}).get(detail_key))
        if not explicit_recs_df.empty:
             try: explicit_recs_df[config.COL_ID] = pd.Categorical(explicit_recs_df[config.COL_ID], categories=valid_explicit_ids, ordered=True); explicit_recs_df = explicit_recs_df.sort_values(config.COL_ID)
             except Exception as e: print(f"Info: Konnte explizite Empfehlungen nicht sortieren: {e}")
             current_selection_id = st.session_state.get(config.STATE_SELECTED_ACTIVITY_INDEX)
             # Requirement 6: Der Code ist gut dokumentiert (durch Docstrings/Kommentare in Modulen und hier)
             for index, row in explicit_recs_df.iterrows():
                 activity_id_display = row[config.COL_ID]; is_expanded = (activity_id_display == current_selection_id)
                 # AUFRUF ANGEPASST: Ohne similarity_matrix, df_all_activities
                 display_activity_details(activity_row=row, activity_id=activity_id_display, is_expanded=is_expanded, openweathermap_api_configured=config.OPENWEATHERMAP_API_CONFIGURED, key_prefix="explicit")
        else: st.info("Keine gültigen Aktivitäten für Profil-Empfehlungen gefunden.")
    elif not explicit_rec_ids: st.info("Keine Aktivitäten für dein Profil gefunden.")
    else: st.error("Keine Aktivitätsdaten zum Anzeigen der Empfehlungen.")
    if st.button("Profil-Liste ausblenden", key="btn_hide_explicit"): st.session_state[config.STATE_EXPLICIT_RECOMMENDATIONS] = None; st.rerun()
    st.markdown("---")

# --- Fall 2: KI-Vorschläge anzeigen (nur wenn keine expliziten Profil-Empfehlungen aktiv sind) ---
elif show_llm_results and isinstance(llm_suggestion_ids, list):
    current_justification = st.session_state.get(config.STATE_LLM_JUSTIFICATION)
    if current_justification:
        if "Fehler" in current_justification or "konnte nicht" in current_justification: justification_placeholder.error(f"KI-Analyse: {current_justification}")
        else: justification_placeholder.info(f"KI-Analyse: {current_justification}")
    else: justification_placeholder.empty()

    if llm_suggestion_ids:
        st.subheader("KI-Vorschläge ✨")
        list_content_shown = True
        if not df_activities.empty:
            valid_suggestion_ids = [int(i) for i in llm_suggestion_ids if isinstance(i, (int, float)) and pd.notna(i)]
            suggestions_df_list = df_activities[df_activities[config.COL_ID].isin(valid_suggestion_ids)].copy()
            if weather_data_map and config.COL_ID in df_activities.columns:
                id_to_idx = {row[config.COL_ID]: index for index, row in df_activities.iterrows() if pd.notna(row[config.COL_ID])}
                weather_cols_keys = ['weather_note', 'location_temp', 'location_icon', 'location_desc']; weather_map_keys = ['note', 'temp', 'icon', 'desc']
                for col_name, detail_key in zip(weather_cols_keys, weather_map_keys): suggestions_df_list[col_name] = suggestions_df_list[config.COL_ID].map(lambda i: weather_data_map.get(id_to_idx.get(i, -1), {}).get(detail_key))
            if not suggestions_df_list.empty:
                try: suggestions_df_list[config.COL_ID] = pd.Categorical(suggestions_df_list[config.COL_ID], categories=valid_suggestion_ids, ordered=True); suggestions_df_list = suggestions_df_list.sort_values(config.COL_ID)
                except Exception as e: print(f"Info: Konnte LLM-Vorschläge nicht sortieren: {e}")
                current_selection_id = st.session_state.get(config.STATE_SELECTED_ACTIVITY_INDEX)
                for index, row in suggestions_df_list.iterrows():
                    is_expanded = (row[config.COL_ID] == current_selection_id)
                    # AUFRUF ANGEPASST: Ohne similarity_matrix, df_all_activities
                    display_activity_details(activity_row=row, activity_id=row[config.COL_ID], is_expanded=is_expanded, openweathermap_api_configured=config.OPENWEATHERMAP_API_CONFIGURED, key_prefix="llm")
            else: st.info("Keine gültigen KI-Vorschläge gefunden (nach Validierung).")
        else: st.error("Keine Aktivitätsdaten zum Anzeigen der Vorschläge.")
    st.markdown("---")

# --- Fall 3: Normale gefilterte Liste anzeigen (wenn weder explizit noch LLM aktiv) ---
else:
    if not final_filtered_df.empty:
        st.subheader(f"Gefilterte Aktivitäten ({len(final_filtered_df)})")
        list_content_shown = True
        MAX_LIST_ITEMS = 50
        display_df_limited = final_filtered_df.head(MAX_LIST_ITEMS)
        current_selection_id = st.session_state.get(config.STATE_SELECTED_ACTIVITY_INDEX)
        for index, row in display_df_limited.iterrows():
            is_expanded = (row[config.COL_ID] == current_selection_id)
            # AUFRUF ANGEPASST: Ohne similarity_matrix, df_all_activities
            display_activity_details(activity_row=row, activity_id=row[config.COL_ID], is_expanded=is_expanded, openweathermap_api_configured=config.OPENWEATHERMAP_API_CONFIGURED, key_prefix="filter")
        if len(final_filtered_df) > MAX_LIST_ITEMS:
            st.info(f"Hinweis: Nur die ersten {MAX_LIST_ITEMS} von {len(final_filtered_df)} Aktivitäten angezeigt.")

# --- Fallback-Meldung, wenn gar keine Aktivitäten angezeigt werden ---
if not list_content_shown:
    if not (show_llm_results and isinstance(llm_suggestion_ids, list) and not llm_suggestion_ids):
         justification_placeholder.empty()
         st.info("Keine Aktivitäten für die gewählten Filter und das Datum gefunden.")

# print("--- App-Durchlauf beendet ---")
