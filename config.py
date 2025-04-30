# config.py
"""
Konfigurationsdatei für die HSGexplorer Streamlit App.

Dieses Modul zentralisiert alle globalen Konstanten und Konfigurationseinstellungen
für die Anwendung. Dazu gehören:
- Geographische Koordinaten für St. Gallen.
- Schlüssel für den Streamlit Session State zur Zustandsverwaltung.
- Spaltennamen für den Aktivitäts-DataFrame zur Vermeidung von Tippfehlern und
  Sicherstellung der Konsistenz.
- Liste der erwarteten Spalten für die Datenvalidierung beim Laden.
- Pfade für Ressourcen (z.B. Logo).
- API-Schlüssel-Handling für Google AI und OpenWeatherMap aus st.secrets.
- Standardwerte für den Session State bei App-Start.
- Konfigurationen und erlaubte Werte für das LLM (Google Gemini).
"""

import streamlit as st
import os # Wird für os.path.basename in anderen Modulen benötigt

# --- Geographische Konstanten ---
ST_GALLEN_LAT: float = 47.4239
ST_GALLEN_LON: float = 9.3794

# --- Streamlit Session State Keys ---
# Verhindert Tippfehler und zentralisiert die Schlüsselnamen
STATE_LLM_FILTERS: str = 'llm_filters'
STATE_LLM_SUGGESTION_IDS: str = 'llm_suggestion_ids'
STATE_LLM_JUSTIFICATION: str = 'llm_justification'
STATE_SHOW_LLM_RESULTS: str = 'show_llm_results'
STATE_NLP_QUERY_SUBMITTED: str = 'nlp_query_submitted'
STATE_SELECTED_ACTIVITY_INDEX: str = 'selected_activity_index' # Für Kartenfokus
STATE_GOOGLE_KEY_WARNING_SHOWN: str = 'google_key_warning_shown' # Tracking für API-Warnungen
STATE_OWM_KEY_WARNING_SHOWN: str = 'owm_key_warning_shown'      # Tracking für API-Warnungen
STATE_LIKED_IDS: str = 'liked_ids'                 # Für Empfehlungsfunktion
STATE_DISLIKED_IDS: str = 'disliked_ids'             # Für Empfehlungsfunktion
STATE_RECOMMENDATIONS_TO_SHOW_IDS: str = 'recommendations_to_show_ids' # Aktuelle Vorschläge
STATE_FEATURES_MATRIX: str = 'features_matrix'       # Berechnete Features für Empfehlungen
STATE_SIMILARITY_MATRIX: str = 'similarity_matrix'     # Berechnete Ähnlichkeitsmatrix
STATE_USER_PROFILE: str = 'user_profile'           # Berechneter Profilvektor
STATE_USER_PROFILE_LABEL: str = 'user_profile_label'   # Generiertes Label für Nutzerprofil
STATE_EXPLICIT_RECOMMENDATIONS: str = 'explicit_recommendations_list' # Explizit angeforderte Profil-Empfehlungen
STATE_GOOGLE_AI_CONFIGURED: str = 'google_ai_configured' # Flag, ob Google AI Konfiguration erfolgreich war

# --- DataFrame Spaltennamen ---
# Definiert die exakten Spaltennamen aus aktivitaeten_neu.csv [cite: 32]
COL_ID: str = 'ID'
COL_NAME: str = 'Name'
COL_BESCHREIBUNG: str = 'Beschreibung'
COL_ART: str = 'Art'
COL_ORT: str = 'Ort_Name'
COL_ADRESSE: str = 'Adresse'
COL_LAT: str = 'Latitude'
COL_LON: str = 'Longitude'
COL_PREIS: str = 'Preis_Ca'
COL_PREIS_INFO: str = 'Preis_Info'
COL_WETTER_PREF: str = 'Wetter_Praeferenz'
COL_DATUM_VON: str = 'Datum_Von'
COL_DATUM_BIS: str = 'Datum_Bis'
COL_PERSONEN_MIN: str = 'Personen_Min'
COL_PERSONEN_MAX: str = 'Personen_Max'
COL_INDOOR_OUTDOOR: str = 'Indoor_Outdoor'
COL_ZIELGRUPPE: str = 'Zielgruppe' # Wichtig für Empfehlungen
COL_DAUER_INFO: str = 'Dauer_Info'
COL_WEBSITE: str = 'Website'
COL_KONTAKT_TEL: str = 'Kontakt_Telefon'
COL_BOOKING_INFO: str = 'Booking_Info'
COL_IMAGE_URL: str = 'Image_URL'

# --- Erwartete Spalten beim Datenladen ---
# Liste aller Spalten, die in der CSV erwartet werden und im DataFrame verfügbar sein sollen.
EXPECTED_COLUMNS: list[str] = [
    COL_ID, COL_NAME, COL_BESCHREIBUNG, COL_ART, COL_ORT, COL_ADRESSE, COL_LAT, COL_LON,
    COL_PREIS, COL_PREIS_INFO, COL_WETTER_PREF, COL_DATUM_VON, COL_DATUM_BIS,
    COL_PERSONEN_MIN, COL_PERSONEN_MAX, COL_INDOOR_OUTDOOR, COL_ZIELGRUPPE,
    COL_DAUER_INFO, COL_WEBSITE, COL_KONTAKT_TEL, COL_BOOKING_INFO, COL_IMAGE_URL
]

# --- UI Konstanten ---
LOGO_PATH: str = "logo.png" # Pfad zum Logo im Projektverzeichnis

# --- API Key Handling & Konfigurations-Flags ---
# Lädt API-Schlüssel sicher aus Streamlit Secrets und setzt Flags für die Verfügbarkeit.
# Requirement 2: Nutzung von externen APIs wird hier vorbereitet.

# Google AI API Key
try:
    # Verwende .get() um KeyError zu vermeiden, falls Secret fehlt
    GOOGLE_API_KEY: str | None = st.secrets.get("GOOGLE_API_KEY")
    # Einfache Prüfung auf Plausibilität (Länge, kein Platzhalter)
    if GOOGLE_API_KEY and len(GOOGLE_API_KEY) > 30 and "DEIN_" not in GOOGLE_API_KEY:
        GOOGLE_API_CONFIGURED: bool = True
    else:
        GOOGLE_API_KEY = None # Setze zurück auf None, wenn ungültig oder nicht vorhanden
        GOOGLE_API_CONFIGURED: bool = False
except Exception as e:
    # Fange unerwartete Fehler beim Lesen des Secrets ab
    print(f"WARNUNG: Unerwarteter Fehler beim Lesen des Google API Keys aus st.secrets: {e}")
    GOOGLE_API_KEY = None
    GOOGLE_API_CONFIGURED = False

# OpenWeatherMap API Key
try:
    OPENWEATHERMAP_API_KEY: str | None = st.secrets.get("OPENWEATHERMAP_API_KEY")
     # Einfache Prüfung auf Plausibilität (Länge, kein Platzhalter)
    if OPENWEATHERMAP_API_KEY and len(OPENWEATHERMAP_API_KEY) > 20 and "DEIN_" not in OPENWEATHERMAP_API_KEY:
        OPENWEATHERMAP_API_CONFIGURED: bool = True
    else:
        OPENWEATHERMAP_API_KEY = None # Setze zurück auf None, wenn ungültig oder nicht vorhanden
        OPENWEATHERMAP_API_CONFIGURED: bool = False
except Exception as e:
    print(f"WARNUNG: Unerwarteter Fehler beim Lesen des OpenWeatherMap API Keys aus st.secrets: {e}")
    OPENWEATHERMAP_API_KEY = None
    OPENWEATHERMAP_API_CONFIGURED = False


# --- Initial Session State Defaults ---
# Definiert die Standardwerte für den Session State beim ersten Start der App.
# Die eigentliche Initialisierung findet in app.py statt.
DEFAULT_SESSION_STATE: dict = {
    STATE_SELECTED_ACTIVITY_INDEX: None,
    STATE_LLM_FILTERS: None,
    STATE_LLM_SUGGESTION_IDS: None,
    STATE_LLM_JUSTIFICATION: None,
    STATE_SHOW_LLM_RESULTS: False,
    STATE_NLP_QUERY_SUBMITTED: None,
    STATE_GOOGLE_KEY_WARNING_SHOWN: False, # Tracking für Warnungen
    STATE_OWM_KEY_WARNING_SHOWN: False,     # Tracking für Warnungen
    STATE_LIKED_IDS: [],                    # Tracking für Empfehlungen
    STATE_DISLIKED_IDS: [],                 # Tracking für Empfehlungen
    STATE_RECOMMENDATIONS_TO_SHOW_IDS: [],  # Aktuelle Vorschlags-IDs
    STATE_FEATURES_MATRIX: None,            # Cache für ML-Features
    STATE_SIMILARITY_MATRIX: None,          # Cache für ML-Matrix
    STATE_USER_PROFILE: None,               # Cache für Nutzerprofil-Vektor
    STATE_USER_PROFILE_LABEL: None,         # Cache für Nutzerprofil-Beschreibung
    STATE_EXPLICIT_RECOMMENDATIONS: None,   # Liste explizit angeforderter Empfehlungen
    STATE_GOOGLE_AI_CONFIGURED: False       # Flag für erfolgreiche Google AI Konfiguration
}

# --- LLM Konfiguration (Google Gemini) ---
# Diese Listen definieren die erlaubten Werte, die das LLM bei der
# Filterextraktion für bestimmte Kategorien verwenden soll.
# Dies verbessert die Konsistenz der LLM-Ausgaben.
LLM_POSSIBLE_ARTEN: list[str] = ['Kultur', 'Natur', 'Sightseeing', 'Genuss', 'Familie', 'Sport', 'Entertainment', 'Shopping', 'Event', 'Wellness', 'Action', 'Freizeit']
LLM_POSSIBLE_PERSONEN_KAT: list[str] = ['Alleine', 'Paar', 'Kleingruppe', 'Grossgruppe'] # Vereinfachte Kategorien für LLM
LLM_POSSIBLE_INDOOR_OUTDOOR: list[str] = ['Indoor', 'Outdoor', 'Mixed']
LLM_POSSIBLE_WETTER_PREF: list[str] = ['Nur Sonne', 'Egal', 'Nur Regen']
# Zielgruppen könnten theoretisch auch dynamisch aus den Daten extrahiert werden,
# aber eine feste Liste gibt dem LLM klarere Vorgaben.
LLM_POSSIBLE_ZIELGRUPPEN: list[str] = [
    'Familie', 'Paare', 'Studenten', 'Senioren', 'Kinder', 'Jugendliche', 'Freunde',
    'Firmen', 'Alle', 'Kulturinteressierte', 'Modeinteressierte', 'Naturfreunde',
    'Touristen', 'Wanderer', 'Sportliche', 'Naschkatzen', 'Wellness-Suchende',
    'Autointeressierte', 'Männer', 'Bierliebhaber', 'Musikliebhaber',
    'Architekturinteressierte', 'Fotografen', 'Geniesser' # 'Geniesser' ergänzt
]