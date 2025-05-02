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

# --- Geographische Konstanten ---
ST_GALLEN_LAT: float = 47.4239
ST_GALLEN_LON: float = 9.3794

# --- Streamlit Session State Keys ---
# Verhindert Tippfehler und zentralisiert die Schlüsselnamen
# Der Session State ist wie das "Gedächtnis" der App über verschiedene Interaktionen hinweg.
# Hier legen wir fest, unter welchen "Namen" (Keys) wir Informationen darin speichern.
STATE_LLM_FILTERS: str = 'llm_filters' # Speicher für vom LLM extrahierte Filter
STATE_LLM_SUGGESTION_IDS: str = 'llm_suggestion_ids' # Speicher für vom LLM vorgeschlagene IDs
STATE_LLM_JUSTIFICATION: str = 'llm_justification' # Speicher für die Begründung des LLM
STATE_SHOW_LLM_RESULTS: str = 'show_llm_results' # Flag: Sollen LLM-Ergebnisse angezeigt werden?
STATE_NLP_QUERY_SUBMITTED: str = 'nlp_query_submitted' # Speicher für die letzte Nutzeranfrage ans LLM
STATE_SELECTED_ACTIVITY_INDEX: str = 'selected_activity_index' # ID der Aktivität, die auf der Karte fokussiert ist
STATE_GOOGLE_KEY_WARNING_SHOWN: str = 'google_key_warning_shown' # Flag: Wurde API-Warnung schon gezeigt?
STATE_OWM_KEY_WARNING_SHOWN: str = 'owm_key_warning_shown'      # Flag: Wurde API-Warnung schon gezeigt?
STATE_LIKED_IDS: str = 'liked_ids'                 # Liste der positiv bewerteten Aktivitäts-IDs
STATE_DISLIKED_IDS: str = 'disliked_ids'             # Liste der negativ bewerteten Aktivitäts-IDs
STATE_RECOMMENDATIONS_TO_SHOW_IDS: str = 'recommendations_to_show_ids' # IDs für die Vorschlagskarten
STATE_FEATURES_MATRIX: str = 'features_matrix'       # Speicher für ML-Feature-Matrix (berechnet)
STATE_SIMILARITY_MATRIX: str = 'similarity_matrix'     # Speicher für ML-Ähnlichkeitsmatrix (berechnet) -> Hinweis: Aktuell nicht verwendet im Code!
STATE_USER_PROFILE: str = 'user_profile'           # Speicher für ML-Nutzerprofil-Vektor (berechnet)
STATE_USER_PROFILE_LABEL: str = 'user_profile_label'   # Speicher für die Beschreibung des Nutzerprofils
STATE_EXPLICIT_RECOMMENDATIONS: str = 'explicit_recommendations_list' # Liste explizit angeforderter Profil-Empfehlungen
STATE_GOOGLE_AI_CONFIGURED: str = 'google_ai_configured' # Flag: Ist Google AI erfolgreich konfiguriert?

# --- DataFrame Spaltennamen ---
# Definiert die exakten Spaltennamen aus der CSV-Datei aktivitaeten_neu.csv [cite: 32]
# Die Verwendung von Konstanten hier vermeidet Tippfehler im restlichen Code.
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
# Wird in data_utils.py verwendet, um die geladene Datei zu prüfen.
EXPECTED_COLUMNS: list[str] = [
    COL_ID, COL_NAME, COL_BESCHREIBUNG, COL_ART, COL_ORT, COL_ADRESSE, COL_LAT, COL_LON,
    COL_PREIS, COL_PREIS_INFO, COL_WETTER_PREF, COL_DATUM_VON, COL_DATUM_BIS,
    COL_PERSONEN_MIN, COL_PERSONEN_MAX, COL_INDOOR_OUTDOOR, COL_ZIELGRUPPE,
    COL_DAUER_INFO, COL_WEBSITE, COL_KONTAKT_TEL, COL_BOOKING_INFO, COL_IMAGE_URL
]

# --- UI Konstanten ---
LOGO_PATH: str = "logo.png" # Pfad zum Logo im Projektverzeichnis

# --- API Key Handling & Konfigurations-Flags ---
# Lädt API-Schlüssel sicher aus Streamlit Secrets (st.secrets)
# st.secrets ist ein spezieller Speicherort für sensible Daten wie API-Keys in Streamlit.
# Requirement 2: Nutzung von externen APIs wird hier vorbereitet.

# Google AI API Key (für Gemini LLM)
try:
    # Versuche, den Key aus den Secrets zu lesen. .get() vermeidet Fehler, wenn der Key fehlt.
    GOOGLE_API_KEY: str | None = st.secrets.get("GOOGLE_API_KEY")
    # Einfache Prüfung, ob der Key vorhanden und nicht nur ein Platzhalter ist.
    if GOOGLE_API_KEY and len(GOOGLE_API_KEY) > 30 and "DEIN_" not in GOOGLE_API_KEY:
        GOOGLE_API_CONFIGURED: bool = True # Setze Flag auf True, wenn Key gültig aussieht
    else:
        GOOGLE_API_KEY = None # Setze auf None, wenn ungültig oder nicht vorhanden
        GOOGLE_API_CONFIGURED: bool = False
except Exception as e:
    # Fange unerwartete Fehler beim Lesen der Secrets ab.
    print(f"WARNUNG: Unerwarteter Fehler beim Lesen des Google API Keys aus st.secrets: {e}")
    GOOGLE_API_KEY = None
    GOOGLE_API_CONFIGURED = False

# OpenWeatherMap API Key (für Wetterdaten)
try:
    OPENWEATHERMAP_API_KEY: str | None = st.secrets.get("OPENWEATHERMAP_API_KEY")
    # Einfache Prüfung, ob der Key vorhanden und nicht nur ein Platzhalter ist.
    if OPENWEATHERMAP_API_KEY and len(OPENWEATHERMAP_API_KEY) > 20 and "DEIN_" not in OPENWEATHERMAP_API_KEY:
        OPENWEATHERMAP_API_CONFIGURED: bool = True # Setze Flag auf True, wenn Key gültig aussieht
    else:
        OPENWEATHERMAP_API_KEY = None # Setze auf None, wenn ungültig oder nicht vorhanden
        OPENWEATHERMAP_API_CONFIGURED: bool = False
except Exception as e:
    print(f"WARNUNG: Unerwarteter Fehler beim Lesen des OpenWeatherMap API Keys aus st.secrets: {e}")
    OPENWEATHERMAP_API_KEY = None
    OPENWEATHERMAP_API_CONFIGURED = False


# --- Initial Session State Defaults ---
# Definiert die Standardwerte für das "Gedächtnis" (Session State) der App beim ersten Start.
# Die eigentliche Initialisierung dieser Werte findet in app.py statt.
DEFAULT_SESSION_STATE: dict = {
    STATE_SELECTED_ACTIVITY_INDEX: None,      # Welche Aktivität ist auf Karte ausgewählt? -> Keine
    STATE_LLM_FILTERS: None,                  # Vom LLM extrahierte Filter -> Keine
    STATE_LLM_SUGGESTION_IDS: None,           # Vom LLM vorgeschlagene IDs -> Keine
    STATE_LLM_JUSTIFICATION: None,            # Begründung des LLM -> Keine
    STATE_SHOW_LLM_RESULTS: False,            # Sollen LLM-Ergebnisse gezeigt werden? -> Nein
    STATE_NLP_QUERY_SUBMITTED: None,          # Letzte Nutzeranfrage -> Keine
    STATE_GOOGLE_KEY_WARNING_SHOWN: False,    # Warnung für Google Key gezeigt? -> Nein
    STATE_OWM_KEY_WARNING_SHOWN: False,       # Warnung für Wetter Key gezeigt? -> Nein
    STATE_LIKED_IDS: [],                      # Liste der gelikten IDs -> Leer
    STATE_DISLIKED_IDS: [],                   # Liste der disliketen IDs -> Leer
    STATE_RECOMMENDATIONS_TO_SHOW_IDS: [],    # Welche IDs in Vorschlagskarte zeigen? -> Leer
    STATE_FEATURES_MATRIX: None,              # Berechnete ML-Features -> Keine
    STATE_SIMILARITY_MATRIX: None,            # Berechnete ML-Ähnlichkeiten -> Keine (derzeit nicht genutzt)
    STATE_USER_PROFILE: None,                 # Berechneter ML-Nutzerprofil-Vektor -> Keiner
    STATE_USER_PROFILE_LABEL: None,           # Beschreibung des Nutzerprofils -> Keine
    STATE_EXPLICIT_RECOMMENDATIONS: None,     # Explizit angeforderte Empfehlungen -> Keine
    STATE_GOOGLE_AI_CONFIGURED: False         # Ist Google AI konfiguriert? -> Nein (wird später geprüft)
}

# --- LLM Konfiguration (Google Gemini) ---
# Diese Listen definieren, welche Werte das LLM (Gemini) bei der Analyse von Nutzereingaben
# für bestimmte Filterkategorien erkennen und zurückgeben soll.
# Das hilft, die Antworten der KI konsistenter und für die App nutzbar zu machen.
LLM_POSSIBLE_ARTEN: list[str] = [
    'Kultur', 'Natur', 'Sightseeing', 'Genuss', 'Familie', 'Sport',
    'Entertainment', 'Shopping', 'Event', 'Wellness', 'Action', 'Freizeit'
]
LLM_POSSIBLE_PERSONEN_KAT: list[str] = [
    'Alleine', 'Paar', 'Kleingruppe', 'Grossgruppe'
] # Vereinfachte Kategorien, die das LLM erkennen soll
LLM_POSSIBLE_INDOOR_OUTDOOR: list[str] = [
    'Indoor', 'Outdoor', 'Mixed'
]
LLM_POSSIBLE_WETTER_PREF: list[str] = [
    'Nur Sonne', 'Egal', 'Nur Regen'
]
# Zielgruppen: Könnten auch aus den Daten gelesen werden, aber eine feste Liste ist für das LLM einfacher.
LLM_POSSIBLE_ZIELGRUPPEN: list[str] = [
    'Alle', 'Familie', 'Paare', 'Studenten', 'Senioren', 'Kinder', 'Jugendliche',
    'Freunde', 'Firmen', 'Kulturinteressierte', 'Modeinteressierte', 'Naturfreunde',
    'Touristen', 'Wanderer', 'Sportliche', 'Naschkatzen', 'Wellness-Suchende',
    'Autointeressierte', 'Männer', 'Bierliebhaber', 'Musikliebhaber',
    'Architekturinteressierte', 'Fotografen', 'Geniesser'
]