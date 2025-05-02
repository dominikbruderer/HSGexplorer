# logic.py
"""
Kernlogik für die Filterung und Anreicherung von Aktivitätsdaten im HSGexplorer.

Dieses Modul beinhaltet Funktionen für:
1. Basisfilterung von Aktivitäten nach Datum, Art, Personenanzahl und Budget
   (`apply_base_filters`).
2. Abruf von Wetterdaten, Anreicherung des DataFrames mit Wetterinformationen
   und optionale Filterung basierend auf Wetterpräferenzen (`apply_weather_filter`).

Die Funktionen nutzen Konstanten aus `config.py` und Wetter-Funktionen aus
`weather_utils.py`.
"""

import pandas as pd
import streamlit as st
import datetime
from typing import Tuple, Dict, Any, Optional, Union # Für Type Hints

# Importiere Konstanten für Spaltennamen
try:
    from config import (
        COL_ID, COL_ART, COL_PREIS, COL_DATUM_VON, COL_DATUM_BIS,
        COL_PERSONEN_MIN, COL_PERSONEN_MAX, COL_WETTER_PREF, COL_LAT, COL_LON
    )
except ImportError:
    st.error("Fehler: config.py konnte nicht importiert werden (logic.py).")
    # Definiere Dummy-Werte, um Importfehler zu vermeiden, aber Funktionalität ist beeinträchtigt
    COL_ID, COL_ART, COL_PREIS, COL_DATUM_VON, COL_DATUM_BIS = 'ID', 'Art', 'Preis_Ca', 'Datum_Von', 'Datum_Bis'
    COL_PERSONEN_MIN, COL_PERSONEN_MAX, COL_WETTER_PREF = 'Personen_Min', 'Personen_Max', 'Wetter_Praeferenz'
    COL_LAT, COL_LON = 'Latitude', 'Longitude'

# Importiere Wetterfunktionen
try:
    from weather_utils import get_weather_forecast_for_day, check_activity_weather_status
except ImportError:
    st.error("Fehler: weather_utils.py nicht gefunden oder fehlerhaft (logic.py).")
    # Definiere Dummy-Funktionen, um Abstürze zu vermeiden
    def get_weather_forecast_for_day(*args, **kwargs): return None
    def check_activity_weather_status(*args, **kwargs): return "Unknown"


def apply_base_filters(
    df: pd.DataFrame,
    selected_date: Optional[Union[datetime.date, pd.Timestamp]],
    activity_type_filter: str,
    people_filter: str,
    budget_filter: Optional[Union[float, int]]
    ) -> pd.DataFrame:
    """
    Wendet Basisfilter (Datum, Art, Personen, Budget) auf den DataFrame an.

    Filtert den übergebenen DataFrame basierend auf:
    - Zeitlicher Verfügbarkeit am `selected_date`.
    - Gewählter Aktivitätsart (`activity_type_filter`).
    - Gewünschter Gruppengröße (`people_filter`).
    - Maximalem Budget (`budget_filter`).

    Behandelt fehlende Werte tolerant und gibt eine gefilterte Kopie zurück.

    Args:
        df (pd.DataFrame): Der ursprüngliche DataFrame mit allen Aktivitäten.
        selected_date (Optional[datetime.date | pd.Timestamp]): Das Datum, für das
            die Aktivitäten verfügbar sein müssen. Wenn None, wird ein leerer DF zurückgegeben.
        activity_type_filter (str): Die gewünschte Aktivitätsart. "Alle" deaktiviert den Filter.
        people_filter (str): Die gewünschte Gruppengröße ("Alle", "Alleine", "Zu zweit",
             "Bis 4 Personen", "Mehr als 4 Personen", "Kleingruppe", "Grossgruppe").
             "Alle" deaktiviert den Filter.
        budget_filter (Optional[float | int]): Das maximale Budget pro Person.
             None oder nicht-numerischer Wert deaktiviert den Filter.

    Returns:
        pd.DataFrame: Eine Kopie des ursprünglichen DataFrames, die nur die Zeilen
            enthält, welche alle aktiven Filterkriterien erfüllen. Der Index wird
            zurückgesetzt. Kann leer sein.
    """
    # Früher Ausstieg bei leerem DataFrame oder fehlendem Datum
    if df.empty:
        # print("Debug: apply_base_filters - Eingabe-DataFrame ist leer.")
        return pd.DataFrame(columns=df.columns) # Leeren DF mit gleichen Spalten zurückgeben
    if selected_date is None:
        # print("Debug: apply_base_filters - Kein Datum ausgewählt.")
        return pd.DataFrame(columns=df.columns) # Ohne Datum keine Filterung möglich

    # Kopie erstellen, um Original nicht zu verändern
    filtered_df = df.copy()
    # print(f"Debug: apply_base_filters - Start mit {len(filtered_df)} Aktivitäten für Datum {selected_date}.")

    # --- 1. Datumsfilter ---
    # Prüfe, ob die Aktivität am `selected_date` verfügbar ist.
    try:
        # Konvertiere das ausgewählte Datum sicher in einen Timestamp für Vergleiche
        selected_datum_ts = pd.Timestamp(selected_date)

        # Prüfe Verfügbarkeit 'Datum_Von' (Startdatum der Aktivität)
        if COL_DATUM_VON in filtered_df.columns:
            # Konvertiere Spalte sicher zu Timestamps, ungültige Einträge werden NaT (Not a Time)
            datum_von_ts = pd.to_datetime(filtered_df[COL_DATUM_VON], errors='coerce')
            # Behalte Zeilen, wenn 'Datum_Von' leer ist ODER das Startdatum vor/am ausgewählten Datum liegt.
            mask_von = datum_von_ts.isna() | (datum_von_ts <= selected_datum_ts)
            filtered_df = filtered_df[mask_von]

        # Prüfe Verfügbarkeit 'Datum_Bis' (Enddatum der Aktivität)
        if COL_DATUM_BIS in filtered_df.columns:
            # Konvertiere Spalte sicher zu Timestamps, ungültige Einträge werden NaT
            datum_bis_ts = pd.to_datetime(filtered_df[COL_DATUM_BIS], errors='coerce')
            # Behalte Zeilen, wenn 'Datum_Bis' leer ist ODER das Enddatum am/nach dem ausgewählten Datum liegt.
            mask_bis = datum_bis_ts.isna() | (datum_bis_ts >= selected_datum_ts)
            filtered_df = filtered_df[mask_bis]

        # print(f"Debug: apply_base_filters - Nach Datumsfilter: {len(filtered_df)} Aktivitäten.")
        if filtered_df.empty: return filtered_df.reset_index(drop=True) # Frühzeitiger Ausstieg, wenn keine Aktivität am Datum verfügbar

    except Exception as e:
        st.warning(f"Fehler beim Anwenden des Datumsfilters: {e}")
        # Im Fehlerfall: Fahre mit den bis dahin gefilterten Daten fort oder gib leeren DF zurück.

    # --- 2. Aktivitätsart ---
    # Filtert nach der ausgewählten Art, wenn nicht "Alle" gewählt wurde.
    if activity_type_filter != "Alle" and COL_ART in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[COL_ART] == activity_type_filter]
        # print(f"Debug: apply_base_filters - Nach Art '{activity_type_filter}': {len(filtered_df)} Aktivitäten.")
        if filtered_df.empty: return filtered_df.reset_index(drop=True)

    # --- 3. Personenanzahl ---
    # Filtert basierend auf der gewünschten Gruppengröße.
    if people_filter != "Alle" and COL_PERSONEN_MIN in filtered_df.columns and COL_PERSONEN_MAX in filtered_df.columns:
        try:
            # Bereite die Personen-Spalten für den Vergleich vor:
            # Wandle in Zahlen um (falls nicht schon passiert), ersetze leere Werte (NaN)
            # mit sinnvollen Annahmen (Min=1, Max=unendlich)
            pers_min_col = pd.to_numeric(filtered_df[COL_PERSONEN_MIN], errors='coerce').fillna(1).astype(float)
            pers_max_col = pd.to_numeric(filtered_df[COL_PERSONEN_MAX], errors='coerce').fillna(float('inf')).astype(float)

            # Wende die Filterlogik für die jeweilige Kategorie an:
            if people_filter == "Alleine": # Personenzahl = 1
                # Aktivität muss min. 1 Person erlauben und max. 1 oder mehr.
                filtered_df = filtered_df[(pers_min_col <= 1) & (pers_max_col >= 1)]
            elif people_filter == "Zu zweit": # Personenzahl = 2
                 # Aktivität muss min. 2 oder weniger erlauben und max. 2 oder mehr.
                filtered_df = filtered_df[(pers_min_col <= 2) & (pers_max_col >= 2)]
            elif people_filter == "Bis 4 Personen": # Personenzahl = 1, 2, 3 oder 4
                # Aktivität muss min. 4 oder weniger erlauben und für mind. 1 Person geeignet sein.
                filtered_df = filtered_df[(pers_min_col <= 4) & (pers_max_col >= 1)]
            elif people_filter == "Mehr als 4 Personen": # Personenzahl = 5 oder mehr
                # Aktivität muss max. 5 oder mehr Personen erlauben.
                filtered_df = filtered_df[pers_max_col >= 5]
            # Interpretation der LLM-Kategorien (aus config.py)
            elif people_filter == "Kleingruppe": # Annahme z.B. 2-5 Personen
                # Aktivität muss für min. 5 oder weniger geeignet sein und max. 2 oder mehr erlauben.
                filtered_df = filtered_df[(pers_min_col <= 5) & (pers_max_col >= 2)]
            elif people_filter == "Grossgruppe": # Annahme z.B. 6+ Personen
                 # Aktivität muss max. 6 oder mehr Personen erlauben.
                filtered_df = filtered_df[pers_max_col >= 6]

            # print(f"Debug: apply_base_filters - Nach Personen '{people_filter}': {len(filtered_df)} Aktivitäten.")
            if filtered_df.empty: return filtered_df.reset_index(drop=True)
        except Exception as e:
            st.warning(f"Fehler beim Anwenden des Personenfilters für '{people_filter}': {e}")
            # Fahre fort ohne diesen Filter im Fehlerfall

    # --- 4. Budget ---
    # Filtert Aktivitäten, deren Preis unter oder gleich dem maximalen Budget liegt.
    if budget_filter is not None and COL_PREIS in filtered_df.columns:
        try:
            # Wandle Budget sicher in eine Zahl um.
            budget_num = float(budget_filter)
            # Vergleiche mit der Preis-Spalte (Fehlende Preise werden als 0 behandelt, siehe data_utils).
            # Behalte nur Zeilen, deren Preis <= budget_num ist.
            filtered_df = filtered_df[filtered_df[COL_PREIS] <= budget_num]
            # print(f"Debug: apply_base_filters - Nach Budget '<={budget_num}': {len(filtered_df)} Aktivitäten.")
        except (ValueError, TypeError):
            # Wenn Budget keine gültige Zahl ist, ignoriere den Filter.
            st.warning(f"Ungültiger Budgetwert '{budget_filter}'. Budgetfilter ignoriert.")
        except Exception as e:
            st.warning(f"Fehler beim Anwenden des Budgetfilters: {e}")
            # Fahre fort ohne diesen Filter im Fehlerfall

    # Index zurücksetzen für saubere weitere Verarbeitung und Rückgabe.
    # drop=True verhindert, dass der alte Index als neue Spalte hinzugefügt wird.
    return filtered_df.reset_index(drop=True)


def apply_weather_filter(
    base_filtered_df: pd.DataFrame,
    original_df: pd.DataFrame,
    selected_date: Optional[datetime.date | pd.Timestamp],
    consider_weather: bool, # Kommt von der Checkbox in der Sidebar
    api_key: Optional[str],
    api_configured: bool
    ) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]], pd.DataFrame]:
    """
    Reichert Aktivitäten mit Wetterinfos an und filtert optional danach.

    Diese Funktion nimmt die bereits nach Basis-Kriterien gefilterten Aktivitäten,
    holt für jede (falls Koordinaten vorhanden) die Wettervorhersage vom `selected_date`
    über `weather_utils.py`, bewertet die Wetterlage ("Good", "Bad", "Uncertain")
    und fügt Wetterinformationen (Temperatur, Symbol, Beschreibung, Hinweis) hinzu.

    Wenn `consider_weather` True ist (Checkbox in Sidebar aktiviert), werden
    Aktivitäten aussortiert, deren Wetterpräferenz (z.B. "Nur Sonne") nicht zur
    Vorhersage passt.

    Args:
        base_filtered_df (pd.DataFrame): DataFrame nach der Basisfilterung.
        original_df (pd.DataFrame): Der ursprüngliche, ungefilterte DataFrame (wichtig für ID-Mapping).
        selected_date (Optional[datetime.date | pd.Timestamp]): Das Datum für die Wettervorhersage.
        consider_weather (bool): Steuert, ob nach Wetterpräferenz gefiltert wird.
        api_key (Optional[str]): Der OpenWeatherMap API Key.
        api_configured (bool): Flag, ob der API Key vorhanden und gültig ist.

    Returns:
        Tuple[pd.DataFrame, Dict[int, Dict[str, Any]], pd.DataFrame]:
        1.  `final_filtered_df`: DataFrame mit Aktivitäten, die alle Filterkriterien erfüllen
            (Basis + optional Wetter). Wird für die Hauptliste angezeigt.
        2.  `weather_data_map`: Ein Dictionary, das Wetterdetails (Hinweis, Temp, etc.)
            der Aktivitäts-ID im *original_df* zuordnet. Nützlich für LLM-Kontext.
        3.  `df_with_weather_columns`: Eine Kopie von `base_filtered_df`, angereichert
            um die Wetterspalten, aber *nicht* notwendigerweise nach Wetter gefiltert.
            Dient als Kandidatenliste mit Wetterinfos für das LLM.
    """
    # Kopiere den basis-gefilterten DF, um ihn mit Wetterdaten anzureichern, ohne das Original zu ändern.
    df_with_weather = base_filtered_df.copy()

    # Initialisiere Wetterspalten mit leeren Werten, falls sie nicht schon existieren.
    weather_columns = ['weather_note', 'location_temp', 'location_icon', 'location_desc']
    for col in weather_columns:
        if col not in df_with_weather.columns:
            df_with_weather[col] = None # Oder pd.NA für Pandas >= 1.0

    # Initialisiere leere Ergebnis-Container
    weather_data_map: Dict[int, Dict[str, Any]] = {} # Map: Aktivitäts-ID -> {Wetterdetails}
    indices_pass_weather_filter: list[int] = [] # Sammelt Indizes der Aktivitäten, die den Wetterfilter (wenn aktiv) bestehen

    # Wetterprüfung überspringen, wenn nicht sinnvoll oder nicht möglich
    if not api_configured or df_with_weather.empty or selected_date is None or api_key is None:
        # Gib den (unveränderten) DF mit leeren Wetterspalten zurück
        df_reset = df_with_weather.reset_index(drop=True)
        return df_reset.copy(), weather_data_map, df_reset.copy()

    # --- Wetterdaten für jede Aktivität sammeln und bewerten ---
    # Erstelle ein Mapping von Aktivitäts-ID zu ihrem Index im *originalen* DataFrame.
    # Dies wird benötigt, um die Wetterdaten korrekt der ursprünglichen Aktivität zuzuordnen (für weather_data_map).
    id_to_original_index_map: Dict[Any, int] = {}
    if not original_df.empty and COL_ID in original_df.columns:
        try:
            # Stelle sicher, dass IDs im Original-DF für die Map eindeutig sind (falls Duplikate existieren)
            original_df_no_duplicates = original_df.drop_duplicates(subset=[COL_ID], keep='first')
            id_to_original_index_map = pd.Series(
                original_df_no_duplicates.index, # Der ursprüngliche Index
                index=original_df_no_duplicates[COL_ID] # Die ID als Schlüssel
            ).to_dict()
        except Exception as e:
            print(f"WARNUNG: Fehler beim Erstellen der ID->Index Map für Wetterdaten: {e}. weather_data_map wird evtl. unvollständig.")
            id_to_original_index_map = {} # Leere Map im Fehlerfall

    # Iteriere über jede Aktivität im *aktuell gefilterten* DataFrame (df_with_weather)
    for index, row in df_with_weather.iterrows():
        activity_id = row.get(COL_ID) # ID der aktuellen Aktivität
        lat = row.get(COL_LAT)
        lon = row.get(COL_LON)
        # Wetterpräferenz der Aktivität holen (Standard: 'Egal')
        activity_weather_pref = str(row.get(COL_WETTER_PREF, 'Egal'))
        if pd.isna(activity_weather_pref): activity_weather_pref = 'Egal'

        # Initialisiere Ergebnisvariablen für diese Aktivität
        note, loc_temp, loc_icon, loc_desc = None, None, None, None
        keep_activity = True # Standard: Aktivität behalten, es sei denn, der Wetterfilter greift

        # Fall 1: Keine Koordinaten -> Kein Wetterabruf möglich
        if pd.isna(lat) or pd.isna(lon):
            note = "❓ Standortkoordinaten fehlen für Wetterprüfung."
            # Aktivität wird immer behalten, da Wetter nicht geprüft werden kann.
            keep_activity = True
        # Fall 2: Koordinaten vorhanden -> Wetter abrufen und bewerten
        else:
            # Rufe Wettervorhersage für den Standort und das Datum ab (Funktion aus weather_utils)
            activity_forecast = get_weather_forecast_for_day(api_key, lat, lon, selected_date)
            # Bewerte die Wetterlage basierend auf der Vorhersage (Funktion aus weather_utils)
            weather_status = check_activity_weather_status(activity_forecast) # Ergebnis: "Good", "Bad", "Uncertain", "Unknown"

            # Extrahiere repräsentative Wetterdetails für die Anzeige (Temperatur, Symbol, Beschreibung - meist Mittagswert)
            if activity_forecast:
                try:
                    # (Logik zur Auswahl des repräsentativen Eintrags - hier vereinfacht dargestellt)
                    valid_forecasts = [f for f in activity_forecast if isinstance(f.get('datetime'), datetime.datetime)]
                    if valid_forecasts:
                        # Wähle erste Vorhersage ab 12 Uhr, sonst die erste überhaupt
                        rep_forecast = next((f for f in valid_forecasts if f['datetime'].hour >= 12), valid_forecasts[0])
                        loc_temp = rep_forecast.get('temp')
                        loc_icon = rep_forecast.get('icon')
                        loc_desc = str(rep_forecast.get('description', '')).capitalize()
                except Exception:
                    pass # Details bleiben None, wenn etwas schiefgeht

            # --- Entscheidung: Aktivität behalten oder rausfiltern? (Nur wenn `consider_weather` aktiv ist) ---
            # Diese Logik wird nur angewendet, wenn die Checkbox "Nach Wetter filtern" aktiviert ist.
            if consider_weather:
                if activity_weather_pref == 'Egal':
                    keep_activity = True # Wetter ist egal, immer behalten.
                elif activity_weather_pref == 'Nur Sonne':
                    # Behalten bei "Good" oder "Uncertain" Wetter, aber Hinweis bei "Uncertain". Nicht behalten bei "Bad".
                    if weather_status == "Good": keep_activity = True
                    elif weather_status == "Uncertain":
                        keep_activity = True; note = "⚠️ Wetter unsicher (z.B. bewölkt), Aktivität bevorzugt aber Sonne."
                    elif weather_status == "Unknown":
                        keep_activity = True; note = "❓ Wetterdaten nicht verfügbar/prüfbar."
                    else: # "Bad" weather
                        keep_activity = False; note = "❌ Passt nicht: Schlechtes Wetter vorhergesagt, aber 'Nur Sonne' gewünscht."
                elif activity_weather_pref == 'Nur Regen':
                    # Behalten nur bei "Bad" Wetter (Regen, Schnee, Gewitter...). Nicht behalten bei "Good" oder "Uncertain".
                    if weather_status == "Bad": keep_activity = True
                    elif weather_status == "Unknown":
                        keep_activity = True; note = "❓ Wetterdaten nicht verfügbar/prüfbar."
                    else: # "Good" or "Uncertain"
                        keep_activity = False; note = "❌ Passt nicht: Gutes/Unsicheres Wetter, aber 'Nur Regen' gewünscht."
                else: # Unbekannte Präferenz
                    keep_activity = True; note = f"❓ Unbekannte Wetterpräferenz: {activity_weather_pref}"
            # else: Wenn consider_weather False ist, bleibt keep_activity = True (Standard)

        # --- Ergebnisse für diese Aktivität speichern ---
        # Füge die gesammelten Wetterinfos zum DataFrame hinzu (immer, auch wenn nicht gefiltert wird).
        df_with_weather.loc[index, 'weather_note'] = note
        df_with_weather.loc[index, 'location_temp'] = loc_temp
        df_with_weather.loc[index, 'location_icon'] = loc_icon
        df_with_weather.loc[index, 'location_desc'] = loc_desc

        # Wenn die Aktivität (nach optionaler Wetterfilterung) behalten werden soll, merke dir ihren Index.
        if keep_activity:
            indices_pass_weather_filter.append(index)

        # Speichere Wetterdaten auch für die Map (gemappt auf originalen Index via ID).
        if activity_id is not None:
            original_index = id_to_original_index_map.get(activity_id)
            if original_index is not None:
                weather_data_map[original_index] = {
                    'note': note, 'temp': loc_temp, 'icon': loc_icon, 'desc': loc_desc
                 }

    # --- Finalen DataFrame basierend auf Wetterfilterung erstellen ---
    if consider_weather:
        # Wenn gefiltert wurde: Wähle nur die Zeilen aus, deren Indizes in `indices_pass_weather_filter` sind.
        final_filtered_df = df_with_weather.loc[indices_pass_weather_filter].copy()
    else:
        # Wenn nicht gefiltert wurde: Der finale DF ist derselbe wie der mit Wetterspalten angereicherte.
        final_filtered_df = df_with_weather.copy()

    # Gib die Ergebnisse zurück (Indizes zurücksetzen für Konsistenz).
    return final_filtered_df.reset_index(drop=True), weather_data_map, df_with_weather.reset_index(drop=True)