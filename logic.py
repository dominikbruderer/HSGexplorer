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
    try:
        # Konvertiere das ausgewählte Datum sicher in einen Timestamp für Vergleiche
        selected_datum_ts = pd.Timestamp(selected_date)

        # Prüfe Verfügbarkeit 'Datum_Von' (Startdatum)
        if COL_DATUM_VON in filtered_df.columns:
            # Konvertiere Spalte sicher zu Timestamps, ungültige Einträge werden NaT
            datum_von_ts = pd.to_datetime(filtered_df[COL_DATUM_VON], errors='coerce')
            # Behalte Zeilen, wenn 'Datum_Von' leer ist ODER <= selected_datum_ts
            mask_von = datum_von_ts.isna() | (datum_von_ts <= selected_datum_ts)
            filtered_df = filtered_df[mask_von]

        # Prüfe Verfügbarkeit 'Datum_Bis' (Enddatum)
        if COL_DATUM_BIS in filtered_df.columns:
            # Konvertiere Spalte sicher zu Timestamps, ungültige Einträge werden NaT
            datum_bis_ts = pd.to_datetime(filtered_df[COL_DATUM_BIS], errors='coerce')
            # Behalte Zeilen, wenn 'Datum_Bis' leer ist ODER >= selected_datum_ts
            mask_bis = datum_bis_ts.isna() | (datum_bis_ts >= selected_datum_ts)
            filtered_df = filtered_df[mask_bis]

        # print(f"Debug: apply_base_filters - Nach Datumsfilter: {len(filtered_df)} Aktivitäten.")
        if filtered_df.empty: return filtered_df.reset_index(drop=True) # Frühzeitiger Ausstieg

    except Exception as e:
        st.warning(f"Fehler beim Anwenden des Datumsfilters: {e}")
        # Im Fehlerfall: Fahre mit den bis dahin gefilterten Daten fort.
        # Alternative wäre: return pd.DataFrame(columns=df.columns)

    # --- 2. Aktivitätsart ---
    if activity_type_filter != "Alle" and COL_ART in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[COL_ART] == activity_type_filter]
        # print(f"Debug: apply_base_filters - Nach Art '{activity_type_filter}': {len(filtered_df)} Aktivitäten.")
        if filtered_df.empty: return filtered_df.reset_index(drop=True)

    # --- 3. Personenanzahl ---
    if people_filter != "Alle" and COL_PERSONEN_MIN in filtered_df.columns and COL_PERSONEN_MAX in filtered_df.columns:
        try:
            # Fülle fehlende Werte (NaN aus Laden/Konvertierung) sinnvoll auf für Vergleich
            # Annahme: Fehlendes Min = 1, Fehlendes Max = unendlich (inf)
            # Konvertiere sicherheitshalber zu float für Vergleich mit float('inf')
            pers_min_col = pd.to_numeric(filtered_df[COL_PERSONEN_MIN], errors='coerce').fillna(1).astype(float)
            pers_max_col = pd.to_numeric(filtered_df[COL_PERSONEN_MAX], errors='coerce').fillna(float('inf')).astype(float)

            # Logik für die verschiedenen Personen-Kategorien
            if people_filter == "Alleine":
                # Min muss <= 1 sein, Max muss >= 1 sein
                filtered_df = filtered_df[(pers_min_col <= 1) & (pers_max_col >= 1)]
            elif people_filter == "Zu zweit":
                 # Min muss <= 2 sein, Max muss >= 2 sein
                filtered_df = filtered_df[(pers_min_col <= 2) & (pers_max_col >= 2)]
            elif people_filter == "Bis 4 Personen":
                # Aktivität muss mind. 1 Person erlauben und max. 4 oder mehr Personen
                # Gleichzeitig muss Min <= 4 sein, um z.B. Aktivitäten nur für >4 auszuschließen
                filtered_df = filtered_df[(pers_min_col <= 4) & (pers_max_col >= 1)]
            elif people_filter == "Mehr als 4 Personen":
                # Aktivität muss mind. 5 Personen oder mehr erlauben (Max >= 5)
                filtered_df = filtered_df[pers_max_col >= 5]
            # Interpretation der LLM-Kategorien (konsistent mit config.LLM_POSSIBLE_PERSONEN_KAT)
            elif people_filter == "Kleingruppe": # Annahme z.B. 2-5 Personen
                # Aktivität muss für mind. 2 und max. 5 (oder mehr) geeignet sein
                filtered_df = filtered_df[(pers_min_col <= 5) & (pers_max_col >= 2)]
            elif people_filter == "Grossgruppe": # Annahme z.B. 6+ Personen
                 # Aktivität muss für 6 oder mehr Personen geeignet sein (Max >= 6)
                filtered_df = filtered_df[pers_max_col >= 6]
            # Hier könnten weitere Kategorien oder eine Logik für exakte Zahlen implementiert werden

            # print(f"Debug: apply_base_filters - Nach Personen '{people_filter}': {len(filtered_df)} Aktivitäten.")
            if filtered_df.empty: return filtered_df.reset_index(drop=True)
        except Exception as e:
            st.warning(f"Fehler beim Anwenden des Personenfilters für '{people_filter}': {e}")
            # Fahre fort ohne diesen Filter im Fehlerfall

    # --- 4. Budget ---
    if budget_filter is not None and COL_PREIS in filtered_df.columns:
        try:
            # Konvertiere Budget sicher in eine Zahl
            budget_num = float(budget_filter)
            # Vergleiche mit Preis_Ca (Fehlende Preise werden als 0 behandelt)
            # Behalte Zeilen, deren Preis <= budget_num ist
            filtered_df = filtered_df[filtered_df[COL_PREIS].fillna(0) <= budget_num]
            # print(f"Debug: apply_base_filters - Nach Budget '<={budget_num}': {len(filtered_df)} Aktivitäten.")
        except (ValueError, TypeError):
            # Wenn Budget keine Zahl ist, ignoriere den Filter und gib Warnung aus
            st.warning(f"Ungültiger Budgetwert '{budget_filter}'. Budgetfilter ignoriert.")
        except Exception as e:
            st.warning(f"Fehler beim Anwenden des Budgetfilters: {e}")
            # Fahre fort ohne diesen Filter im Fehlerfall

    # Index zurücksetzen für saubere weitere Verarbeitung und Rückgabe
    return filtered_df.reset_index(drop=True)


def apply_weather_filter(
    base_filtered_df: pd.DataFrame,
    original_df: pd.DataFrame,
    selected_date: Optional[datetime.date | pd.Timestamp],
    consider_weather: bool,
    api_key: Optional[str],
    api_configured: bool
    ) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]], pd.DataFrame]:
    """
    Ruft Wetterdaten ab, reichert den DataFrame an und filtert optional nach Wetter.

    Diese Funktion nimmt den bereits basis-gefilterten DataFrame (`base_filtered_df`),
    ruft für jede Aktivität mit Koordinaten die Wettervorhersage für den `selected_date`
    ab (via `weather_utils`), bewertet die Wetterlage und fügt Wetterinformationen
    (`weather_note`, `location_temp`, etc.) als neue Spalten hinzu.
    Wenn `consider_weather` True ist, filtert sie den DataFrame zusätzlich basierend
    auf der Wetterpräferenz der Aktivität (`COL_WETTER_PREF`) und der Vorhersage.
    Requirement 2: Nutzt indirekt die Wetter-API über weather_utils.

    Args:
        base_filtered_df (pd.DataFrame): Der DataFrame nach der Basisfilterung.
        original_df (pd.DataFrame): Der ursprüngliche, ungefilterte DataFrame.
            Wird benötigt, um Wetterdaten dem ursprünglichen Index zuzuordnen.
            Muss `COL_ID` enthalten.
        selected_date (Optional[datetime.date | pd.Timestamp]): Das Datum für die
            Wettervorhersage. Wenn None, wird Wetterprüfung übersprungen.
        consider_weather (bool): Steuert, ob der Wetterfilter angewendet wird.
        api_key (Optional[str]): Der OpenWeatherMap API Key.
        api_configured (bool): Flag, ob der API Key vorhanden und gültig ist.

    Returns:
        Tuple[pd.DataFrame, Dict[int, Dict[str, Any]], pd.DataFrame]:
        1.  `final_filtered_df` (pd.DataFrame): Der DataFrame, der alle Filter
            (Basis + optional Wetter) durchlaufen hat. Für die Anzeige der
            gefilterten Liste. Index ist zurückgesetzt.
        2.  `weather_data_map` (Dict): Ein Dictionary, das Wetterdetails
            ({note, temp, icon, desc}) auf den Index der Aktivität im
            *original_df* mappt (via `COL_ID`). Nützlich für LLM-Kontext.
        3.  `df_with_weather_columns` (pd.DataFrame): Eine Kopie des
            `base_filtered_df`, angereichert um die Wetterspalten (aber *nicht*
            nach Wetter gefiltert). Dient als Kandidatenliste mit Wetterinfos
            für das LLM. Index ist zurückgesetzt.
    """
    # Kopiere den basis-gefilterten DF, um ihn mit Wetterdaten anzureichern
    df_with_weather = base_filtered_df.copy()

    # Initialisiere Wetterspalten sicherheitshalber mit None
    weather_columns = ['weather_note', 'location_temp', 'location_icon', 'location_desc']
    for col in weather_columns:
        if col not in df_with_weather.columns:
            df_with_weather[col] = None # Oder pd.NA

    # Initialisiere leere Ergebnisse
    weather_data_map: Dict[int, Dict[str, Any]] = {} # Map: original_index -> {wetterdetails}
    indices_pass_weather_filter: list[int] = [] # Indizes im df_with_weather, die Filter bestehen

    # Überspringe Wetterabruf, wenn nicht sinnvoll/möglich
    if not api_configured or df_with_weather.empty or selected_date is None or api_key is None:
        # print("Debug: apply_weather_filter - Übersprungen (Keine API, Daten, Datum oder Key).")
        df_reset = df_with_weather.reset_index(drop=True)
        # Gib den (unveränderten) DF mit leeren Wetterspalten zurück, leere Map, und denselben DF als 'final'
        return df_reset.copy(), weather_data_map, df_reset.copy()

    # --- Wetterdaten sammeln und bewerten ---
    # print(f"Debug: apply_weather_filter - Starte Wetterabruf für {len(df_with_weather)} Aktivitäten am {selected_date}...")

    # Erstelle Mapping von Aktivitäts-ID zu ursprünglichem Index für weather_data_map
    # Wird benötigt, um Wetterinfos dem richtigen Eintrag im originalen DF zuzuordnen
    id_to_original_index_map: Dict[Any, int] = {}
    if not original_df.empty and COL_ID in original_df.columns:
        try:
            # Stelle sicher, dass IDs im Original-DF für die Map eindeutig sind
            original_df_no_duplicates = original_df.drop_duplicates(subset=[COL_ID])
            id_to_original_index_map = pd.Series(
                original_df_no_duplicates.index,
                index=original_df_no_duplicates[COL_ID]
            ).to_dict()
        except Exception as e:
            print(f"WARNUNG: Fehler beim Erstellen der ID->Index Map: {e}. weather_data_map wird unvollständig sein.")
            id_to_original_index_map = {} # Leere Map im Fehlerfall
    else:
        print("WARNUNG: Original DataFrame für ID-Index-Map fehlt oder hat keine ID-Spalte. weather_data_map kann nicht befüllt werden.")

    # Iteriere über die Indizes des *aktuellen* DataFrames (df_with_weather)
    for index in df_with_weather.index:
        row = df_with_weather.loc[index]
        activity_id = row.get(COL_ID) # ID der aktuellen Aktivität holen
        lat = row.get(COL_LAT)
        lon = row.get(COL_LON)
        # Wetterpräferenz der Aktivität (Default 'Egal')
        activity_weather_pref = str(row.get(COL_WETTER_PREF, 'Egal'))
        if pd.isna(activity_weather_pref): activity_weather_pref = 'Egal'

        # Initialisiere Variablen für diese Iteration
        note, loc_temp, loc_icon, loc_desc = None, None, None, None
        keep_based_on_weather = True # Standard: Behalten, außer Filter schlägt fehl

        # --- Fall 1: Keine Koordinaten -> Kein Wetterabruf möglich ---
        if pd.isna(lat) or pd.isna(lon):
            note = "❓ Standortkoordinaten fehlen."
            # Aktivität wird immer behalten, da Wetter nicht geprüft werden kann
            keep_based_on_weather = True
        # --- Fall 2: Koordinaten vorhanden -> Wetter abrufen und bewerten ---
        else:
            activity_forecast = get_weather_forecast_for_day(api_key, lat, lon, selected_date)
            weather_status = check_activity_weather_status(activity_forecast) # "Good", "Bad", "Uncertain", "Unknown"

            # Extrahiere repräsentative Wetterdetails für Anzeige/Map (Mittag/erste Vorhersage)
            if activity_forecast:
                try:
                    valid_forecasts = [f for f in activity_forecast if isinstance(f.get('datetime'), datetime.datetime)]
                    if valid_forecasts:
                        loc_representative_forecast = next((f for f in valid_forecasts if f['datetime'].hour >= 12), valid_forecasts[0])
                        loc_temp = loc_representative_forecast.get('temp')
                        loc_icon = loc_representative_forecast.get('icon')
                        loc_desc = str(loc_representative_forecast.get('description', '')).capitalize()
                except Exception as e:
                    # print(f"Debug: Fehler bei Auswahl repr. Wettervorhersage für Index {index} ({lat},{lon}): {e}")
                    pass # Details bleiben None

            # --- Entscheidung treffen, ob Aktivität behalten wird (wenn Wetterfilter aktiv ist) ---
            if activity_weather_pref == 'Egal':
                keep_based_on_weather = True # Wetter ist egal
            elif activity_weather_pref == 'Nur Sonne':
                if weather_status == "Good":
                    keep_based_on_weather = True
                elif weather_status == "Uncertain":
                    keep_based_on_weather = True # Unsicher ist ok, aber mit Hinweis
                    note = "⚠️ Wetter unsicher (z.B. bewölkt), Aktivität bevorzugt aber Sonne."
                elif weather_status == "Unknown":
                     keep_based_on_weather = True # Wenn nicht prüfbar, vorsichtshalber behalten
                     note = "❓ Wetterdaten nicht verfügbar/verarbeitbar."
                else: # "Bad" weather
                    keep_based_on_weather = False # Nicht behalten bei schlechtem Wetter
            elif activity_weather_pref == 'Nur Regen':
                 # Behalten nur wenn Wetter als 'Bad' klassifiziert wurde (Regen, Schnee, Gewitter etc.)
                 if weather_status == "Bad":
                     keep_based_on_weather = True
                 elif weather_status == "Unknown":
                     keep_based_on_weather = True # Wenn nicht prüfbar, vorsichtshalber behalten
                     note = "❓ Wetterdaten nicht verfügbar/verarbeitbar."
                 else: # "Good" or "Uncertain"
                     keep_based_on_weather = False
                     note = "ℹ️ Aktivität bevorzugt Regen, aber Wetter ist nicht 'Bad'."
            else: # Unbekannte Präferenz -> Sicherheitshalber behalten
                keep_based_on_weather = True
                # print(f"Debug: Unbekannte Wetterpräferenz '{activity_weather_pref}' bei Index {index}.")
                note = f"❓ Unbekannte Wetterpräferenz: {activity_weather_pref}"

        # --- Ergebnisse für diese Aktivität speichern ---
        # Wetterdetails IMMER im DataFrame speichern (auch wenn nicht gefiltert wird)
        df_with_weather.loc[index, 'weather_note'] = note
        df_with_weather.loc[index, 'location_temp'] = loc_temp
        df_with_weather.loc[index, 'location_icon'] = loc_icon
        df_with_weather.loc[index, 'location_desc'] = loc_desc

        # Füge Index zur Liste der zu behaltenden hinzu, wenn Wetter passt
        if keep_based_on_weather:
            indices_pass_weather_filter.append(index)

        # Wetterdaten auch für die Map (gemappt auf originalen Index via ID) speichern
        if activity_id is not None:
            original_index = id_to_original_index_map.get(activity_id)
            if original_index is not None:
                weather_data_map[original_index] = {'note': note, 'temp': loc_temp, 'icon': loc_icon, 'desc': loc_desc}
            # else: ID nicht in Map gefunden (Warnung wurde oben ausgegeben)

    # print(f"Debug: apply_weather_filter - Wetterabruf beendet. {len(indices_pass_weather_filter)} Aktivitäten bestehen potenziellen Wetterfilter.")

    # --- Optionales Filtern basierend auf Checkbox ---
    if consider_weather:
        # Filtere den DataFrame `df_with_weather` basierend auf den gesammelten Indizes
        # .loc ist sicher, auch wenn indices_pass_weather_filter leer ist
        final_filtered_df = df_with_weather.loc[df_with_weather.index.intersection(indices_pass_weather_filter)].copy()
        # print(f"Debug: apply_weather_filter - Nach Wetterfilterung (consider=True): {len(final_filtered_df)} Aktivitäten.")
    else:
        # Wenn nicht gefiltert wird, ist das Ergebnis der DataFrame mit Wetterspalten
        final_filtered_df = df_with_weather.copy()
        # print("Debug: apply_weather_filter - Wetterfilter nicht angewendet (consider=False).")

    # Gib die Ergebnisse zurück (Indizes zurücksetzen für Konsistenz)
    return final_filtered_df.reset_index(drop=True), weather_data_map, df_with_weather.reset_index(drop=True)