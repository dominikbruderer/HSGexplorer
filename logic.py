# logic.py
"""
Kernlogik für die Filterung und Anreicherung von Aktivitätsdaten in der explore-it App.

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
        COL_WETTER_PREF, COL_LAT, COL_LON
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
    base_filtered_df: pd.DataFrame, # Die Tabelle mit Aktivitäten, die bereits vor-gefiltert wurden (z.B. nach Datum, Art).
    original_df: pd.DataFrame,      # Die ursprüngliche, komplette Tabelle aller Aktivitäten (wichtig für Querverweise).
    selected_date: Optional[Union[datetime.date, pd.Timestamp]], # Das vom Nutzer gewählte Datum für die Wettervorhersage.
    consider_weather: bool,         # Ein Schalter (True/False): Soll nach Wetterpräferenz gefiltert werden?
    api_key: Optional[str],         # Der persönliche Schlüssel für den Wetterdienst (OpenWeatherMap).
    api_configured: bool            # Ein Schalter (True/False): Ist der Wetterdienst überhaupt startklar (API-Key vorhanden)?
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]], pd.DataFrame]:
    """
    Reichert Aktivitäten mit Wetterinformationen an und filtert sie optional basierend auf diesen Informationen.
    Diese Funktion ist optimiert, um Wetterdaten nur einmal pro einzigartigem geografischen Standort abzurufen,
    um die Geschwindigkeit zu erhöhen und API-Anfragen zu minimieren.

    Args:
        base_filtered_df: DataFrame mit Aktivitäten, die bereits durch Basisfilter gegangen sind.
        original_df: Der ursprüngliche, ungefilterte DataFrame aller Aktivitäten.
        selected_date: Das Datum, für das die Wettervorhersage relevant ist.
        consider_weather: Boolean, ob Aktivitäten basierend auf ihrer Wetterpräferenz
                          und der Vorhersage gefiltert werden sollen.
        api_key: Der API-Schlüssel für OpenWeatherMap.
        api_configured: Boolean, das anzeigt, ob der API-Schlüssel konfiguriert ist.

    Returns:
        Ein Tupel bestehend aus drei Elementen:
        1. final_filtered_df (pd.DataFrame): DataFrame mit Aktivitäten, die alle Filterkriterien
           (Basis + optional Wetter) erfüllen. Dieser wird typischerweise in der App angezeigt.
        2. weather_data_map_for_original (Dict): Ein Dictionary, das jeder Aktivitäts-ID aus dem
           *original_df* die zugehörigen Wetterdetails (Hinweis, Temperatur etc.) zuordnet.
           Nützlich für andere Programmteile wie die KI-Verarbeitung.
        3. df_with_weather_cols (pd.DataFrame): Eine Kopie von `base_filtered_df`, angereichert
           um die Wetterspalten, aber *nicht* notwendigerweise nach Wetterpräferenz gefiltert.
           Dient als Kandidatenliste mit Wetterinfos für die KI.
    """

    # --- Vorab-Prüfungen: Ist eine Wetterverarbeitung überhaupt sinnvoll oder möglich? ---
    # Wenn der Wetterdienst nicht konfiguriert ist, keine Aktivitäten in der Liste sind,
    # kein Datum gewählt wurde oder der API-Schlüssel fehlt, macht eine Wetterprüfung keinen Sinn.
    if not api_configured or base_filtered_df.empty or selected_date is None or api_key is None:
        # In diesem Fall machen wir eine Kopie der bereits gefilterten Liste.
        df_reset = base_filtered_df.copy()
        # Wir stellen sicher, dass die Spalten, in die später Wetterinfos kommen würden, trotzdem existieren (wenn auch leer).
        # Das verhindert Fehler in anderen Programmteilen, die diese Spalten erwarten.
        for col in ['weather_note', 'location_temp', 'location_icon', 'location_desc']:
            if col not in df_reset.columns:
                df_reset[col] = None # Füllt die Spalte mit "Nichts"
        
        # Der Index der Tabelle wird zurückgesetzt (neu durchnummeriert von 0 an).
        df_reset = df_reset.reset_index(drop=True)
        # Gib die (fast) unveränderte Liste und leere Wetter-Infos zurück.
        return df_reset.copy(), {}, df_reset.copy()

    # Erstelle eine Arbeitskopie der vor-gefilterten Aktivitäten-Tabelle.
    # So bleibt die ursprüngliche `base_filtered_df` unverändert.
    df_processing = base_filtered_df.copy()

    # --- Schritt 1: Einzigartige Standorte finden und Wetterdaten dafür sammeln ---
    # Ziel: Nicht für jede Aktivität einzeln das Wetter abfragen, wenn viele am selben Ort sind.
    # Wir extrahieren nur die Spalten für Breiten- und Längengrad.
    # `.dropna()`: Entfernt Zeilen, bei denen eine Koordinate fehlt.
    # `.drop_duplicates()`: Behält jede einzigartige Kombination von (Breitengrad, Längengrad) nur einmal.
    unique_locations_df = df_processing[[COL_LAT, COL_LON]].dropna().drop_duplicates()
    
    # Dies ist ein "Zwischenspeicher" (Cache) nur für diese Funktion.
    # Er merkt sich die Wetterdaten für jeden einzigartigen Ort (Schlüssel: (lat, lon)).
    # Inhalt pro Ort: Ein weiteres Dictionary mit 'status', 'temp', 'icon', 'desc'.
    location_weather_data_cache: Dict[Tuple[float, float], Dict[str, Any]] = {}

    # print(f"Debug: Prüfe Wetter für {len(unique_locations_df)} einzigartige Standorte.") # Nützlich für Entwickler

    # Gehe nun jeden einzigartigen Standort durch.
    for index, loc_row in unique_locations_df.iterrows():
        lat, lon = loc_row[COL_LAT], loc_row[COL_LON] # Breiten- und Längengrad des Standorts

        # Rufe die Wettervorhersage für diesen Standort und das gewählte Datum ab.
        # Diese Funktion (`get_weather_forecast_for_day`) kommt aus `weather_utils.py`
        # und hat ihren eigenen Cache, um API-Anfragen zu minimieren.
        forecast_list = get_weather_forecast_for_day(api_key, lat, lon, selected_date)
        
        # Bewerte die allgemeine Wetterlage ("Good", "Bad", "Uncertain", "Unknown").
        # Auch diese Funktion (`check_activity_weather_status`) kommt aus `weather_utils.py`.
        weather_status = check_activity_weather_status(forecast_list)
        
        # Initialisiere Variablen für Temperatur, Icon-Code und Wetterbeschreibung für diesen Standort.
        loc_temp, loc_icon, loc_desc = None, None, None
        if forecast_list: # Nur wenn eine Vorhersage vorhanden ist...
            try:
                # Filtere nach gültigen Vorhersageeinträgen (die ein 'datetime'-Objekt enthalten).
                valid_forecasts = [f for f in forecast_list if isinstance(f.get('datetime'), datetime.datetime)]
                if valid_forecasts:
                    # Wähle eine repräsentative Vorhersage (z.B. die erste ab 12 Uhr mittags, sonst die erste verfügbare).
                    rep_forecast = next((f for f in valid_forecasts if f['datetime'].hour >= 12), valid_forecasts[0])
                    loc_temp = rep_forecast.get('temp') # Temperatur
                    loc_icon = rep_forecast.get('icon') # Code für das Wettersymbol
                    loc_desc = str(rep_forecast.get('description', '')).capitalize() # Beschreibung, z.B. "Leichter Regen"
            except Exception:
                # Falls bei der Auswahl der repräsentativen Vorhersage etwas schiefgeht, bleiben die Werte None.
                pass 
        
        # Speichere die gesammelten Wetterinfos für diesen Standort im lokalen Cache.
        location_weather_data_cache[(lat, lon)] = {
            'status': weather_status, # z.B. "Good", "Bad"
            'temp': loc_temp,         # z.B. 20.5 (°C)
            'icon': loc_icon,         # z.B. "01d" (Code für sonnig)
            'desc': loc_desc          # z.B. "Klarer Himmel"
        }

    # --- Schritt 2: Wetterdaten den einzelnen Aktivitäten zuordnen und Filterlogik anwenden ---
    # Jetzt gehen wir die (vor-gefilterte) Aktivitätenliste `df_processing` durch
    # und fügen jeder Aktivität die passenden, bereits abgerufenen Wetterinfos hinzu.

    # Wir erstellen leere Listen, um die Wetterinfos für jede Aktivität zu sammeln.
    # Das ist effizienter, als den DataFrame in der Schleife ständig direkt zu ändern.
    weather_notes_list: List[Optional[str]] = []    # Für Hinweise wie "Passt nicht: Schlechtes Wetter"
    location_temps_list: List[Optional[float]] = [] # Für die Temperatur am Aktivitätsort
    location_icons_list: List[Optional[str]] = []   # Für den Icon-Code am Aktivitätsort
    location_descs_list: List[Optional[str]] = []   # Für die Wetterbeschreibung am Aktivitätsort
    keep_activity_flags: List[bool] = []            # True, wenn Aktivität behalten wird, sonst False

    # Dieser "Spickzettel" ist für andere Programmteile (z.B. die KI).
    # Er speichert Wetterinfos für die IDs aus der *ursprünglichen*, ungefilterten Aktivitätstabelle.
    weather_data_map_for_original: Dict[int, Dict[str, Any]] = {}
    # Hilfs-Dictionary, um von einer Aktivitäts-ID schnell zum Index in `original_df` zu gelangen.
    id_to_original_index_map: Dict[Any, int] = {}
    if not original_df.empty and COL_ID in original_df.columns:
        try:
            # Entferne Duplikate nach ID, falls vorhanden, um eindeutige Zuordnung zu gewährleisten.
            original_df_no_duplicates = original_df.drop_duplicates(subset=[COL_ID], keep='first')
            # Erstelle die Map: Aktivitäts-ID -> Zeilenindex im original_df.
            id_to_original_index_map = pd.Series(
                original_df_no_duplicates.index, # Der Wert ist der Index
                index=original_df_no_duplicates[COL_ID]  # Der Schlüssel ist die ID
            ).to_dict()
        except Exception as e:
            # Nur eine Warnung, falls das Erstellen der Map fehlschlägt.
            print(f"WARNUNG: Fehler beim Erstellen der ID->Index Map für Wetterdaten: {e}.")


    # Gehe jede Aktivität in der (noch nicht nach Wetter gefilterten) Liste `df_processing` durch.
    for index, activity_row in df_processing.iterrows():
        activity_id = activity_row.get(COL_ID)              # ID der aktuellen Aktivität
        lat = activity_row.get(COL_LAT)                     # Breitengrad
        lon = activity_row.get(COL_LON)                     # Längengrad
        activity_weather_pref = str(activity_row.get(COL_WETTER_PREF, 'Egal')).strip() # Wetterpräferenz der Aktivität
        # Falls die Präferenz leer oder ungültig ist, setze sie auf 'Egal'.
        if pd.isna(activity_weather_pref) or not activity_weather_pref : activity_weather_pref = 'Egal'

        # Initialisiere Variablen für die aktuelle Aktivität.
        current_note, current_temp, current_icon, current_desc = None, None, None, None
        should_keep_activity = True # Annahme: Aktivität wird erstmal behalten.

        # Prüfe, ob für den Standort dieser Aktivität Wetterdaten im Cache vorliegen.
        if pd.notna(lat) and pd.notna(lon) and (lat, lon) in location_weather_data_cache:
            # Ja, Daten sind da! Hole sie aus dem Cache.
            cached_loc_data = location_weather_data_cache[(lat, lon)]
            weather_status = cached_loc_data['status'] # "Good", "Bad", etc.
            current_temp = cached_loc_data['temp']
            current_icon = cached_loc_data['icon']
            current_desc = cached_loc_data['desc']

            # --- Hier findet die eigentliche Filterlogik statt, WENN `consider_weather` True ist ---
            if consider_weather: # Nur filtern, wenn der Nutzer das in der Sidebar ausgewählt hat.
                if activity_weather_pref == 'Nur Sonne':
                    if weather_status == "Good": # Perfekt! Gutes Wetter und "Nur Sonne" gewünscht.
                        should_keep_activity = True
                    elif weather_status == "Uncertain": # Unsicheres Wetter (z.B. bewölkt).
                        should_keep_activity = True # Aktivität trotzdem behalten...
                        current_note = "⚠️ Wetter unsicher (z.B. bewölkt), Aktivität bevorzugt aber Sonne." # ...aber mit Hinweis.
                    elif weather_status == "Unknown": # Wetterdaten konnten nicht eindeutig bestimmt werden.
                        should_keep_activity = True # Behalten...
                        current_note = "❓ Wetterdaten für Standort nicht eindeutig verfügbar/prüfbar." # ...mit Hinweis.
                    else:  # "Bad" (schlechtes) Wetter.
                        should_keep_activity = False # Aktivität rausfiltern.
                        current_note = "❌ Passt nicht: Schlechtes Wetter vorhergesagt, 'Nur Sonne' gewünscht."
                
                elif activity_weather_pref == 'Nur Regen':
                    if weather_status == "Bad": # Annahme: "Bad" beinhaltet Regen, Schnee etc.
                        should_keep_activity = True # Perfekt! Schlechtes Wetter und "Nur Regen" gewünscht.
                    elif weather_status == "Unknown":
                        should_keep_activity = True
                        current_note = "❓ Wetterdaten für Standort nicht eindeutig verfügbar/prüfbar."
                    else:  # "Good" (gutes) oder "Uncertain" (unsicheres) Wetter.
                        should_keep_activity = False # Aktivität rausfiltern.
                        current_note = "❌ Passt nicht: Gutes/Unsicheres Wetter, 'Nur Regen' gewünscht."
                
                # Wenn Wetterpräferenz 'Egal' ist, bleibt `should_keep_activity` True.
                # Falls eine unbekannte Wetterpräferenz im Datensatz steht:
                elif activity_weather_pref != 'Egal':
                    current_note = f"❓ Unbekannte Wetterpräferenz: {activity_weather_pref}"
            
            # Selbst wenn nicht gefiltert wird (`consider_weather` ist False),
            # kann ein allgemeiner Hinweis zum Wetter nützlich sein.
            #if not current_note and weather_status == "Bad": # Wenn noch kein spezifischer Hinweis da ist...
             #    current_note = "INFO: Schlechtes Wetter am Standort vorhergesagt."
            #elif not current_note and weather_status == "Uncertain":
             #    current_note = "INFO: Unsicheres Wetter am Standort vorhergesagt."

        else: # Keine gültigen Koordinaten für diese Aktivität vorhanden.
            current_note = "❓ Standortkoordinaten fehlen für Wetterprüfung."
            # Da das Wetter nicht geprüft werden kann, wird die Aktivität standardmäßig behalten.
            # Man könnte hier anders entscheiden, wenn `consider_weather` True ist und die Präferenz
            # z.B. "Nur Sonne" ist – aber das würde Aktivitäten ohne Koordinaten generell benachteiligen.
            should_keep_activity = True 
            
        # Füge die ermittelten Werte für diese Aktivität den entsprechenden Listen hinzu.
        weather_notes_list.append(current_note)
        location_temps_list.append(current_temp)
        location_icons_list.append(current_icon)
        location_descs_list.append(current_desc)
        keep_activity_flags.append(should_keep_activity) # Merken, ob diese Aktivität behalten wird.

        # Speichere die Wetterdaten auch für den "Spickzettel" (`weather_data_map_for_original`),
        # der sich auf die IDs im `original_df` bezieht.
        if activity_id is not None: # Nur wenn eine ID vorhanden ist.
            original_idx = id_to_original_index_map.get(activity_id) # Finde den Index im original_df.
            if original_idx is not None: # Nur wenn die ID im original_df gefunden wurde.
                weather_data_map_for_original[original_idx] = {
                    'note': current_note, 'temp': current_temp, 
                    'icon': current_icon, 'desc': current_desc
                }
            # else: print(f"Debug: activity_id {activity_id} nicht in id_to_original_index_map gefunden.") # Für Entwickler


    # --- Schritt 3: Finale DataFrames erstellen ---
    # Erstelle eine Kopie des `base_filtered_df`, um die neuen Wetterspalten hinzuzufügen.
    # Dieser DataFrame (`df_with_weather_cols`) enthält *alle* Aktivitäten aus `base_filtered_df`,
    # aber jetzt angereichert um die Wetterinformationen. Er dient z.B. als Kandidatenliste für die KI.
    df_with_weather_cols = base_filtered_df.copy()
    df_with_weather_cols['weather_note'] = weather_notes_list
    df_with_weather_cols['location_temp'] = location_temps_list
    df_with_weather_cols['location_icon'] = location_icons_list
    df_with_weather_cols['location_desc'] = location_descs_list
    
    # `final_filtered_df` ist die Tabelle, die tatsächlich in der App angezeigt wird.
    if consider_weather: # Wenn der Nutzer nach Wetter filtern wollte...
        # Erstelle eine "Maske" (eine Serie von True/False-Werten), basierend auf `keep_activity_flags`.
        # Der Index dieser Maske muss zum Index von `df_with_weather_cols` passen.
        final_keep_mask = pd.Series(keep_activity_flags, index=df_with_weather_cols.index)
        # Wende die Maske an, um nur die Aktivitäten zu behalten, bei denen der Flag True ist.
        final_filtered_df = df_with_weather_cols[final_keep_mask].copy()
    else: # Wenn nicht nach Wetter gefiltert wurde...
        # ...dann ist die finale Liste dieselbe wie die mit Wetterspalten angereicherte Liste.
        final_filtered_df = df_with_weather_cols.copy()
        
    # Setze die Indizes der Ergebnis-DataFrames zurück (saubere Nummerierung von 0 an).
    # `drop=True` verhindert, dass der alte Index als neue Spalte hinzugefügt wird.
    return final_filtered_df.reset_index(drop=True), weather_data_map_for_original, df_with_weather_cols.reset_index(drop=True)
