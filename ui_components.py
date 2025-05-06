# ui_components.py
"""
Funktionen zur Erstellung von UI-Komponenten für die explore-it Streamlit App.

Dieses Modul enthält Funktionen, die für die Erzeugung spezifischer Teile
der Benutzeroberfläche verantwortlich sind (z.B. Sidebar, Karte, Detailansicht).
Ziel ist es, die UI-Logik von der Hauptanwendungslogik (`app.py`) zu trennen.
Die Funktionen nutzen Streamlit-Widgets und -Layouts sowie externe Bibliotheken
wie streamlit-folium und Plotly zur Visualisierung.
"""

import streamlit as st
import pandas as pd
import datetime
import os
import folium # Für die Kartenerstellung
import plotly.graph_objects as go # Für Diagramme
from streamlit_folium import st_folium # Zur Anzeige von Folium-Karten in Streamlit
from typing import List, Dict, Any, Optional, Union, Callable, Tuple # Für Type Hints

# Importiere Konstanten und Konfigurationen
try:
    import config
    # Importiere spezifische Konstanten für Lesbarkeit (optional, aber kann helfen)
    from config import (
        LOGO_PATH, STATE_SHOW_LLM_RESULTS, STATE_SELECTED_ACTIVITY_INDEX,
        ST_GALLEN_LAT, ST_GALLEN_LON, COL_ID, COL_ART, COL_PREIS, COL_ORT,
        COL_NAME, COL_LAT, COL_LON, COL_WETTER_PREF, COL_DATUM_VON, COL_DATUM_BIS,
        COL_BESCHREIBUNG, COL_ZIELGRUPPE,
        COL_INDOOR_OUTDOOR, COL_DAUER_INFO, COL_WEBSITE, COL_BOOKING_INFO,
        COL_KONTAKT_TEL, COL_IMAGE_URL
    )
except ImportError:
    st.error("Fehler: config.py konnte nicht importiert werden (ui_components.py).")
    # Definiere Fallback-Werte, um Abstürze zu vermeiden, aber Funktionalität ist beeinträchtigt
    LOGO_PATH = "logo.png"; STATE_SHOW_LLM_RESULTS = 'show_llm_results'; STATE_SELECTED_ACTIVITY_INDEX = 'selected_activity_index';
    ST_GALLEN_LAT, ST_GALLEN_LON = 47.4239, 9.3794; COL_ID = 'ID'; COL_ART = 'Art'; COL_PREIS = 'Preis_Ca'; COL_ORT = 'Ort_Name';
    COL_NAME = 'Name'; COL_LAT = 'Latitude'; COL_LON = 'Longitude'; COL_WETTER_PREF = 'Wetter_Praeferenz'; COL_DATUM_VON = 'Datum_Von';
    COL_DATUM_BIS = 'Datum_Bis'; COL_BESCHREIBUNG = 'Beschreibung'; COL_PERSONEN_MIN = 'Personen_Min'; COL_PERSONEN_MAX = 'Personen_Max';
    COL_ZIELGRUPPE = 'Zielgruppe'; COL_INDOOR_OUTDOOR = 'Indoor_Outdoor'; COL_DAUER_INFO = 'Dauer_Info'; COL_WEBSITE = 'Website';
    COL_BOOKING_INFO = 'Booking_Info'; COL_KONTAKT_TEL = 'Kontakt_Telefon'; COL_IMAGE_URL = 'Image_URL';


def display_sidebar(
    df: pd.DataFrame,
    today_date: datetime.date,
    openweathermap_api_configured: bool
    ) -> Tuple[datetime.date, bool, str, Optional[float], bool]:
    """
    Erstellt und zeigt die Streamlit Sidebar an und gibt die Filterwerte zurück.

    Die Sidebar enthält das Logo, die Datumsauswahl, die Wetter-Checkbox
    und die manuellen Filter für Aktivitätsart, Personenzahl und Budget.
    Sie gibt die vom Nutzer ausgewählten Werte zurück.

    Args:
        df (pd.DataFrame): Der DataFrame mit allen Aktivitäten (wird benötigt, um
                           dynamisch Filteroptionen zu generieren, z.B. für Aktivitätsart).
        today_date (datetime.date): Das heutige Datum (als Startwert für Datumsauswahl).
        openweathermap_api_configured (bool): Flag, ob die Wetter-API konfiguriert ist
                                            (um die Wetter-Checkbox ggf. zu deaktivieren).

    Returns:
        Tuple[datetime.date, bool, str, str, Optional[float], bool]:
        Ein Tupel mit den ausgewählten Werten:
        - selected_date: Das vom Nutzer gewählte Datum.
        - consider_weather: Boolean, ob die Wetter-Checkbox aktiviert ist.
        - activity_type: Ausgewählte Aktivitätsart ("Alle" oder spezifische Art).
        - people_count: Ausgewählte Personenkategorie ("Alle", "Alleine", etc.).
        - budget: Ausgewähltes maximales Budget (Zahl) oder None.
        - reset_llm_pressed: Boolean, ob der "Manuelle Filter verwenden"-Button gedrückt wurde.
    """
    with st.sidebar: # Alles innerhalb dieses Blocks erscheint in der Sidebar
        # --- Logo ---
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=200) # Zeige Logo, wenn Datei existiert

        # --- Datumsauswahl ---
        st.header("🗓️ Datum & Filter")
        selected_date = st.date_input(
            "Datum wählen", value=today_date, min_value=today_date, key="date_input_sidebar"
        )

        # --- Wetterfilter Checkbox ---
        st.header("⚙️ Aktivitäten filtern")
        consider_weather = st.checkbox(
            "Nach Wetter filtern", value=True, # Standardmäßig aktiviert
            help="Filtert Aktivitäten basierend auf der Wettervorhersage und ihrer Präferenz.",
            disabled=not openweathermap_api_configured, # Deaktivieren, wenn kein Wetter-API-Key da ist
            key="sb_wetter" # Eindeutiger Schlüssel für dieses Widget
        )

        # --- Button, um vom LLM-Modus zurück zu manuellen Filtern zu wechseln ---
        reset_llm_pressed = False
        # Zeige Button nur an, wenn gerade LLM-Ergebnisse angezeigt werden (erkennbar am Session State)
        if st.session_state.get(STATE_SHOW_LLM_RESULTS):
            if st.button("Manuelle Filter verwenden", key="btn_reset_llm_sidebar"):
                reset_llm_pressed = True # Merken, dass der Button geklickt wurde

        # --- Manuelle Filter Widgets (Dropdowns, Slider) ---
        # Initialisiere Standardwerte
        activity_type = "Alle"; budget = None

        # Zeige Filter nur an, wenn Aktivitätsdaten vorhanden sind
        if not df.empty:
            # Aktivitätsart Filter (Dropdown)
            if COL_ART in df.columns:
                try:
                    # Hole alle einzigartigen Aktivitätsarten aus den Daten, sortiere sie
                    unique_arts = sorted(df[COL_ART].dropna().astype(str).unique())
                    # Erstelle die Optionen für das Dropdown ("Alle" + gefundene Arten)
                    activity_types_options = ["Alle"] + unique_arts
                except Exception as e:
                    # Fallback, falls etwas beim Holen der Arten schiefgeht
                    # print(f"Fehler beim Extrahieren der Aktivitätsarten: {e}") # Debug
                    activity_types_options = ["Alle"]
                # Erstelle das Dropdown-Widget
                activity_type = st.selectbox("Aktivitätsart", options=activity_types_options, key="sb_art")
            else:
                 st.caption(f"Spalte '{COL_ART}' fehlt."); st.selectbox("Aktivitätsart", ["Alle"], disabled=True, key="sb_art_disabled")

            # Budget Filter (Slider)
            # Prüfe, ob die Preis-Spalte existiert und mindestens einen Wert enthält
            if COL_PREIS in df.columns and df[COL_PREIS].notna().any():
                # Fülle fehlende Preise mit 0 für Min/Max-Berechnung
                preis_col_filled = df[COL_PREIS].fillna(0)
                min_price_default, max_price_default = 0, 150 # Standard-Range für Slider
                try:
                    # Finde den minimalen und maximalen Preis in den Daten
                    min_val = preis_col_filled.min(); max_val = preis_col_filled.max()
                    # Setze sichere Grenzen für den Slider
                    min_price = int(min_val) if pd.notna(min_val) and min_val >= 0 else min_price_default
                    max_price = int(max_val) if pd.notna(max_val) and max_val > 0 else max_price_default
                    if min_price > max_price: min_price = max_price # Sicherstellen, dass min <= max
                    # Erstelle den Slider; Standardwert ist der Maximalpreis
                    budget = st.slider("Max. Budget (CHF)", min_value=min_price, max_value=max_price, value=max_price, key="sb_budget")
                except Exception as e:
                    # Fallback, wenn bei Preisberechnung Fehler auftreten
                    st.warning(f"Fehler bei Budget-Slider: {e}. Verwende Standard.")
                    budget = st.slider("Max. Budget (CHF)", min_price_default, max_price_default, max_price_default, key="sb_budget_error")
            else:
                 st.caption("Keine Preisdaten für Budget-Filter verfügbar."); st.slider("Max. Budget (CHF)", 0, 1, 0, disabled=True, key="sb_budget_disabled_fallback"); budget = None
        else: # Fall: df ist leer
             st.warning("Keine Aktivitätsdaten geladen. Filter deaktiviert.")
             # Zeige deaktivierte Filter-Widgets an
             st.selectbox("Aktivitätsart", ["Alle"], disabled=True, key="sb_art_disabled_empty")
             st.slider("Max. Budget (CHF)", 0, 1, 0, disabled=True, key="sb_budget_disabled_empty")

    # Gib die ausgewählten Werte zurück, damit sie in app.py verwendet werden können
    return selected_date, consider_weather, activity_type, budget, reset_llm_pressed


def display_map(
    df_for_map: pd.DataFrame,
    selected_activity_id: Optional[int] = None # Optional: ID der Aktivität, die hervorgehoben werden soll
    ) -> None:
    """
    Erstellt und zeigt eine interaktive Folium-Karte mit Aktivitäts-Markern an.

    Zeigt die Aktivitäten aus dem übergebenen DataFrame als Marker auf einer Karte.
    Hebt optional eine ausgewählte Aktivität hervor und zentriert die Karte darauf.

    Args:
        df_for_map (pd.DataFrame): DataFrame mit den Aktivitäten, die angezeigt werden sollen
                                   (muss COL_LAT, COL_LON, COL_ID, COL_NAME enthalten).
        selected_activity_id (Optional[int]): ID der Aktivität, die auf der Karte
                                            hervorgehoben und zentriert werden soll.
    """
    st.subheader("Karte der Aktivitäten")
    # Standard-Kartenzentrum (St. Gallen) und Zoomstufe
    map_center = [ST_GALLEN_LAT, ST_GALLEN_LON]; map_zoom = 11

    # Prüfe, ob Daten und benötigte Spalten vorhanden sind
    required_map_cols = [COL_LAT, COL_LON, COL_ID, COL_NAME]
    if df_for_map.empty or not all(col in df_for_map.columns for col in required_map_cols):
         st.warning("Karte kann nicht angezeigt werden: Daten oder notwendige Spalten (Lat/Lon/ID/Name) fehlen.")
         # Zeige leere Karte als Fallback
         m = folium.Map(location=map_center, zoom_start=map_zoom, tiles='CartoDB positron')
         # Zeige die Folium-Karte in Streamlit an
         st_folium(m, height=500, key="folium_map_empty", use_container_width=True, returned_objects=[])
         return # Beende die Funktion hier

    # Berechne das Zentrum und den Zoom basierend auf den angezeigten Aktivitäten
    valid_coords = df_for_map[[COL_LAT, COL_LON]].dropna() # Ignoriere Aktivitäten ohne Koordinaten
    if not valid_coords.empty:
        try:
             # Zentriere Karte auf den Durchschnitt der Koordinaten
             map_center = [valid_coords[COL_LAT].mean(), valid_coords[COL_LON].mean()]
             # Passe Zoomstufe an Anzahl der Aktivitäten an
             num_activities = len(valid_coords)
             if num_activities == 1: map_zoom = 14 # Näher ran bei nur einer Aktivität
             elif num_activities < 10: map_zoom = 12
             else: map_zoom = 11 # Weiter weg bei vielen Aktivitäten
        except Exception as e:
            # print(f"Warnung: Fehler bei Berechnung des Kartenzentrums/Zooms: {e}") # Debug
            pass # Verwende Standardwerte im Fehlerfall

    # Wenn eine Aktivität ausgewählt ist, zentriere die Karte darauf und zoome näher heran
    if selected_activity_id is not None:
        selected_row = df_for_map[df_for_map[COL_ID] == selected_activity_id]
        if not selected_row.empty:
            try:
                sel_lat = selected_row.iloc[0][COL_LAT]; sel_lon = selected_row.iloc[0][COL_LON]
                if pd.notna(sel_lat) and pd.notna(sel_lon):
                     map_center = [float(sel_lat), float(sel_lon)]; map_zoom = 15 # Starker Zoom auf Auswahl
            except Exception as e:
                # print(f"Warnung: Fehler beim Zentrieren auf ID {selected_activity_id}: {e}") # Debug
                pass # Verwende vorher berechnete Werte im Fehlerfall

    # Erstelle das Folium-Kartenobjekt
    # 'CartoDB positron' ist ein heller, unaufdringlicher Kartenstil
    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles='CartoDB positron')

    # Füge für jede Aktivität einen Marker zur Karte hinzu
    for index, row in df_for_map.iterrows():
        # Hole Daten für den Marker (verwende .get() für Sicherheit gegen fehlende Spalten)
        lat = row.get(COL_LAT); lon = row.get(COL_LON); activity_id = row.get(COL_ID)
        name = row.get(COL_NAME, 'N/A'); art = row.get(COL_ART, 'N/A'); ort = row.get(COL_ORT, 'N/A')
        weather_note = row.get('weather_note') # Wetterhinweis aus logic.py

        # Füge Marker nur hinzu, wenn Koordinaten gültig sind
        if pd.notna(lat) and pd.notna(lon):
            try:
                # Style für den Marker: Rot/Stern für ausgewählte, Blau/Info für andere
                is_selected = (activity_id == selected_activity_id)
                icon_color = 'red' if is_selected else 'blue'
                icon_symbol = 'star' if is_selected else 'info-sign'
                # Text, der beim Hovern erscheint (Tooltip)
                tooltip_text = f"<b>{name}</b>"
                # Text/HTML, das im Popup erscheint, wenn man klickt
                popup_html = f"<b>{name}</b><br>Art: {art}<br>Ort: {ort}"
                # Füge Wetterhinweis zum Tooltip hinzu, falls vorhanden
                if pd.notna(weather_note): tooltip_text += f"<br>{weather_note}"

                # Erstelle den Marker und füge ihn zur Karte 'm' hinzu
                folium.Marker(
                    location=[float(lat), float(lon)],
                    popup=folium.Popup(popup_html, max_width=250), # Begrenze Popup-Breite
                    tooltip=tooltip_text,
                    icon=folium.Icon(color=icon_color, icon=icon_symbol, prefix='glyphicon') # Bootstrap Glyphs
                ).add_to(m)
            except Exception as e:
                # print(f"Warnung: Fehler beim Hinzufügen des Markers für ID {activity_id} ({name}): {e}") # Debug
                pass # Überspringe Marker im Fehlerfall

    # Zeige die fertige Folium-Karte in der Streamlit-App an
    st_folium(m, height=500, key="folium_map_main", use_container_width=True, returned_objects=[])


def display_weather_overview(
    location_name: str, # Name des Ortes (z.B. "St. Gallen")
    target_date: Optional[datetime.date], # Das Datum, für das die Übersicht gilt
    forecast_list: Optional[List[Dict[str, Any]]], # Liste der Wettervorhersagen (von weather_utils)
    api_configured: bool # Ist der Wetter-API Key konfiguriert?
    ) -> None:
    """
    Zeigt eine Wetterübersicht für einen bestimmten Ort und Tag an.

    Stellt die repräsentative Wetterlage (Temperatur, Beschreibung, Icon) dar
    und optional einen Temperatur-Tagesverlauf als Liniendiagramm sowie eine
    kleine Tabelle mit stündlichen Vorhersagen.

    Args:
        location_name (str): Name des Ortes für die Überschrift.
        target_date (Optional[datetime.date]): Das Zieldatum für die Übersicht.
        forecast_list (Optional[List[Dict[str, Any]]]): Die Wetterdatenliste von
                                                     `get_weather_forecast_for_day`.
        api_configured (bool): Ob der OpenWeatherMap API Key konfiguriert ist.
    """
    st.subheader(f"Wetterübersicht {location_name}")
    if not target_date:
        st.markdown("###### Wähle ein Datum in der Sidebar."); return

    st.markdown(f"###### {target_date.strftime('%A, %d. %B %Y')}") # Zeige Datum formatiert an

    # Finde eine repräsentative Vorhersage für die Hauptanzeige (Temperatur/Icon)
    # (Logik ähnlich wie in weather_utils.check_activity_weather_status)
    representative_forecast = None
    if forecast_list:
        try:
            valid_forecasts = [f for f in forecast_list if isinstance(f.get('datetime'), datetime.datetime)]
            if valid_forecasts:
                representative_forecast = next((f for f in valid_forecasts if f['datetime'].hour >= 12), valid_forecasts[0])
        except Exception: pass # Ignoriere Fehler bei der Auswahl

    # Zeige Haupt-Wetterinfo (Temp/Icon/Beschreibung)
    if representative_forecast:
        temp = representative_forecast.get('temp'); desc = representative_forecast.get('description'); icon = representative_forecast.get('icon')
        # Formatiere Werte für Anzeige (mit Fallbacks für fehlende Daten)
        temp_str = f"{temp:.1f}°C" if pd.notna(temp) else "N/A"
        desc_str = desc.capitalize() if pd.notna(desc) else "N/A"
        icon_url = f"http://openweathermap.org/img/wn/{icon}@2x.png" if pd.notna(icon) else None
        # Zeige Icon und Metrik nebeneinander an
        subcol1, subcol2 = st.columns([0.3, 0.7])
        with subcol1:
            if icon_url: st.image(icon_url, width=60)
        with subcol2:
            # st.metric zeigt eine Zahl groß an, mit einem Label darüber
            st.metric(label=desc_str, value=temp_str)
    elif api_configured:
        # Zeige Meldung, wenn API konfiguriert ist, aber keine Daten da sind
        st.info(f"Wetterdaten für {location_name} am {target_date.strftime('%d.%m.')} nicht verfügbar.")
    elif location_name:
        # Zeige Meldung, wenn API nicht konfiguriert ist
        st.caption("Kein OpenWeatherMap API Key konfiguriert.")

    # Zeige optional den Temperatur-Tagesverlauf und eine stündliche Tabelle
    if forecast_list:
        st.markdown(f"###### Tagesverlauf (UTC)")
        try:
            # Bereite Daten für das Liniendiagramm vor
            forecast_df_display = pd.DataFrame(forecast_list)
            # Prüfe, ob benötigte Spalten da sind und Daten enthalten
            if not forecast_df_display.empty and 'datetime' in forecast_df_display.columns and 'temp' in forecast_df_display.columns and forecast_df_display['temp'].notna().any():
                # Wandle 'datetime' sicher um und setze es als Index für das Diagramm
                forecast_df_display['datetime'] = pd.to_datetime(forecast_df_display['datetime'], errors='coerce', utc=True)
                forecast_df_display.dropna(subset=['datetime', 'temp'], inplace=True) # Entferne Zeilen mit fehlenden Werten
                if not forecast_df_display.empty:
                    forecast_df_display.set_index('datetime', inplace=True)
                    # Zeige das Liniendiagramm nur für die Temperatur
                    st.line_chart(forecast_df_display[['temp']], use_container_width=True, height=150)

                    # Erstelle eine kleine Tabelle mit Vorhersagen zu bestimmten Zeiten (z.B. 9, 12, 15, 18 Uhr UTC)
                    st.caption("Vorhersage-Überblick (ca.):")
                    target_hours_utc = [9, 12, 15, 18]; hourly_summary = []; processed_times = set()
                    for hour_utc in target_hours_utc:
                        # Finde die Vorhersage, die zeitlich am nächsten zur Zielstunde liegt
                        target_time_utc = datetime.datetime.combine(target_date, datetime.time(hour_utc), tzinfo=datetime.timezone.utc)
                        try:
                            # Finde den Index der nächsten Zeit im DataFrame
                            nearest_index_loc = forecast_df_display.index.get_indexer([target_time_utc], method='nearest')[0]
                            closest_forecast_row = forecast_df_display.iloc[nearest_index_loc]
                            actual_time_utc = closest_forecast_row.name # Echte Zeit der Vorhersage
                            forecast_time_str = actual_time_utc.strftime('%H:%M')
                            # Verhindere doppelte Einträge, falls mehrere Zielstunden auf dieselbe Vorhersage zeigen
                            if forecast_time_str not in processed_times:
                                temp_val = closest_forecast_row.get('temp')
                                desc_val = closest_forecast_row.get('description')
                                # Füge formatierte Daten zur Liste für die Tabelle hinzu
                                hourly_summary.append({
                                    "Zeit (UTC)": forecast_time_str,
                                    "°C": f"{temp_val:.0f}°" if pd.notna(temp_val) else "?",
                                    "Wetter": str(desc_val).capitalize() if pd.notna(desc_val) else "N/A"
                                })
                                processed_times.add(forecast_time_str)
                        except Exception: pass # Ignoriere Fehler beim Finden einzelner Stunden

                    # Zeige die zusammengefasste Tabelle an, wenn Daten vorhanden sind
                    summary_df_display = pd.DataFrame(hourly_summary)
                    if not summary_df_display.empty:
                         st.dataframe(
                             summary_df_display,
                             use_container_width=True,
                             hide_index=True,
                             # Konfiguriere Spaltenbreiten etc. für bessere Darstellung
                             column_config={
                                 "Zeit (UTC)": st.column_config.TextColumn("Zeit (UTC)", width="small"),
                                 "°C": st.column_config.TextColumn("Temp.", width="small"),
                                 "Wetter": st.column_config.TextColumn("Beschreibung", width="large")
                             }
                         )
        except Exception as e:
             # Catchall für Fehler bei der Diagramm/Tabellen-Erstellung
             st.error(f"Fehler bei Darstellung des Wetter-Tagesverlaufs ({location_name}): {e}")


# In ui_components.py

def display_activity_details(
    activity_row: pd.Series, # Eine Zeile aus dem Aktivitäten-DataFrame
    activity_id: int, # Die ID der Aktivität
    is_expanded: bool, # Soll der Expander standardmäßig geöffnet sein?
    openweathermap_api_configured: bool, # Ist Wetter-API konfiguriert?
    key_prefix: str = "" # Ein Präfix für eindeutige Widget-Keys
    ) -> None:
    """
    Zeigt die Details einer einzelnen Aktivität in einem aufklappbaren Bereich (Expander).

    Stellt formatierte Informationen zur Aktivität dar, inklusive Wetter am Standort
    (falls verfügbar), Buttons für externe Links (Website, Maps) und einen Button,
    um diese Aktivität auf der Karte zu fokussieren. (Layout angepasst an ursprüngliche Version).

    Args:
        activity_row (pd.Series): Die Datenzeile der anzuzeigenden Aktivität.
        activity_id (int): Die ID der Aktivität (wichtig für Keys und State).
        is_expanded (bool): Ob der Expander standardmäßig geöffnet sein soll.
        openweathermap_api_configured (bool): Flag für Wetter-API-Status.
        key_prefix (str): Präfix, um sicherzustellen, dass die Keys der Buttons etc.
                          innerhalb des Expanders eindeutig sind (z.B. "llm" oder "filter").
    """
    # --- Daten sicher aus der Reihe extrahieren ---
    # (Dieser Teil bleibt unverändert)
    name = activity_row.get(config.COL_NAME, 'Unbekannt')
    art_str = activity_row.get(config.COL_ART, "Typ N/A")
    ort_str = activity_row.get(config.COL_ORT, "Ort N/A")
    preis_ca_val = activity_row.get(config.COL_PREIS)
    beschreibung_str = activity_row.get(config.COL_BESCHREIBUNG, "Keine Beschreibung verfügbar.")
    image_url = activity_row.get(config.COL_IMAGE_URL)
    website_str = activity_row.get(config.COL_WEBSITE)
    latitude = activity_row.get(config.COL_LAT); longitude = activity_row.get(config.COL_LON)
    dauer_str = activity_row.get(config.COL_DAUER_INFO, "N/A")
    zielgruppe_str = activity_row.get(config.COL_ZIELGRUPPE)
    indoor_outdoor_str = activity_row.get(config.COL_INDOOR_OUTDOOR, "N/A")
    wetter_pref_str = activity_row.get(config.COL_WETTER_PREF, "N/A")
    von_datum = activity_row.get(config.COL_DATUM_VON); bis_datum = activity_row.get(config.COL_DATUM_BIS)
    booking_str = activity_row.get(config.COL_BOOKING_INFO)
    kontakt_str = activity_row.get(config.COL_KONTAKT_TEL)
    loc_temp_val = activity_row.get('location_temp')
    loc_desc_val = activity_row.get('location_desc')
    loc_icon_val = activity_row.get('location_icon')
    weather_note = activity_row.get('weather_note')

    # --- Werte für die Anzeige formatieren ---
    if pd.notna(preis_ca_val):
        if preis_ca_val > 0: preis_str = f"{preis_ca_val:.2f} CHF"
        else: preis_str = "Kostenlos"
    else: preis_str = "Preis N/A"
    try:
        von_str = von_datum.strftime('%d.%m.%Y') if pd.notna(von_datum) else None
        bis_str = bis_datum.strftime('%d.%m.%Y') if pd.notna(bis_datum) else None
        if von_str and bis_str: verfuegbar_str = f"{von_str} - {bis_str}"
        elif von_str: verfuegbar_str = f"Ab {von_str}"
        elif bis_str: verfuegbar_str = f"Bis {bis_str}"
        else: verfuegbar_str = "Ganzjährig verfügbar"
    except Exception: verfuegbar_str = "Datumsinfo fehlerhaft"

    # --- Expander erstellen ---
    expander_label = f"{name} ({art_str} in {ort_str} | {preis_str})"
    with st.expander(expander_label, expanded=is_expanded):

        # *** Layout Block 1: Wetter (links) und Buttons (rechts) ***
        col1, col2 = st.columns([1, 1]) # Zwei gleich breite Spalten

        with col1: # Linke Spalte für Wetter
            st.markdown("**Wetter am Standort (ca. Mittag):**")
            if pd.notna(loc_temp_val) and pd.notna(loc_desc_val):
                loc_temp_str = f"{loc_temp_val:.1f}°C"
                loc_icon_url = f"http://openweathermap.org/img/wn/{loc_icon_val}.png" if pd.notna(loc_icon_val) else None
                w_col1, w_col2 = st.columns([0.2, 0.8])
                with w_col1:
                    if loc_icon_url: st.image(loc_icon_url, width=40)
                with w_col2:
                    st.write(f"{loc_temp_str}, {loc_desc_val}")
            elif pd.notna(weather_note) and "Koordinaten fehlen" in weather_note:
                st.caption(weather_note)
            elif openweathermap_api_configured:
                st.caption("Wetterdaten für Standort nicht verfügbar.")
            if pd.notna(weather_note) and "Koordinaten fehlen" not in weather_note:
                st.warning(weather_note)

        with col2: # Rechte Spalte für Buttons
             st.markdown("**Aktionen:**") # Überschrift hinzugefügt für Konsistenz
             # Buttons nebeneinander in 3 (Unter-)Spalten
             btn_cols = st.columns(3)
             with btn_cols[0]: # Fokus-Button
                 focus_key = f"btn_focus_{key_prefix}_{activity_id}"
                 if st.button("📍 Fokus", key=focus_key, help="Auf Karte fokussieren"):
                     st.session_state[config.STATE_SELECTED_ACTIVITY_INDEX] = activity_id
                     st.rerun()
             with btn_cols[1]: # Web-Button
                 if pd.notna(website_str) and isinstance(website_str, str) and website_str.startswith('http'):
                     st.link_button("🌐 Web", website_str, help="Website besuchen")
                 else:
                     st.button("🌐 Web", disabled=True, help="Keine Website verfügbar", key=f"btn_web_disabled_{key_prefix}_{activity_id}")
             with btn_cols[2]: # Maps-Button
                 if pd.notna(latitude) and pd.notna(longitude):
                     try:
                         # Verwendung des Standard-Google Maps Links
                         maps_url = f"https://www.google.com/maps/search/?api=1&query={float(latitude)},{float(longitude)}"
                         st.link_button("🗺️ Maps", maps_url, help="Auf Google Maps anzeigen")
                     except (ValueError, TypeError):
                         st.button("🗺️ Maps", disabled=True, help="Ungültige Koordinaten", key=f"btn_maps_disabled_invalid_{key_prefix}_{activity_id}")
                 else:
                     st.button("🗺️ Maps", disabled=True, help="Keine Koordinaten verfügbar", key=f"btn_maps_disabled_missing_{key_prefix}_{activity_id}")

        st.markdown("---") # Trennlinie

        # *** Layout Block 2: Bild (links) und Beschreibung (rechts) ***
        # Verwende Verhältnis 1:2, damit Beschreibung mehr Platz hat
        col3, col4 = st.columns([1, 2])

        with col3: # Linke, schmalere Spalte für Bild
            if pd.notna(image_url) and isinstance(image_url, str) and image_url.startswith('http'):
                st.image(image_url, caption=f"Bild: {name}", use_container_width=True)
            else:
                st.caption("Kein Bild verfügbar")

        with col4: # Rechte, breitere Spalte für Beschreibung
            st.markdown(f"**Beschreibung:**")
            st.markdown(beschreibung_str if pd.notna(beschreibung_str) else "_Keine Beschreibung vorhanden._")

        st.markdown("---") # Trennlinie

        # *** Layout Block 3: Weitere Details ***
        # Aufgeteilt in zwei gleich breite Spalten
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1: # Linke Spalte für Details
            st.markdown(f"**Art:** {art_str}")
            st.markdown(f"**Ort:** {ort_str}")
            # Optional: Adresse anzeigen, falls vorhanden
            # adresse_str = activity_row.get(config.COL_ADRESSE)
            # if pd.notna(adresse_str): st.markdown(f"**Adresse:** {adresse_str}")
            st.markdown(f"**Typ:** {indoor_outdoor_str}")
            st.markdown(f"**Bevorzugtes Wetter:** {wetter_pref_str}")
            # Zielgruppe wieder hier anzeigen (wie im Original?)
            if pd.notna(zielgruppe_str):
                st.markdown(f"**Zielgruppe:** {zielgruppe_str}")

        with detail_col2: # Rechte Spalte für Details
            st.markdown(f"**Dauer ca.:** {dauer_str}")
            st.markdown(f"**Verfügbar:** {verfuegbar_str}")
            # Zeige Buchungsinfo und Telefon nur an, wenn vorhanden
            if pd.notna(booking_str):
                st.markdown(f"**Buchung:** {booking_str}")
            if pd.notna(kontakt_str):
                st.markdown(f"**Tel:** {kontakt_str}")

def display_recommendation_card(
    activity_row: pd.Series, # Daten der Aktivität
    card_key_suffix: str, # Suffix für eindeutige Button-Keys
    on_like_callback: Callable[[int, int], None], # Funktion, die bei Klick aufgerufen wird (erwartet ID und Rating)
    on_dislike_callback: Callable[[int, int], None] # Funktion, die bei Klick aufgerufen wird (erwartet ID und Rating)
    ) -> None:
    """
    Zeigt eine Aktivität als "Karte" mit Bild, Text und Like/Dislike-Buttons an.

    Wird im Personalisierungs-Expander verwendet, um dem Nutzer Aktivitäten
    zum Bewerten vorzuschlagen.

    Args:
        activity_row (pd.Series): Die Datenzeile der anzuzeigenden Aktivität.
        card_key_suffix (str): Ein Suffix, um die Keys der Buttons eindeutig zu machen.
        on_like_callback (Callable[[int, int], None]): Die Funktion (aus app.py), die
                                                ausgeführt werden soll, wenn der
                                                Like-Button geklickt wird. Erwartet ID und Rating (1).
        on_dislike_callback (Callable[[int, int], None]): Die Funktion (aus app.py), die
                                                   ausgeführt werden soll, wenn der
                                                   Dislike-Button geklickt wird. Erwartet ID und Rating (-1).
    """
    # Hole benötigte Daten aus der Zeile
    activity_id = activity_row.get(config.COL_ID)
    name = activity_row.get(config.COL_NAME, "Unbekannte Aktivität")
    image_url = activity_row.get(config.COL_IMAGE_URL)
    description = activity_row.get(config.COL_BESCHREIBUNG, "Keine Beschreibung.")

    # Stelle sicher, dass die ID eine gültige Zahl ist für die Callback-Funktionen
    try:
        activity_id_for_callback = int(activity_id)
        button_disabled = False # Buttons aktivieren
    except (ValueError, TypeError):
        print(f"WARNUNG (RecCard): Ungültige ID '{activity_id}'. Buttons deaktiviert.") # Log für Entwickler
        activity_id_for_callback = -1 # Ungültiger Wert
        button_disabled = True # Buttons deaktivieren

    # Kürze lange Beschreibungen für die Kartenansicht
    max_desc_length = 100
    short_description = description
    if pd.notna(description) and len(description) > max_desc_length:
        short_description = description[:max_desc_length].strip() + "..."

    # Erstelle einen Container mit Rahmen für die Karte
    with st.container(border=True):
        # Zeige Bild, wenn vorhanden
        if pd.notna(image_url) and isinstance(image_url, str) and image_url.startswith('http'):
            st.image(image_url, use_container_width=True)
        else:
            st.caption("Kein Bild") # Platzhalter
        # Zeige Name und gekürzte Beschreibung
        st.subheader(name)
        st.markdown(short_description)
        st.markdown("---") # Trennlinie
        # Zeige Like/Dislike Buttons nebeneinander
        col1, col2 = st.columns(2)
        with col1:
            # Wichtig: on_click übergibt die Callback-Funktion. args übergibt die Argumente.
            st.button("👍 Gefällt mir",
                      key=f"btn_like_{card_key_suffix}_{activity_id}", # Eindeutiger Key
                      use_container_width=True, # Button füllt Spaltenbreite
                      on_click=on_like_callback, # Funktion, die bei Klick aufgerufen wird
                      args=(activity_id_for_callback, 1), # *** KORRIGIERT: ID und Rating 1 übergeben ***
                      disabled=button_disabled) # Deaktivieren bei ungültiger ID
        with col2:
            st.button("👎 Eher nicht",
                      key=f"btn_dislike_{card_key_suffix}_{activity_id}",
                      use_container_width=True,
                      on_click=on_dislike_callback,
                      args=(activity_id_for_callback, -1), # *** KORRIGIERT: ID und Rating -1 übergeben ***
                      disabled=button_disabled)

def display_preference_visualization(
    profile_label: Optional[str], # Das generierte Profil-Label (z.B. "Kultur-Fan")
    preference_scores_art: Optional[pd.Series], # Gezählte Likes pro Aktivitätsart
    top_target_groups: Optional[pd.Series], # Gezählte Likes pro Zielgruppe
    liked_prices_list: Optional[List[float]] # Liste der Preise geliketer Aktivitäten
    ) -> None:
    """
    Zeigt eine Visualisierung der gelernten Nutzerpräferenzen an.

    Stellt das Profil-Label, ein Kuchendiagramm für bevorzugte Aktivitätsarten,
    ein Balkendiagramm für bevorzugte Zielgruppen und ein Histogramm für die
    Preisverteilung der gelikten Aktivitäten dar.

    Args:
        profile_label (Optional[str]): Das Text-Label für das Profil.
        preference_scores_art (Optional[pd.Series]): Series mit Aktivitätsarten als Index
                                                     und Anzahl Likes als Werte.
        top_target_groups (Optional[pd.Series]): Series mit Zielgruppen als Index
                                                  und Anzahl Likes als Werte.
        liked_prices_list (Optional[List[float]]): Liste der Preise der gelikten Aktivitäten.
    """
    # Zeige das generierte Profil-Label an, falls vorhanden
    if profile_label:
        st.info(f"Dein gelerntes Profil: **{profile_label}**")

    # Layout mit zwei Spalten für die ersten beiden Diagramme
    col1, col2 = st.columns(2)

    # Linke Spalte: Kuchendiagramm für Aktivitätsarten
    with col1:
        st.markdown("**Bevorzugte Aktivitätsarten:**")
        # Prüfe, ob Daten für das Diagramm vorhanden sind
        if preference_scores_art is None or preference_scores_art.empty:
            st.caption("Bewerte Aktivitäten 👍, um deine bevorzugten Arten zu sehen.")
        else:
            # Zeige nur Arten mit mindestens einem Like im Diagramm
            scores_to_display = preference_scores_art[preference_scores_art > 0]
            if scores_to_display.empty:
                st.caption("Bewerte Aktivitäten 👍, um deine bevorzugten Arten zu sehen.")
            else:
                try:
                    # Erstelle das Plotly Kuchendiagramm
                    fig_pie = go.Figure(data=[go.Pie(labels=scores_to_display.index,
                                                     values=scores_to_display.values,
                                                     hole=.3)]) # Kleines Loch in der Mitte
                    # Konfiguriere Layout (Größe, Ränder, Legende ausblenden)
                    fig_pie.update_layout(margin=dict(l=10, r=10, t=30, b=20), showlegend=False, height=250)
                    # Zeige Prozent und Label im Diagramm an
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    # Zeige das Diagramm in Streamlit an
                    st.plotly_chart(fig_pie, use_container_width=True)
                except Exception as e:
                    st.error(f"Fehler beim Erstellen des Kuchendiagramms: {e}")

    # Rechte Spalte: Balkendiagramm für Zielgruppen
    with col2:
        st.markdown("**Bevorzugte Zielgruppen:**")
        # Prüfe, ob Daten vorhanden sind
        if top_target_groups is None or top_target_groups.empty:
            st.caption("Bewerte Aktivitäten 👍, um deine bevorzugten Zielgruppen zu sehen.")
        else:
            try:
                # Zeige einfaches Balkendiagramm mit Streamlit Bordmitteln
                st.bar_chart(top_target_groups, height=230)
            except Exception as e:
                st.error(f"Fehler beim Erstellen des Zielgruppen-Diagramms: {e}")

    # Unter den Spalten: Histogramm für Preise
    st.markdown("**Preisverteilung deiner Favoriten:**")
    # Prüfe, ob Preisdaten vorhanden sind
    if liked_prices_list is None or not liked_prices_list:
        st.caption("Bewerte kostenpflichtige Aktivitäten 👍, um die Preisverteilung zu sehen.")
    else:
        try:
            # Erstelle das Plotly Histogramm
            fig_hist = go.Figure(data=[go.Histogram(x=liked_prices_list, nbinsx=10)]) # 10 Balken
            # Konfiguriere Layout (Achsenbeschriftung, Höhe, Ränder)
            fig_hist.update_layout(xaxis_title_text='Preis (CHF)', yaxis_title_text='Anzahl',
                                   bargap=0.1, height=250, margin=dict(l=10, r=10, t=10, b=20))
            # Zeige das Diagramm in Streamlit an
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e:
            st.error(f"Fehler beim Erstellen des Preis-Histogramms: {e}")

    st.markdown("---") # Trennlinie am Ende des Expanders