# ui_components.py
"""
Funktionen zur Erstellung von UI-Komponenten für die HSGexplorer Streamlit App.

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
    # Importiere spezifische Konstanten für Lesbarkeit
    from config import (
        LOGO_PATH, STATE_SHOW_LLM_RESULTS, STATE_SELECTED_ACTIVITY_INDEX,
        ST_GALLEN_LAT, ST_GALLEN_LON, COL_ID, COL_ART, COL_PREIS, COL_ORT,
        COL_NAME, COL_LAT, COL_LON, COL_WETTER_PREF, COL_DATUM_VON, COL_DATUM_BIS,
        COL_BESCHREIBUNG, COL_PERSONEN_MIN, COL_PERSONEN_MAX, COL_ZIELGRUPPE,
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

# Die Funktion get_recommendations wird hier nicht mehr benötigt

def display_sidebar(
    df: pd.DataFrame,
    today_date: datetime.date,
    openweathermap_api_configured: bool
    ) -> Tuple[datetime.date, bool, str, str, Optional[float], bool]:
    """
    Zeigt die Streamlit Sidebar an und gibt die ausgewählten Filterwerte zurück.
    (Implementierung wie zuvor)
    """
    with st.sidebar:
        # --- Logo ---
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=200)

        # --- Datumsauswahl ---
        st.header("🗓️ Datum & Filter")
        selected_date = st.date_input(
            "Datum wählen", value=today_date, min_value=today_date, key="date_input_sidebar"
        )

        # --- Wetterfilter Checkbox ---
        st.header("⚙️ Aktivitäten filtern")
        consider_weather = st.checkbox(
            "Nach Wetter filtern", value=True,
            help="Filtert Aktivitäten basierend auf der Wettervorhersage und ihrer Präferenz.",
            disabled=not openweathermap_api_configured, key="sb_wetter"
        )

        # --- Reset Button für LLM ---
        reset_llm_pressed = False
        if st.session_state.get(STATE_SHOW_LLM_RESULTS):
            if st.button("Manuelle Filter verwenden", key="btn_reset_llm_sidebar"): # Key hinzugefügt
                reset_llm_pressed = True

        # --- Manuelle Filter Widgets ---
        activity_type = "Alle"; people_count = "Alle"; budget = None

        if not df.empty:
            # Aktivitätsart Filter
            if COL_ART in df.columns:
                try:
                    unique_arts = sorted(df[COL_ART].dropna().astype(str).unique())
                    activity_types_options = ["Alle"] + unique_arts
                except Exception as e:
                    print(f"Fehler beim Extrahieren der Aktivitätsarten: {e}")
                    activity_types_options = ["Alle"]
                activity_type = st.selectbox("Aktivitätsart", options=activity_types_options, key="sb_art")
            else:
                 st.caption(f"Spalte '{COL_ART}' fehlt."); st.selectbox("Aktivitätsart", ["Alle"], disabled=True, key="sb_art_disabled")

            # Personen Filter
            personen_options = ["Alle", "Alleine", "Zu zweit", "Bis 4 Personen", "Mehr als 4 Personen"]
            people_count = st.selectbox("Anzahl Personen", options=personen_options, key="sb_personen")

            # Budget Filter
            if COL_PREIS in df.columns and df[COL_PREIS].notna().any():
                preis_col_filled = df[COL_PREIS].fillna(0)
                min_price_default, max_price_default = 0, 150
                try:
                    min_val = preis_col_filled.min(); max_val = preis_col_filled.max()
                    min_price = int(min_val) if pd.notna(min_val) and min_val >= 0 else min_price_default
                    max_price = int(max_val) if pd.notna(max_val) and max_val > 0 else max_price_default
                    if min_price > max_price: min_price = max_price
                    budget = st.slider("Max. Budget (CHF)", min_value=min_price, max_value=max_price, value=max_price, key="sb_budget")
                except Exception as e:
                    st.warning(f"Fehler bei Budget-Slider: {e}. Verwende Standard.")
                    budget = st.slider("Max. Budget (CHF)", min_price_default, max_price_default, max_price_default, key="sb_budget_error")
            else:
                 st.caption("Keine Preisdaten für Budget-Filter verfügbar."); st.slider("Max. Budget (CHF)", 0, 1, 0, disabled=True, key="sb_budget_disabled_fallback"); budget = None
        else: # Fall: df ist leer
             st.warning("Keine Aktivitätsdaten geladen. Filter deaktiviert.")
             st.selectbox("Aktivitätsart", ["Alle"], disabled=True, key="sb_art_disabled_empty")
             st.selectbox("Anzahl Personen", ["Alle"], disabled=True, key="sb_personen_disabled_empty")
             st.slider("Max. Budget (CHF)", 0, 1, 0, disabled=True, key="sb_budget_disabled_empty")

    return selected_date, consider_weather, activity_type, people_count, budget, reset_llm_pressed


def display_map(
    df_for_map: pd.DataFrame,
    selected_activity_id: Optional[int] = None
    ) -> None:
    """
    Erstellt und zeigt eine interaktive Folium-Karte mit Aktivitäten an.
    (Implementierung wie zuvor)
    """
    st.subheader("Karte der Aktivitäten")
    map_center = [ST_GALLEN_LAT, ST_GALLEN_LON]; map_zoom = 11

    required_map_cols = [COL_LAT, COL_LON, COL_ID, COL_NAME]
    if df_for_map.empty or not all(col in df_for_map.columns for col in required_map_cols):
         st.warning("Karte kann nicht angezeigt werden: Daten oder notwendige Spalten fehlen.")
         m = folium.Map(location=map_center, zoom_start=map_zoom, tiles='CartoDB positron')
         st_folium(m, height=500, key="folium_map_empty", use_container_width=True, returned_objects=[])
         return

    valid_coords = df_for_map[[COL_LAT, COL_LON]].dropna()
    if not valid_coords.empty:
        try:
             map_center = [valid_coords[COL_LAT].mean(), valid_coords[COL_LON].mean()]
             num_activities = len(valid_coords)
             if num_activities == 1: map_zoom = 14
             elif num_activities < 10: map_zoom = 12
             else: map_zoom = 11
        except Exception as e: print(f"Warnung: Fehler bei Berechnung des Kartenzentrums/Zooms: {e}")

    if selected_activity_id is not None:
        selected_row = df_for_map[df_for_map[COL_ID] == selected_activity_id]
        if not selected_row.empty:
            try:
                sel_lat = selected_row.iloc[0][COL_LAT]; sel_lon = selected_row.iloc[0][COL_LON]
                if pd.notna(sel_lat) and pd.notna(sel_lon):
                     map_center = [float(sel_lat), float(sel_lon)]; map_zoom = 15
            except Exception as e: print(f"Warnung: Fehler beim Zentrieren auf ID {selected_activity_id}: {e}")

    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles='CartoDB positron')

    for index, row in df_for_map.iterrows():
        lat = row.get(COL_LAT); lon = row.get(COL_LON); activity_id = row.get(COL_ID)
        name = row.get(COL_NAME, 'N/A'); art = row.get(COL_ART, 'N/A'); ort = row.get(COL_ORT, 'N/A')
        weather_note = row.get('weather_note')

        if pd.notna(lat) and pd.notna(lon):
            try:
                is_selected = (activity_id == selected_activity_id)
                icon_color = 'red' if is_selected else 'blue'; icon_symbol = 'star' if is_selected else 'info-sign'
                tooltip_text = f"<b>{name}</b>"; popup_html = f"<b>{name}</b><br>Art: {art}<br>Ort: {ort}"
                if pd.notna(weather_note): tooltip_text += f"<br>{weather_note}"

                folium.Marker(
                    location=[float(lat), float(lon)], popup=folium.Popup(popup_html, max_width=250),
                    tooltip=tooltip_text, icon=folium.Icon(color=icon_color, icon=icon_symbol, prefix='glyphicon')
                ).add_to(m)
            except Exception as e: print(f"Warnung: Fehler beim Hinzufügen des Markers für ID {activity_id} ({name}): {e}")

    st_folium(m, height=500, key="folium_map_main", use_container_width=True, returned_objects=[])


def display_weather_overview(
    location_name: str, target_date: Optional[datetime.date],
    forecast_list: Optional[List[Dict[str, Any]]], api_configured: bool
    ) -> None:
    """
    Zeigt eine Wetterübersicht für einen Ort und Tag an.
    (Implementierung wie zuvor)
    """
    st.subheader(f"Wetterübersicht {location_name}")
    if not target_date: st.markdown("###### Wähle ein Datum"); return
    st.markdown(f"###### {target_date.strftime('%A, %d. %B %Y')}")

    representative_forecast = None
    if forecast_list:
        try:
            valid_forecasts = [f for f in forecast_list if isinstance(f.get('datetime'), datetime.datetime)]
            if valid_forecasts: representative_forecast = next((f for f in valid_forecasts if f['datetime'].hour >= 12), valid_forecasts[0])
        except Exception: pass

    if representative_forecast:
        temp = representative_forecast.get('temp'); desc = representative_forecast.get('description'); icon = representative_forecast.get('icon')
        temp_str = f"{temp:.1f}°C" if pd.notna(temp) else "N/A"; desc_str = desc.capitalize() if pd.notna(desc) else "N/A"
        icon_url = f"http://openweathermap.org/img/wn/{icon}@2x.png" if pd.notna(icon) else None
        subcol1, subcol2 = st.columns([0.3, 0.7])
        with subcol1:
            if icon_url: st.image(icon_url, width=60)
        with subcol2: st.metric(label=desc_str, value=temp_str)
    elif api_configured: st.info(f"Wetterdaten für {location_name} am {target_date.strftime('%d.%m.')} nicht verfügbar.")
    elif location_name: st.caption("Kein OpenWeatherMap API Key konfiguriert.")

    if forecast_list:
        st.markdown(f"###### Tagesverlauf (UTC)")
        try:
            forecast_df_display = pd.DataFrame(forecast_list)
            if not forecast_df_display.empty and 'datetime' in forecast_df_display.columns and 'temp' in forecast_df_display.columns and forecast_df_display['temp'].notna().any():
                forecast_df_display['datetime'] = pd.to_datetime(forecast_df_display['datetime'], errors='coerce', utc=True)
                forecast_df_display.dropna(subset=['datetime', 'temp'], inplace=True)
                if not forecast_df_display.empty:
                    forecast_df_display.set_index('datetime', inplace=True)
                    st.line_chart(forecast_df_display[['temp']], use_container_width=True, height=150)
                    st.caption("Vorhersage-Überblick:")
                    target_hours_utc = [9, 12, 15, 18]; hourly_summary = []; processed_times = set()
                    for hour_utc in target_hours_utc:
                        target_time_utc = datetime.datetime.combine(target_date, datetime.time(hour_utc), tzinfo=datetime.timezone.utc)
                        try:
                            nearest_index_loc = forecast_df_display.index.get_indexer([target_time_utc], method='nearest')[0]
                            closest_forecast_row = forecast_df_display.iloc[nearest_index_loc]
                            actual_time_utc = closest_forecast_row.name; forecast_time_str = actual_time_utc.strftime('%H:%M')
                            if forecast_time_str not in processed_times:
                                temp_val = closest_forecast_row.get('temp'); desc_val = closest_forecast_row.get('description')
                                hourly_summary.append({"Zeit (UTC)": forecast_time_str, "°C": f"{temp_val:.0f}°" if pd.notna(temp_val) else "?", "Wetter": str(desc_val).capitalize() if pd.notna(desc_val) else "N/A"})
                                processed_times.add(forecast_time_str)
                        except Exception: pass
                    summary_df_display = pd.DataFrame(hourly_summary)
                    if not summary_df_display.empty:
                         st.dataframe(summary_df_display, use_container_width=True, hide_index=True, column_config={"Zeit (UTC)": st.column_config.TextColumn("Zeit (UTC)", width="small"), "°C": st.column_config.TextColumn("Temp.", width="small"), "Wetter": st.column_config.TextColumn("Beschreibung", width="large")})
        except Exception as e: st.error(f"Fehler bei Darstellung des Wetter-Tagesverlaufs ({location_name}): {e}")


def display_activity_details(
    activity_row: pd.Series,
    activity_id: int,
    is_expanded: bool,
    openweathermap_api_configured: bool,
    key_prefix: str = ""
    ) -> None:
    """
    Zeigt die Details einer einzelnen Aktivität in einem Streamlit Expander an.

    Stellt formatierte Informationen zur Aktivität dar, inklusive Wetter am Standort,
    Buttons für externe Links und Kartenfokus.

    Args:
        activity_row (pd.Series): Die Datenzeile der anzuzeigenden Aktivität.
        activity_id (int): Die ID der anzuzeigenden Aktivität.
        is_expanded (bool): Ob der Expander standardmäßig geöffnet sein soll.
        openweathermap_api_configured (bool): Flag, ob Wetter-API konfiguriert ist.
        key_prefix (str): Ein Präfix, um eindeutige Widget-Keys zu gewährleisten.
    """
    # --- Datenextraktion ---
    name = activity_row.get(COL_NAME, 'Unbekannt'); art_str = activity_row.get(COL_ART, "Typ N/A"); ort_str = activity_row.get(COL_ORT, "Ort N/A")
    preis_ca_val = activity_row.get(COL_PREIS); beschreibung_str = activity_row.get(COL_BESCHREIBUNG, "Keine Beschreibung verfügbar.")
    image_url = activity_row.get(COL_IMAGE_URL); website_str = activity_row.get(COL_WEBSITE); latitude = activity_row.get(COL_LAT); longitude = activity_row.get(COL_LON)
    dauer_str = activity_row.get(COL_DAUER_INFO, "N/A"); pers_min_val = activity_row.get(COL_PERSONEN_MIN); pers_max_val = activity_row.get(COL_PERSONEN_MAX)
    zielgruppe_str = activity_row.get(COL_ZIELGRUPPE); indoor_outdoor_str = activity_row.get(COL_INDOOR_OUTDOOR, "N/A"); wetter_pref_str = activity_row.get(COL_WETTER_PREF, "N/A")
    von_datum = activity_row.get(COL_DATUM_VON); bis_datum = activity_row.get(COL_DATUM_BIS); booking_str = activity_row.get(COL_BOOKING_INFO); kontakt_str = activity_row.get(COL_KONTAKT_TEL)
    loc_temp_val = activity_row.get('location_temp'); loc_desc_val = activity_row.get('location_desc'); loc_icon_val = activity_row.get('location_icon'); weather_note = activity_row.get('weather_note')

    # --- Formatierungen ---
    # Preis
    if pd.notna(preis_ca_val):
        if preis_ca_val > 0: preis_str = f"{preis_ca_val:.2f} CHF"
        else: preis_str = "Kostenlos"
    else: preis_str = "Preis N/A"
    # Personen Min
    pers_min_str = str(int(pers_min_val)) if pd.notna(pers_min_val) and pers_min_val != float('inf') else "1"
    # Personen Max
    if pd.notna(pers_max_val):
        if pers_max_val == float('inf'): pers_max_str = "∞"
        else:
            try: pers_max_str = str(int(pers_max_val))
            except (ValueError, TypeError): pers_max_str = "?"
    else: pers_max_str = "?"
    personen_str = f"{pers_min_str} - {pers_max_str}"
    # Verfügbarkeit (Datum) - KORRIGIERT
    try:
        von_datum_str = von_datum.strftime('%d.%m.%Y') if pd.notna(von_datum) else None
        bis_datum_str = bis_datum.strftime('%d.%m.%Y') if pd.notna(bis_datum) else None
        # Korrekter if/elif/else Block
        if von_datum_str and bis_datum_str:
            verfuegbar_str = f"{von_datum_str} - {bis_datum_str}"
        elif von_datum_str:
            verfuegbar_str = f"Ab {von_datum_str}"
        elif bis_datum_str:
            verfuegbar_str = f"Bis {bis_datum_str}"
        else:
            verfuegbar_str = "Ganzjährig"
    except Exception:
        verfuegbar_str = "Datumsinfo fehlerhaft"

    # --- Expander ---
    expander_label = f"{name} ({art_str} in {ort_str} | {preis_str})"
    with st.expander(expander_label, expanded=is_expanded):
        # Zeile 1: Wetter & Buttons
        col1, col2 = st.columns([1, 1])
        with col1: # Wetter
            st.markdown("**Wetter am Standort (ca. Mittag):**")
            if pd.notna(loc_temp_val) and pd.notna(loc_desc_val):
                loc_temp_str = f"{loc_temp_val:.1f}°C"; loc_icon_url = f"http://openweathermap.org/img/wn/{loc_icon_val}.png" if pd.notna(loc_icon_val) else None
                w_col1, w_col2 = st.columns([0.2, 0.8]) 
                with w_col1:                             
                    if loc_icon_url: st.image(loc_icon_url, width=40)
                with w_col2:
                    st.write(f"{loc_temp_str}, {loc_desc_val}")
            elif pd.notna(weather_note) and "Koordinaten fehlen" in weather_note: st.caption(weather_note)
            elif openweathermap_api_configured: st.caption("Wetterdaten für Standort nicht verfügbar.")
            if pd.notna(weather_note) and "Koordinaten fehlen" not in weather_note: st.warning(weather_note)
        with col2: # Buttons
             btn_cols = st.columns(3)
             with btn_cols[0]: # Fokus
                 focus_key = f"btn_focus_{key_prefix}_{activity_id}"
                 if st.button("Fokus", key=focus_key, help="Auf Karte fokussieren"): st.session_state[STATE_SELECTED_ACTIVITY_INDEX] = activity_id; st.rerun()
             with btn_cols[1]: # Web
                 if pd.notna(website_str) and isinstance(website_str, str) and website_str.startswith('http'): st.link_button("Web", website_str, help="Website besuchen")
                 else: st.button("Web", disabled=True, help="Keine Website verfügbar", key=f"btn_web_disabled_{key_prefix}_{activity_id}")
             with btn_cols[2]: # Maps
                 if pd.notna(latitude) and pd.notna(longitude):
                     try: maps_url = f"https://www.google.com/maps/search/?api=1&query={float(latitude)},{float(longitude)}"; st.link_button("Maps", maps_url, help="Auf Google Maps anzeigen")
                     except (ValueError, TypeError): st.button("Maps", disabled=True, help="Ungültige Koordinaten", key=f"btn_maps_disabled_invalid_{key_prefix}_{activity_id}")
                 else: st.button("Maps", disabled=True, help="Keine Koordinaten verfügbar", key=f"btn_maps_disabled_missing_{key_prefix}_{activity_id}")
        st.markdown("---")
        # Zeile 2: Bild & Beschreibung
        col3, col4 = st.columns([1, 2])
        with col3: # Bild
            if pd.notna(image_url) and isinstance(image_url, str) and image_url.startswith('http'): st.image(image_url, caption=f"Bild: {name}", use_container_width=True)
            else: st.caption("Kein Bild verfügbar")
        with col4: # Beschreibung
            st.markdown(f"**Beschreibung:**"); st.markdown(beschreibung_str if pd.notna(beschreibung_str) else "_Keine Beschreibung vorhanden._")
        st.markdown("---")
        
        # --- Zeile 3: Weitere Details (strukturierter in 2 Spalten) ---
        detail_col1, detail_col2 = st.columns(2) # Nur noch ZWEI Spalten

        with detail_col1:
            st.markdown(f"**Art:** {art_str}")
            st.markdown(f"**Ort:** {ort_str}")
            # Optional: Adresse hinzufügen, falls vorhanden und gewünscht
            # adresse_str = activity_row.get(config.COL_ADRESSE)
            # if pd.notna(adresse_str): st.markdown(f"**Adresse:** {adresse_str}")
            st.markdown(f"**Typ:** {indoor_outdoor_str}")
            st.markdown(f"**Wetter:** {wetter_pref_str}")
            if pd.notna(zielgruppe_str):
                st.markdown(f"**Zielgruppe:** {zielgruppe_str}")

        with detail_col2:
            st.markdown(f"**Personen:** {personen_str}")
            st.markdown(f"**Dauer ca.:** {dauer_str}")
            st.markdown(f"**Verfügbar:** {verfuegbar_str}")
            if pd.notna(booking_str):
                st.markdown(f"**Buchung:** {booking_str}")
            if pd.notna(kontakt_str):
                st.markdown(f"**Tel:** {kontakt_str}")

        # --- Abschnitt für Ähnliche Aktivitäten wurde entfernt ---


def display_recommendation_card(
    activity_row: pd.Series, card_key_suffix: str,
    on_like_callback: Callable[[int], None], on_dislike_callback: Callable[[int], None]
    ) -> None:
    """
    Zeigt eine einzelne Aktivität als Karte mit Bild, Text und Like/Dislike-Buttons an.
    (Implementierung wie zuvor)
    """
    activity_id = activity_row.get(config.COL_ID); name = activity_row.get(config.COL_NAME, "Unbekannte Aktivität")
    image_url = activity_row.get(config.COL_IMAGE_URL); description = activity_row.get(config.COL_BESCHREIBUNG, "Keine Beschreibung.")
    try: activity_id_for_callback = int(activity_id); button_disabled = False
    except (ValueError, TypeError): print(f"WARNUNG (RecCard): Ungültige ID '{activity_id}'."); activity_id_for_callback = -1; button_disabled = True
    max_desc_length = 100; short_description = description
    if pd.notna(description) and len(description) > max_desc_length: short_description = description[:max_desc_length].strip() + "..."

    with st.container(border=True):
        if pd.notna(image_url) and isinstance(image_url, str) and image_url.startswith('http'): st.image(image_url, use_container_width=True)
        else: st.caption("Kein Bild")
        st.subheader(name); st.markdown(short_description); st.markdown("---")
        col1, col2 = st.columns(2)
        with col1: st.button("👍 Gefällt mir", key=f"btn_like_{card_key_suffix}_{activity_id}", use_container_width=True, on_click=on_like_callback, args=(activity_id_for_callback,), disabled=button_disabled)
        with col2: st.button("👎 Eher nicht", key=f"btn_dislike_{card_key_suffix}_{activity_id}", use_container_width=True, on_click=on_dislike_callback, args=(activity_id_for_callback,), disabled=button_disabled)


def display_preference_visualization(
    profile_label: Optional[str], preference_scores_art: Optional[pd.Series],
    top_target_groups: Optional[pd.Series], liked_prices_list: Optional[List[float]]
    ) -> None:
    """
    Zeigt eine kombinierte Übersicht der gelernten Nutzerpräferenzen.
    (Implementierung wie zuvor)
    """
    if profile_label: st.info(f"Dein gelerntes Profil: **{profile_label}**")
    col1, col2 = st.columns(2)
    with col1: # Kuchendiagramm Arten
        st.markdown("**Bevorzugte Aktivitätsarten:**")
        if preference_scores_art is None or preference_scores_art.empty: st.caption("Bewerte Aktivitäten 👍...")
        else:
            scores_to_display = preference_scores_art[preference_scores_art > 0]
            if scores_to_display.empty: st.caption("Bewerte Aktivitäten 👍...")
            else:
                try:
                    fig_pie = go.Figure(data=[go.Pie(labels=scores_to_display.index, values=scores_to_display.values, hole=.3)])
                    fig_pie.update_layout(margin=dict(l=10, r=10, t=30, b=20), showlegend=False, height=250)
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                except Exception as e: st.error(f"Fehler Kuchendiagramm: {e}"); st.caption("Diagramm nicht erstellt.")
    with col2: # Balkendiagramm Zielgruppen
        st.markdown("**Bevorzugte Zielgruppen:**")
        if top_target_groups is None or top_target_groups.empty: st.caption("Bewerte Aktivitäten...")
        else:
            try: st.bar_chart(top_target_groups, height=230)
            except Exception as e: st.error(f"Fehler Zielgruppen-Chart: {e}"); st.caption("Chart nicht erstellt.")
    # Histogramm Preise
    st.markdown("**Preisverteilung deiner Favoriten:**")
    if liked_prices_list is None or not liked_prices_list: st.caption("Bewerte kostenpflichtige Aktivitäten...")
    else:
        try:
            fig_hist = go.Figure(data=[go.Histogram(x=liked_prices_list, nbinsx=10)])
            fig_hist.update_layout(xaxis_title_text='Preis (CHF)', yaxis_title_text='Anzahl', bargap=0.1, height=250, margin=dict(l=10, r=10, t=10, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e: st.error(f"Fehler Preis-Histogramm: {e}"); st.caption("Histogramm nicht erstellt.")
    st.markdown("---")