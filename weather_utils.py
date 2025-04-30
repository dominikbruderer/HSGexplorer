# weather_utils.py
"""
Hilfsfunktionen für den Abruf und die Bewertung von Wetterdaten von OpenWeatherMap.

Dieses Modul stellt Funktionen zur Verfügung, um:
1. Wettervorhersagen für einen bestimmten Tag und Ort von der OpenWeatherMap API
   abzurufen (`get_weather_forecast_for_day`).
2. Die Eignung des Wetters für eine geplante Aktivität basierend auf der Vorhersage
   einzuschätzen (`check_activity_weather_status`).

Es beinhaltet Caching für API-Antworten zur Performance-Optimierung und robuste
Fehlerbehandlung.
"""

import streamlit as st
import requests
import datetime
import pandas as pd
from typing import List, Dict, Any, Optional # Für Type Hints

# Konstante für den API-Endpunkt
OPENWEATHERMAP_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

@st.cache_data(ttl=3600) # Cache API-Antworten für 1 Stunde
def get_weather_forecast_for_day(
    api_key: str,
    lat: float,
    lon: float,
    target_date: datetime.date | pd.Timestamp
    ) -> Optional[List[Dict[str, Any]]]:
    """
    Ruft die 5-Tage/3-Stunden Wettervorhersage ab und filtert für einen Zieldatum.

    Fragt die OpenWeatherMap API nach der 3-Stunden-Vorhersage für die gegebenen
    Koordinaten ab. Filtert die Daten, um nur Vorhersagepunkte zurückzugeben,
    die auf das `target_date` fallen. Zeitstempel sind in UTC.
    Requirement 2: Nutzt eine externe API (OpenWeatherMap).

    Args:
        api_key (str): Der API-Schlüssel für die OpenWeatherMap API.
        lat (float): Der Breitengrad des gewünschten Ortes.
        lon (float): Der Längengrad des gewünschten Ortes.
        target_date (datetime.date | pd.Timestamp): Das Datum, für das die
            Vorhersage gefiltert werden soll.

    Returns:
        Optional[List[Dict[str, Any]]]: Eine Liste von Dictionaries, wobei jedes
            Dictionary eine 3-Stunden-Wettervorhersage am `target_date` repräsentiert.
            Enthält Schlüssel wie 'datetime' (datetime.datetime UTC), 'time_str' (str),
            'temp' (float), 'main' (str), 'description' (str), 'icon' (str).
            Gibt `None` zurück bei ungültiger Eingabe, API-Fehlern oder wenn keine
            Daten für das Zieldatum gefunden wurden.

    Raises:
        (Intern behandelt) requests.exceptions.RequestException: Bei Netzwerk-/API-Fehlern.
        (Intern behandelt) ValueError: Bei ungültigem Datum oder Zeitstempel.
        (Intern behandelt) Exception: Bei anderen unerwarteten Fehlern.
    """
    # Eingabevalidierung
    if not api_key or pd.isna(lat) or pd.isna(lon) or target_date is None:
        # print("Debug: get_weather_forecast_for_day - Ungültige Eingabe oder kein API Key.")
        return None

    # Sichere Konvertierung des Zieldatums in ein date-Objekt
    try:
        target_date_obj = target_date.date() if hasattr(target_date, 'date') else target_date
        if not isinstance(target_date_obj, datetime.date):
            raise ValueError("target_date ist kein gültiges Datumsobjekt.")
    except Exception as e:
        # print(f"Debug: Fehler bei Datumskonvertierung: {e}")
        return None

    # Parameter für die API-Anfrage
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric', # Temperaturen in Celsius
        'lang': 'de'       # Wetterbeschreibungen auf Deutsch
    }

    try:
        # API-Aufruf mit Timeout
        response = requests.get(OPENWEATHERMAP_FORECAST_URL, params=params, timeout=10)
        # Fehlerprüfung für HTTP-Statuscodes (4xx, 5xx)
        response.raise_for_status()
        forecast_data = response.json()

        # Prüfung des Statuscodes *innerhalb* der JSON-Antwort (OpenWeatherMap-spezifisch)
        if str(forecast_data.get("cod")) != "200":
            # print(f"WARNUNG: Wetter-API-Fehler ({lat},{lon}): {forecast_data.get('message', 'Unbekannt')}")
            return None

        # Verarbeitung der Vorhersageliste
        daily_forecasts: List[Dict[str, Any]] = []
        for forecast in forecast_data.get('list', []):
            try:
                # Zeitstempel (Unix, UTC) sicher in datetime-Objekt (UTC) konvertieren
                forecast_dt_utc = datetime.datetime.fromtimestamp(int(forecast['dt']), tz=datetime.timezone.utc)
            except (TypeError, ValueError, KeyError, OSError) as ts_e:
                # Überspringe Eintrag bei ungültigem Zeitstempel
                # print(f"Debug: Ungültiger Zeitstempel übersprungen: {forecast.get('dt')} ({ts_e})")
                continue

            # Filtern nach Zieldatum
            if forecast_dt_utc.date() == target_date_obj:
                # Extrahiere relevante Wetterinformationen (mit sicheren .get()-Zugriffen)
                weather_info = {
                    'datetime': forecast_dt_utc,
                    'time_str': forecast_dt_utc.strftime('%H:%M'), # Nur zur Anzeige (UTC)
                    'temp': forecast.get('main', {}).get('temp'),
                    'main': forecast.get('weather', [{}])[0].get('main'),
                    'description': forecast.get('weather', [{}])[0].get('description'),
                    'icon': forecast.get('weather', [{}])[0].get('icon')
                }
                # Füge hinzu, wenn wesentliche Informationen vorhanden sind
                if all(k in weather_info and weather_info[k] is not None for k in ['temp', 'description', 'icon']):
                    daily_forecasts.append(weather_info)

        # Gib die gefilterte Liste zurück, oder None wenn leer
        # print(f"Debug: Wetterdaten für {target_date_obj} ({lat},{lon}): {len(daily_forecasts)} Einträge gefunden.")
        return daily_forecasts if daily_forecasts else None

    except requests.exceptions.Timeout:
        # print(f"WARNUNG: Timeout beim Abrufen der Wetterdaten für {lat},{lon}.")
        return None
    except requests.exceptions.RequestException as e:
        # Fängt andere requests-Fehler ab (ConnectionError, HTTPError etc.)
        # print(f"WARNUNG: Netzwerk-/API-Fehler beim Abrufen der Wetterdaten für {lat},{lon}: {e}")
        return None
    except Exception as e:
        # Fängt andere Fehler ab (z.B. JSON-Parsing)
        # print(f"WARNUNG: Allgemeiner Fehler beim Verarbeiten der Wetterdaten für {lat},{lon}: {e}")
        return None

def check_activity_weather_status(activity_forecast_list: Optional[List[Dict[str, Any]]]) -> str:
    """
    Bewertet die allgemeine Wetterlage für eine Aktivität anhand ihrer Tagesvorhersagen.

    Analysiert eine Liste von 3-Stunden-Vorhersagen für einen Tag. Wählt eine
    repräsentative Vorhersage aus (bevorzugt die erste ab 12 Uhr UTC, ansonsten
    die erste verfügbare) und kategorisiert die Wetterlage.

    Logik:
    1. Leichter Regen/Niesel erkannt -> "Uncertain"
    2. Anderer Niederschlag (Regen, Schnee, Gewitter) oder Sturm/Nebel -> "Bad"
    3. Klarer Himmel ('Clear') -> "Good"
    4. Alle anderen Fälle (z.B. Wolken ohne Niederschlag) -> "Uncertain"

    Args:
        activity_forecast_list (Optional[List[Dict[str, Any]]]): Eine Liste von
            Wetter-Vorhersage-Dictionaries (wie von get_weather_forecast_for_day)
            oder None.

    Returns:
        str: Eine der folgenden Bewertungen: "Good", "Bad", "Uncertain", "Unknown".
             "Unknown" wird zurückgegeben, wenn keine gültige Vorhersage
             analysiert werden kann.
    """
    if not activity_forecast_list:
        return "Unknown" # Keine Daten zum Bewerten

    representative_forecast: Optional[Dict[str, Any]] = None
    try:
        # Filtere nach gültigen Einträgen mit datetime Objekt
        valid_forecasts = [f for f in activity_forecast_list if isinstance(f.get('datetime'), datetime.datetime)]
        if not valid_forecasts:
            return "Unknown" # Keine gültigen Einträge in der Liste

        # Finde erste Vorhersage >= 12 Uhr UTC, sonst nimm die erste
        midday_forecast = next((f for f in valid_forecasts if f['datetime'].hour >= 12), None)
        representative_forecast = midday_forecast if midday_forecast else valid_forecasts[0]

    except (IndexError, StopIteration, TypeError, AttributeError, KeyError) as e:
        # print(f"Debug: Fehler bei Auswahl der repräsentativen Wettervorhersage: {e}")
        return "Unknown" # Fehler bei der Auswahl

    if not representative_forecast:
        return "Unknown" # Sollte nicht passieren, aber sicher ist sicher

    # Hole Hauptbedingung ('main') UND Beschreibung ('description') zur Bewertung
    main_condition = str(representative_forecast.get('main', '')).lower()
    description = str(representative_forecast.get('description', '')).lower()

    # --- Bewertungslogik ---
    # 1. Prüfe auf spezifisch LEICHTEN Regen / Niesel --> Uncertain
    light_rain_keywords = ["leichter regen", "niesel", "drizzle", "light rain"]
    if any(keyword in description for keyword in light_rain_keywords) or main_condition == "drizzle":
        # print(f"Debug: Leichter Regen/Niesel erkannt (Main: '{main_condition}', Desc: '{description}') -> Uncertain")
        return "Uncertain"

    # 2. Prüfe auf anderen (stärkeren) Niederschlag oder Sturm/Nebel --> Bad
    bad_weather_keywords = [
        "regen", "schauer", "gewitter", "schnee", "hagel", # Niederschlag (ohne 'niesel')
        "thunderstorm", "squall", "tornado",             # Sturm (main)
        "mist", "smoke", "haze", "dust", "fog", "sand", "ash" # Sicht/Andere (main)
        ]
    # Prüfe Beschreibung UND main condition auf schlechtes Wetter
    if any(keyword in description for keyword in bad_weather_keywords) or \
       main_condition in ["rain", "snow", "thunderstorm", "squall", "tornado",
                          "mist", "smoke", "haze", "dust", "fog", "sand", "ash"]:
        # print(f"Debug: Schlechtes Wetter erkannt (Main: '{main_condition}', Desc: '{description}') -> Bad")
        return "Bad"

    # 3. Prüfe auf eindeutig gutes Wetter (main) --> Good
    if main_condition == "clear":
        return "Good"

    # 4. Alle anderen Bedingungen (z.B. "clouds" ohne Niederschlag) --> Uncertain
    # print(f"Debug: Wetter als 'Uncertain' eingestuft (weder klar noch schlecht erkannt) (Main: '{main_condition}', Desc: '{description}')")
    return "Uncertain"