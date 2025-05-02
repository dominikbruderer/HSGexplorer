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
        # API-Aufruf mit Timeout (verhindert ewiges Warten)
        response = requests.get(OPENWEATHERMAP_FORECAST_URL, params=params, timeout=10)
        # Fehlerprüfung für HTTP-Statuscodes (z.B. 404 Not Found, 500 Server Error)
        response.raise_for_status()
        forecast_data = response.json()

        # Prüfung des Statuscodes *innerhalb* der JSON-Antwort (OpenWeatherMap-spezifisch)
        # Manchmal gibt OWM einen HTTP 200 zurück, aber die Nachricht enthält einen Fehler.
        if str(forecast_data.get("cod")) != "200":
            # print(f"WARNUNG: Wetter-API-Fehler ({lat},{lon}): {forecast_data.get('message', 'Unbekannt')}")
            return None

        # Verarbeitung der Vorhersageliste (das 'list'-Element in der API-Antwort)
        daily_forecasts: List[Dict[str, Any]] = []
        for forecast in forecast_data.get('list', []):
            try:
                # Zeitstempel (Unix, UTC) sicher in datetime-Objekt (UTC) konvertieren
                # OWM liefert die Zeit als 'dt' (Unix Timestamp)
                forecast_dt_utc = datetime.datetime.fromtimestamp(int(forecast['dt']), tz=datetime.timezone.utc)
            except (TypeError, ValueError, KeyError, OSError) as ts_e:
                # Überspringe diesen Vorhersage-Eintrag bei ungültigem Zeitstempel
                # print(f"Debug: Ungültiger Zeitstempel übersprungen: {forecast.get('dt')} ({ts_e})")
                continue

            # Filtern: Behalte nur Vorhersagen für das Zieldatum
            if forecast_dt_utc.date() == target_date_obj:
                # Extrahiere relevante Wetterinformationen
                # Verwende .get() mit Default-Werten ({}, [{}], None), um Fehler bei fehlenden Schlüsseln zu vermeiden.
                weather_info = {
                    'datetime': forecast_dt_utc, # Das eigentliche datetime-Objekt (UTC)
                    'time_str': forecast_dt_utc.strftime('%H:%M'), # Uhrzeit als Text (z.B. "15:00") für Anzeige
                    'temp': forecast.get('main', {}).get('temp'), # Temperatur
                    'main': forecast.get('weather', [{}])[0].get('main'), # Hauptzustand (z.B. 'Clouds', 'Rain')
                    'description': forecast.get('weather', [{}])[0].get('description'), # Detailbeschreibung (z.B. 'leichter Regen')
                    'icon': forecast.get('weather', [{}])[0].get('icon') # Code für das Wettersymbol
                }
                # Füge den Eintrag nur hinzu, wenn wichtige Infos (Temp, Beschreibung, Icon) vorhanden sind.
                if all(k in weather_info and weather_info[k] is not None for k in ['temp', 'description', 'icon']):
                    daily_forecasts.append(weather_info)

        # Gib die gefilterte Liste zurück, oder None wenn keine passenden Einträge gefunden wurden.
        # print(f"Debug: Wetterdaten für {target_date_obj} ({lat},{lon}): {len(daily_forecasts)} Einträge gefunden.")
        return daily_forecasts if daily_forecasts else None

    except requests.exceptions.Timeout:
        # Fehlerbehandlung, wenn die API zu lange für eine Antwort braucht.
        # print(f"WARNUNG: Timeout beim Abrufen der Wetterdaten für {lat},{lon}.")
        return None
    except requests.exceptions.RequestException as e:
        # Fehlerbehandlung für andere Netzwerk-/Verbindungsprobleme (z.B. kein Internet, DNS-Fehler).
        # print(f"WARNUNG: Netzwerk-/API-Fehler beim Abrufen der Wetterdaten für {lat},{lon}: {e}")
        return None
    except Exception as e:
        # Fängt alle anderen möglichen Fehler ab (z.B. Fehler beim Verarbeiten der JSON-Antwort).
        # print(f"WARNUNG: Allgemeiner Fehler beim Verarbeiten der Wetterdaten für {lat},{lon}: {e}")
        return None

def check_activity_weather_status(activity_forecast_list: Optional[List[Dict[str, Any]]]) -> str:
    """
    Bewertet die allgemeine Wetterlage ("Gut", "Schlecht", "Unsicher", "Unbekannt").

    Analysiert eine Liste von 3-Stunden-Vorhersagen für einen Tag (Ergebnis von
    `get_weather_forecast_for_day`). Wählt eine repräsentative Vorhersage aus
    (idealerweise zur Mittagszeit) und kategorisiert das Wetter.

    Logik zur Bewertung:
    1. "Schlecht": Deutlicher Niederschlag (Regen, Schnee, Gewitter), Sturm oder Nebel.
    2. "Unsicher": Nur leichter Regen/Niesel oder unklare Bedingungen (z.B. nur Wolken).
    3. "Gut": Klarer Himmel ('Clear').
    4. "Unbekannt": Wenn keine Daten vorhanden oder ein Fehler auftritt.

    Args:
        activity_forecast_list (Optional[List[Dict[str, Any]]]): Die Liste der
            3-Stunden-Vorhersagen für den Tag oder None.

    Returns:
        str: Eine der Bewertungen: "Good", "Bad", "Uncertain", "Unknown".
    """
    if not activity_forecast_list:
        return "Unknown" # Keine Daten zum Bewerten

    representative_forecast: Optional[Dict[str, Any]] = None
    try:
        # Filtere nach gültigen Einträgen (die ein 'datetime'-Objekt enthalten)
        valid_forecasts = [f for f in activity_forecast_list if isinstance(f.get('datetime'), datetime.datetime)]
        if not valid_forecasts:
            return "Unknown" # Keine verwertbaren Vorhersagen in der Liste

        # Finde die erste Vorhersage ab 12 Uhr UTC. Wenn es keine gibt, nimm die allererste verfügbare.
        # Dies dient dazu, eine möglichst relevante Vorhersage für den Hauptteil des Tages zu wählen.
        midday_forecast = next((f for f in valid_forecasts if f['datetime'].hour >= 12), None)
        representative_forecast = midday_forecast if midday_forecast else valid_forecasts[0]

    except Exception as e:
        # Fehler bei der Auswahl der repräsentativen Vorhersage
        # print(f"Debug: Fehler bei Auswahl der repräsentativen Wettervorhersage: {e}")
        return "Unknown"

    if not representative_forecast:
        return "Unknown" # Sollte nicht passieren, aber zur Sicherheit

    # Hole Hauptbedingung ('main', z.B. "Rain") UND Beschreibung ('description', z.B. "leichter Regen")
    # Beide werden für die Bewertung benötigt, da 'main' allein manchmal nicht ausreicht.
    # Umwandlung in Kleinbuchstaben für einfacheren Vergleich.
    main_condition = str(representative_forecast.get('main', '')).lower()
    description = str(representative_forecast.get('description', '')).lower()

    # --- Bewertungslogik ---

    # 1. Prüfe auf eindeutig schlechtes Wetter (stärkerer Niederschlag, Sturm, Nebel etc.)
    # Definiere Schlüsselwörter, die auf schlechtes Wetter hindeuten.
    bad_weather_keywords = [
        "regen", "schauer", "gewitter", "schnee", "hagel", # Niederschlag (Deutsch)
        "thunderstorm", "squall", "tornado",             # Sturm (Englisch, von OWM 'main')
        "mist", "smoke", "haze", "dust", "fog", "sand", "ash" # Sichtbehinderung etc. (Englisch, von OWM 'main')
    ]
    # Prüfe, ob eines der Schlüsselwörter in der Beschreibung vorkommt ODER
    # ob die 'main' condition einer der schlechten Kategorien entspricht.
    # WICHTIG: "Drizzle" (Niesel) wird hier noch NICHT als "Bad" gewertet.
    if any(keyword in description for keyword in bad_weather_keywords if keyword not in ["niesel", "leichter regen"]) or \
       main_condition in ["rain", "snow", "thunderstorm", "squall", "tornado",
                          "mist", "smoke", "haze", "dust", "fog", "sand", "ash"]:
        # print(f"Debug: Schlechtes Wetter erkannt (Main: '{main_condition}', Desc: '{description}') -> Bad")
        return "Bad"

    # 2. Prüfe auf spezifisch LEICHTEN Regen / Niesel --> Unsicher
    # Wenn es nicht "Bad" ist, aber leichten Regen oder Niesel enthält.
    light_rain_keywords = ["leichter regen", "niesel", "drizzle", "light rain"]
    if any(keyword in description for keyword in light_rain_keywords) or main_condition == "drizzle":
        # print(f"Debug: Leichter Regen/Niesel erkannt (Main: '{main_condition}', Desc: '{description}') -> Uncertain")
        return "Uncertain"

    # 3. Prüfe auf eindeutig gutes Wetter (klarer Himmel) --> Gut
    if main_condition == "clear":
        return "Good"

    # 4. Alle anderen Bedingungen (z.B. nur Wolken ohne Niederschlag) --> Unsicher
    # Wenn es weder eindeutig schlecht, noch klar ist, und auch kein leichter Regen/Niesel erkannt wurde.
    # print(f"Debug: Wetter als 'Uncertain' eingestuft (weder klar noch schlecht erkannt) (Main: '{main_condition}', Desc: '{description}')")
    return "Uncertain"