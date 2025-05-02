# data_utils.py
"""
Dieses Modul enthält die Funktion zum Laden und Aufbereiten der Aktivitätsdaten.

Es stellt sicher, dass die Daten aus der CSV-Datei korrekt eingelesen,
fehlende Werte sinnvoll ersetzt und die Datentypen (z.B. für Datum, Preis)
korrigiert werden. Das ist die Grundlage für alle weiteren Schritte in der App.
"""

import pandas as pd
import streamlit as st
import datetime # Wird für Datums-Verarbeitung benötigt
import os # Wird verwendet, um Dateinamen aus Pfaden zu extrahieren (für Fehlermeldungen)

# Importiere die Namen der erwarteten Spalten aus der Konfigurationsdatei (config.py)
# Das hilft, Tippfehler zu vermeiden und den Code übersichtlich zu halten.
try:
    from config import (
        EXPECTED_COLUMNS, COL_ID, COL_LAT, COL_LON, COL_PREIS,
        COL_DATUM_VON, COL_DATUM_BIS, COL_PERSONEN_MIN, COL_PERSONEN_MAX,
        COL_WETTER_PREF, COL_INDOOR_OUTDOOR
        # Hier könnten bei Bedarf weitere importiert werden, aber die oben genannten
        # werden direkt in dieser Datei für die Bereinigung verwendet.
        # EXPECTED_COLUMNS enthält die vollständige Liste.
    )
except ImportError:
    # Dieser Fehler sollte nicht auftreten, wenn config.py im selben Ordner liegt.
    st.error("Wichtige Konfigurationsdatei (config.py) nicht gefunden!")
    # Definiere leere Liste als Notlösung, um Absturz zu vermeiden.
    EXPECTED_COLUMNS = []

# '@st.cache_data' sorgt dafür, dass die Daten nur einmal geladen werden und
# beim nächsten Mal aus einem Zwischenspeicher kommen. Das macht die App schneller.
@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    """
    Lädt und bereinigt die Aktivitätsdaten aus der angegebenen CSV-Datei.

    Diese Funktion führt mehrere Schritte durch, um die Daten für die App nutzbar zu machen:
    1. Einlesen der CSV-Datei (Trennzeichen ';', Umlaute beachten).
    2. Bereinigen der Spaltennamen (entfernt z.B. Leerzeichen).
    3. Sicherstellen, dass alle erwarteten Spalten vorhanden sind (fügt fehlende hinzu).
    4. Umwandeln von Datumsangaben in ein Datumsformat.
    5. Umwandeln von Zahlen (Preise, Koordinaten etc.) in ein Zahlenformat.
       Texte, die keine Zahlen sind (z.B. "Gratis"), werden dabei zu 'NaN' (Not a Number).
    6. Ersetzen von fehlenden Werten (NaN) durch sinnvolle Standardwerte:
       - Preis: 0 (kostenlos)
       - Mindestpersonen: 1
       - Maximalpersonen: unendlich
       - Wetterpräferenz: 'Egal'
       - Indoor/Outdoor: 'Mixed'
    7. Entfernen von Aktivitäten ohne gültige Koordinaten (Latitude/Longitude),
       da diese nicht auf der Karte angezeigt werden können.
    8. Überprüfen der Aktivitäts-IDs: Sie müssen vorhanden und eindeutig sein.
       Wenn nicht, werden neue, fortlaufende IDs erstellt und eine Warnung angezeigt.
    9. Zurücksetzen des DataFrame-Index für saubere Nummerierung.

    Args:
        filepath (str): Der Dateipfad zur CSV-Datei (z.B. "aktivitaeten_neu.csv").

    Returns:
        pd.DataFrame: Eine Tabelle (DataFrame) mit den aufbereiteten Aktivitätsdaten.
                      Enthält nur die in `config.EXPECTED_COLUMNS` definierten Spalten.
                      Gibt eine leere Tabelle zurück, wenn die Datei nicht lesbar ist.
    """
    try:
        # Schritt 1: CSV-Datei einlesen
        try:
            # Versuche, die Datei zu öffnen und als Tabelle (DataFrame) einzulesen.
            # sep=';' gibt an, dass die Spalten durch Semikolons getrennt sind.
            # encoding='utf-8' stellt sicher, dass Umlaute etc. korrekt gelesen werden.
            df_load = pd.read_csv(filepath, sep=';', encoding='utf-8')
        except FileNotFoundError:
            # Wenn die Datei nicht existiert, zeige eine Fehlermeldung und gib eine leere Tabelle zurück.
            st.error(f"Daten-Datei nicht gefunden unter: {filepath}")
            return pd.DataFrame(columns=EXPECTED_COLUMNS)
        except pd.errors.EmptyDataError:
             # Wenn die Datei leer ist (nur Spaltennamen), zeige eine Warnung und gib eine leere Tabelle zurück.
            st.warning(f"Die Daten-Datei '{filepath}' ist leer.")
            return pd.DataFrame(columns=EXPECTED_COLUMNS)


        # Schritt 2: Spaltennamen bereinigen und auf Vollständigkeit prüfen
        # Entferne mögliche Leerzeichen oder unsichtbare Zeichen (BOM) am Anfang/Ende der Spaltennamen.
        df_load.columns = df_load.columns.str.strip().str.replace('\ufeff', '', regex=True)

        # Prüfe, ob alle Spalten, die wir erwarten (aus config.py), in der Datei vorhanden sind.
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df_load.columns]
        if missing_cols:
            # Wenn Spalten fehlen, zeige eine Warnung an und füge sie als leere Spalten hinzu.
            # Das verhindert Fehler in späteren Schritten.
            filename = os.path.basename(filepath) # Nur Dateiname für die Meldung
            st.warning(
                f"In der Datei '{filename}' fehlen Spalten: {', '.join(missing_cols)}. "
                "Diese werden als leere Spalten hinzugefügt."
            )
            for col in missing_cols:
                df_load[col] = None # Fügt die Spalte mit leeren Werten hinzu

        # Wähle nur die erwarteten Spalten in der definierten Reihenfolge aus.
        # Das sorgt für eine konsistente Datenstruktur.
        df_load = df_load[EXPECTED_COLUMNS]


        # Schritt 3: Datentypen korrekt umwandeln (Text zu Datum, Text zu Zahl)
        # Wandle die Datumsspalten um. Wenn ein Eintrag kein gültiges Datum im Format TT.MM.JJJJ ist,
        # wird er zu 'NaT' (Not a Time), also einem leeren Datumswert.
        for col in [COL_DATUM_VON, COL_DATUM_BIS]:
            if col in df_load.columns:
                 df_load[col] = pd.to_datetime(df_load[col], format='%d.%m.%Y', errors='coerce')

        # Wandle Spalten, die Zahlen enthalten sollten, in Zahlen um.
        # Wenn ein Eintrag keine Zahl ist (z.B. Text), wird er zu 'NaN' (Not a Number),
        # also einem leeren Zahlenwert.
        numeric_cols = [COL_PREIS, COL_LAT, COL_LON, COL_PERSONEN_MIN, COL_PERSONEN_MAX, COL_ID]
        for col in numeric_cols:
            if col in df_load.columns:
                df_load[col] = pd.to_numeric(df_load[col], errors='coerce')


        # Schritt 4: Fehlende oder ungültige Werte (NaN/NaT) durch Standardwerte ersetzen
        # Annahme: Fehlender Preis -> kostenlos (0 CHF)
        if COL_PREIS in df_load.columns: df_load[COL_PREIS] = df_load[COL_PREIS].fillna(0)
        # Annahme: Fehlendes Personen-Minimum -> ab 1 Person
        if COL_PERSONEN_MIN in df_load.columns: df_load[COL_PERSONEN_MIN] = df_load[COL_PERSONEN_MIN].fillna(1)
        # Annahme: Fehlendes Personen-Maximum -> keine Obergrenze (unendlich)
        if COL_PERSONEN_MAX in df_load.columns: df_load[COL_PERSONEN_MAX] = df_load[COL_PERSONEN_MAX].fillna(float('inf'))
        # Annahme: Fehlende Wetterpräferenz -> Wetter egal
        if COL_WETTER_PREF in df_load.columns: df_load[COL_WETTER_PREF] = df_load[COL_WETTER_PREF].fillna('Egal')
        # Annahme: Fehlende Indoor/Outdoor-Angabe -> Gemischt/Unbekannt
        if COL_INDOOR_OUTDOOR in df_load.columns: df_load[COL_INDOOR_OUTDOOR] = df_load[COL_INDOOR_OUTDOOR].fillna('Mixed')


        # Schritt 5: Daten anhand des Inhalts validieren und filtern
        # Entferne Aktivitäten ohne gültige Koordinaten (wichtig für die Karte).
        if COL_LAT in df_load.columns and COL_LON in df_load.columns:
            # Behalte nur Zeilen, bei denen sowohl Latitude als auch Longitude gültige Zahlen sind.
            df_load.dropna(subset=[COL_LAT, COL_LON], inplace=True)

        # Überprüfe die Aktivitäts-IDs auf Gültigkeit (müssen Zahlen sein) und Eindeutigkeit.
        if COL_ID in df_load.columns:
            # Wandle sicherheitshalber nochmal zu Zahl, falls durch fillna etwas geändert wurde.
            df_load[COL_ID] = pd.to_numeric(df_load[COL_ID], errors='coerce')
            # Prüfe, ob es fehlende (NaN) oder doppelte IDs gibt.
            ids_invalid_or_missing = df_load[COL_ID].isna().any()
            ids_duplicated = df_load[COL_ID].dropna().duplicated().any()

            if ids_invalid_or_missing or ids_duplicated:
                # Wenn IDs problematisch sind: Gib eine Warnung aus und erstelle neue IDs.
                filename = os.path.basename(filepath)
                warn_msg = f"Warnung in '{filename}': "
                if ids_invalid_or_missing: warn_msg += "IDs fehlen oder sind ungültig. "
                if ids_duplicated: warn_msg += "IDs sind nicht eindeutig. "
                warn_msg += "Erstelle neue fortlaufende IDs (0, 1, 2,...)."
                st.warning(warn_msg)
                # Nummeriere die Zeilen neu (Index 0, 1, 2,...) und nutze diese Nummern als ID.
                df_load = df_load.reset_index(drop=True) # Wichtig: drop=True entfernt den alten Index
                df_load[COL_ID] = df_load.index
            else:
                # Wenn alle IDs okay sind, wandle sie in ganze Zahlen um (Integer).
                df_load[COL_ID] = df_load[COL_ID].astype(int)
        else:
            # Notfall: Sollte nicht passieren, da COL_ID in EXPECTED_COLUMNS ist.
            filename = os.path.basename(filepath)
            st.error(f"Kritisch: ID-Spalte ('{COL_ID}') fehlt in '{filename}'! Verwende Index als ID.")
            df_load = df_load.reset_index(drop=True)
            df_load[COL_ID] = df_load.index


        # Schritt 6: Finaler Index-Reset und Rückgabe der aufbereiteten Daten
        # Stelle sicher, dass der Index der Tabelle sauber bei 0 beginnt und fortlaufend ist.
        df_final = df_load.reset_index(drop=True)
        return df_final

    # Fange alle anderen möglichen Fehler beim Laden/Verarbeiten ab.
    except Exception as e:
        filename = os.path.basename(filepath)
        st.error(f"Ein unerwarteter Fehler ist beim Verarbeiten der Datei '{filename}' aufgetreten: {e}")
        # Gib zur Fehlersuche die Details in der Konsole aus (nicht für den Endnutzer sichtbar).
        import traceback
        traceback.print_exc()
        # Gib eine leere Tabelle zurück, um den Rest der App nicht abstürzen zu lassen.
        return pd.DataFrame(columns=EXPECTED_COLUMNS)