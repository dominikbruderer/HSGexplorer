# data_utils.py
"""
Hilfsfunktionen zum Laden und Vorverarbeiten der Aktivitätsdaten aus einer CSV-Datei.

Dieses Modul stellt Funktionen bereit, um die Aktivitätsdaten aus einer
CSV-Datei zu laden, sie zu validieren und grundlegend zu bereinigen.
Die Hauptfunktion `load_data` kümmert sich um das Einlesen, die
Konvertierung von Datentypen, das Handling fehlender Werte und Spalten
sowie die Validierung von Koordinaten und eindeutigen IDs.
"""

import pandas as pd
import streamlit as st
import datetime # Wird für Typ-Annotationen und pd.to_datetime benötigt
import os # Wird für os.path.basename in Fehlermeldungen genutzt

# Importiere Konstanten für Spaltennamen und erwartete Spalten aus der Konfigurationsdatei
try:
    from config import (
        EXPECTED_COLUMNS, COL_ID, COL_NAME, COL_BESCHREIBUNG, COL_ART, COL_ORT,
        COL_ADRESSE, COL_LAT, COL_LON, COL_PREIS, COL_PREIS_INFO, COL_WETTER_PREF,
        COL_DATUM_VON, COL_DATUM_BIS, COL_PERSONEN_MIN, COL_PERSONEN_MAX,
        COL_INDOOR_OUTDOOR, COL_ZIELGRUPPE, COL_DAUER_INFO, COL_WEBSITE,
        COL_KONTAKT_TEL, COL_BOOKING_INFO, COL_IMAGE_URL
    )
except ImportError:
    # Fallback, falls config.py nicht gefunden wird (sollte nicht passieren)
    st.error("Fehler: config.py konnte nicht importiert werden. Spaltennamen nicht verfügbar.")
    # Definiere leere Liste, um Absturz zu vermeiden, aber Funktion wird fehlschlagen
    EXPECTED_COLUMNS = []
    # Könnte hier auch st.stop() aufrufen

@st.cache_data # Cache das Ergebnis, um die Datei nicht bei jeder Interaktion neu zu laden
def load_data(filepath: str) -> pd.DataFrame:
    """
    Lädt Aktivitätsdaten aus einer CSV-Datei, validiert und bereinigt sie.

    Liest die angegebene CSV-Datei ein (erwartet Semikolon als Trennzeichen),
    bereinigt Spaltennamen, prüft auf das Vorhandensein erwarteter Spalten
    (gemäß `config.EXPECTED_COLUMNS`), konvertiert Datums- und numerische
    Spalten in die korrekten Typen (ungültige Werte werden zu NaT/NaN),
    füllt fehlende Werte sinnvoll auf (z.B. Preis=0, Personen_Min=1)
    und validiert IDs (müssen eindeutig und vorhanden sein) sowie Koordinaten
    (Zeilen ohne gültige Koordinaten werden entfernt).

    Args:
        filepath (str): Der Pfad zur CSV-Datei (z.B. "aktivitaeten_neu.csv").

    Returns:
        pd.DataFrame: Ein bereinigter Pandas DataFrame mit den Aktivitätsdaten.
            Enthält die in `config.EXPECTED_COLUMNS` definierten Spalten.
            Gibt einen leeren DataFrame mit diesen Spalten zurück, falls die
            Datei nicht gefunden wird, leer ist oder ein anderer Fehler beim
            Laden/Verarbeiten auftritt. Stellt sicher, dass der Index des
            zurückgegebenen DataFrames immer bei 0 beginnt und fortlaufend ist.

    Raises:
        FileNotFoundError: Indirekt über `pd.read_csv`, wenn der `filepath`
            ungültig ist. Wird intern abgefangen.
        pd.errors.EmptyDataError: Wenn die CSV-Datei keine Daten enthält (nur
            Header). Wird intern abgefangen.
        Exception: Andere unerwartete Fehler beim Verarbeiten der Datei. Werden
            intern abgefangen und eine Fehlermeldung via `st.error` ausgegeben.
    """
    # Requirement 2: Lädt Daten aus einer "Datenbank" (CSV-Datei)
    print(f"INFO: Lade Daten aus {filepath}...") # Log-Ausgabe für Debugging
    try:
        # Fange FileNotFoundError spezifisch ab
        try:
            # Lese CSV mit Semikolon-Trennzeichen und UTF-8 Encoding
            df_load = pd.read_csv(filepath, sep=';', encoding='utf-8')
            print(f"INFO: CSV geladen, {len(df_load)} Zeilen gefunden.")
        except FileNotFoundError:
            st.error(f"Datei nicht gefunden: {filepath}")
            # Gib leeren DataFrame mit erwarteten Spalten zurück
            return pd.DataFrame(columns=EXPECTED_COLUMNS)

        # --- Spaltenbereinigung und -validierung ---
        # Bereinige Spaltennamen (Entferne Leerzeichen am Anfang/Ende, BOM falls vorhanden)
        df_load.columns = df_load.columns.str.strip().str.replace('\ufeff', '', regex=True)

        # Prüfe auf erwartete Spalten und füge fehlende hinzu (mit None gefüllt)
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df_load.columns]
        if missing_cols:
            # Verwende basename für eine saubere Fehlermeldung
            filename = os.path.basename(filepath)
            st.warning(
                f"Fehlende Spalten in '{filename}': {', '.join(missing_cols)}. "
                "Diese Spalten werden hinzugefügt und mit 'None' gefüllt."
            )
            for col in missing_cols:
                df_load[col] = None # Füge fehlende Spalte hinzu

        # Wähle nur die erwarteten Spalten aus und behalte ihre Reihenfolge bei
        # Dies stellt sicher, dass der DataFrame immer die gleiche Struktur hat.
        df_load = df_load[EXPECTED_COLUMNS]

        # --- Datenbereinigung und Typkonvertierung ---
        print("INFO: Starte Datenbereinigung und Typkonvertierung...")
        # Konvertiere Datumsspalten (Format TT.MM.JJJJ)
        # `errors='coerce'` setzt ungültige Datumswerte auf NaT (Not a Time)
        for col in [COL_DATUM_VON, COL_DATUM_BIS]:
            if col in df_load.columns: # Sicherstellen, dass Spalte existiert
                 df_load[col] = pd.to_datetime(df_load[col], format='%d.%m.%Y', errors='coerce')

        # Konvertiere numerische Spalten
        # `errors='coerce'` setzt Werte, die nicht in Zahlen konvertiert werden können, auf NaN (Not a Number)
        numeric_cols = [COL_PREIS, COL_LAT, COL_LON, COL_PERSONEN_MIN, COL_PERSONEN_MAX, COL_ID]
        for col in numeric_cols:
            if col in df_load.columns:
                df_load[col] = pd.to_numeric(df_load[col], errors='coerce')

        # --- Sinnvolles Füllen von fehlenden Werten (NaN/NaT) ---
        # Dies sollte NACH der Typkonvertierung erfolgen, um Konvertierungsfehler (NaN) abzudecken.
        # Fülle fehlende numerische Werte
        if COL_PREIS in df_load.columns: df_load[COL_PREIS] = df_load[COL_PREIS].fillna(0) # Fehlender Preis = 0 CHF
        if COL_PERSONEN_MIN in df_load.columns: df_load[COL_PERSONEN_MIN] = df_load[COL_PERSONEN_MIN].fillna(1) # Fehlendes Minimum = 1 Person
        # Verwende float('inf') für unbegrenzte Maximalpersonenanzahl
        if COL_PERSONEN_MAX in df_load.columns: df_load[COL_PERSONEN_MAX] = df_load[COL_PERSONEN_MAX].fillna(float('inf'))

        # Fülle fehlende kategorische/Text-Werte mit sinnvollen Defaults
        if COL_WETTER_PREF in df_load.columns: df_load[COL_WETTER_PREF] = df_load[COL_WETTER_PREF].fillna('Egal')
        if COL_INDOOR_OUTDOOR in df_load.columns: df_load[COL_INDOOR_OUTDOOR] = df_load[COL_INDOOR_OUTDOOR].fillna('Mixed')
        # Weitere Textspalten bei Bedarf mit '' oder 'N/A' füllen:
        # for col in [COL_BESCHREIBUNG, COL_ART, COL_ORT, ...]:
        #     if col in df_load.columns: df_load[col] = df_load[col].fillna('N/A')

        # --- Validierung von Koordinaten und IDs ---
        # Entferne Zeilen ohne gültige Koordinaten (wichtig für Kartenanzeige)
        if COL_LAT in df_load.columns and COL_LON in df_load.columns:
            initial_count = len(df_load)
            # dropna entfernt Zeilen, bei denen *irgendeine* der Spalten in 'subset' NaN ist
            df_load.dropna(subset=[COL_LAT, COL_LON], inplace=True)
            removed_count = initial_count - len(df_load)
            if removed_count > 0:
                print(f"INFO: {removed_count} Zeilen ohne gültige Koordinaten entfernt.")

        # ID Handling: Stelle sicher, dass jede Aktivität eine eindeutige, numerische ID hat.
        # Dies muss nach dem dropna erfolgen, da Zeilen entfernt wurden.
        if COL_ID in df_load.columns:
            # Erneute Konvertierung zu numerisch, falls fillna den Typ beeinflusst hat
            df_load[COL_ID] = pd.to_numeric(df_load[COL_ID], errors='coerce')
            # Prüfe auf NaN ODER Duplikate nach dem dropna
            ids_invalid = df_load[COL_ID].isna().any()
            ids_duplicated = df_load[COL_ID].duplicated().any()

            if ids_invalid or ids_duplicated:
                filename = os.path.basename(filepath)
                warn_msg = f"IDs in '{filename}' sind ungültig/fehlen." if ids_invalid else ""
                if ids_duplicated: warn_msg += " IDs sind nicht eindeutig." if not warn_msg else " und nicht eindeutig."
                warn_msg += " Erstelle neue fortlaufende IDs basierend auf dem Index."
                st.warning(warn_msg)
                # Setze Index zurück (0, 1, 2,...) und weise ihn als neue ID zu
                df_load.reset_index(drop=True, inplace=True)
                df_load[COL_ID] = df_load.index
            else:
                # Wenn IDs gültig und eindeutig sind, konvertiere sicher zu Integer
                df_load[COL_ID] = df_load[COL_ID].astype(int)
        else:
            # Fallback, wenn ID-Spalte komplett fehlt (sollte durch EXPECTED_COLUMNS nicht passieren)
            filename = os.path.basename(filepath)
            st.error(f"Kritischer Fehler: CSV '{filename}' fehlt die benötigte Spalte '{COL_ID}'! Verwende Index als ID.")
            df_load.reset_index(drop=True, inplace=True)
            df_load[COL_ID] = df_load.index # Notfall-ID-Zuweisung

        # Setze den Index final zurück, um sicherzustellen, dass er 0, 1, 2,... ist
        # Dies ist wichtig für spätere Zugriffe über .iloc oder .loc mit Index
        df_final = df_load.reset_index(drop=True)
        print(f"INFO: Datenverarbeitung abgeschlossen. {len(df_final)} Aktivitäten bereit.")
        return df_final

    # Fange leere Dateien ab (nach read_csv)
    except pd.errors.EmptyDataError:
        st.warning(f"Die Datei '{filepath}' ist leer oder enthält nur Header.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS) # Leerer DF mit korrekten Spalten
    # Fange alle anderen unerwarteten Fehler ab
    except Exception as e:
        filename = os.path.basename(filepath)
        st.error(f"Allgemeiner Fehler beim Laden und Verarbeiten der Daten aus {filename}: {e}")
        import traceback
        traceback.print_exc() # Gib detaillierten Traceback in der Konsole aus (für Debugging)
        return pd.DataFrame(columns=EXPECTED_COLUMNS) # Leerer DF mit korrekten Spalten