# recommender.py
"""
Empfehlungssystem: Funktionen für personalisierte Vorschläge.

Dieses Modul enthält die Logik, um Nutzern Aktivitäten vorzuschlagen,
die ihrem Geschmack ähneln könnten, basierend auf dem, was sie zuvor
positiv ("Like" 👍) bewertet haben.

Stell dir vor, die App lernt, was dir gefällt, und nutzt dieses Wissen,
um passende neue Aktivitäten zu finden.

Hauptaufgaben hier drin:
1. Aktivitäten analysieren und "verstehen" (sie in Zahlen übersetzen).
2. Ein "Geschmacksprofil" für den Nutzer erstellen (basierend auf Likes).
3. Aktivitäten finden, die gut zum Geschmacksprofil passen.
4. Helfen, das Gelernte anzuzeigen (z.B. welche Arten du magst).
"""

import pandas as pd
import numpy as np
import random 
from collections import Counter # Zum Zählen von Elementen (z.B. Zielgruppen)
from typing import Tuple, Optional, List, Dict, Any, Set # Nur technische Typ-Hinweise für Entwickler

# Benötigte Werkzeuge aus der scikit-learn Bibliothek für die Datenanalyse/Empfehlungen
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity # Zum Berechnen der Ähnlichkeit

# Lade die Spaltennamen aus unserer Konfigurationsdatei (config.py)
try:
    from config import (
        COL_ID, COL_NAME, COL_ART, COL_ZIELGRUPPE, COL_INDOOR_OUTDOOR,
        COL_PREIS, COL_BESCHREIBUNG
    )
except ImportError:
    print("FATAL: config.py konnte nicht importiert werden (recommender.py).")
    # Notfall-Werte:
    COL_ID, COL_NAME = 'ID', 'Name'; COL_ART, COL_ZIELGRUPPE = 'Art', 'Zielgruppe'
    COL_INDOOR_OUTDOOR, COL_PREIS = 'Indoor_Outdoor', 'Preis_Ca'; COL_BESCHREIBUNG = 'Beschreibung'


def preprocess_features(df: pd.DataFrame) -> Tuple[None, Optional[np.ndarray]]:
    """
    Bereitet die Aktivitätsdaten für den Computer auf, damit er sie vergleichen kann.

    Computer können nicht direkt Text oder Kategorien vergleichen. Diese Funktion
    wandelt alle relevanten Informationen über eine Aktivität (wie Beschreibung,
    Art, Preis, Zielgruppe) in eine Reihe von Zahlen um. Man kann sich das wie
    einen einzigartigen "digitalen Fingerabdruck" für jede Aktivität vorstellen.
    Diese Fingerabdrücke können dann mathematisch verglichen werden, um Ähnlichkeiten
    zu finden.

    Wie die Umwandlung (vereinfacht) funktioniert:
    - **Beschreibungstext:** Analysiert den Text, findet wichtige Schlüsselwörter
      und bewertet deren Wichtigkeit für jede Aktivität. Das Ergebnis sind Zahlen.
    - **Zielgruppen (z.B. "Familie, Kinder"):** Macht aus jeder möglichen Zielgruppe
      eine Art "Ja/Nein"-Frage (Ist diese Aktivität für Familien? Ja/Nein).
    - **Preis:** Bringt alle Preise auf eine einheitliche Skala (z.B. zwischen 0 und 1),
      damit teure Aktivitäten nicht automatisch "unähnlicher" sind.
    - **Art & Indoor/Outdoor (z.B. "Kultur", "Indoor"):** Macht ebenfalls aus jeder
      Möglichkeit eine "Ja/Nein"-Frage (Ist es Kultur? Ja/Nein. Ist es Indoor? Ja/Nein).

    Args:
        df (pd.DataFrame): Die Tabelle mit den ursprünglichen Aktivitätsdaten.

    Returns:
        Tuple[None, Optional[np.ndarray]]:
        - Internes Objekt (None): Wird nicht weiter benötigt.
        - features (Optional[np.ndarray]): Die Tabelle mit den "digitalen Fingerabdrücken".
                                         Jede Zeile ist eine Aktivität, jede Spalte eine
                                         Zahl, die ein Merkmal beschreibt. Gibt None
                                         zurück, wenn etwas schiefgeht.
    """
    if df.empty:
        print("WARNUNG (preprocess): Leere Tabelle zum Vorbereiten erhalten.")
        return None, None

    df_processed = df.copy()
    final_features_list = [] # Hier sammeln wir die einzelnen Zahlen-Teile

    # --- 1. Textbeschreibung analysieren ---
    if COL_BESCHREIBUNG in df_processed.columns:
        df_processed[COL_BESCHREIBUNG] = df_processed[COL_BESCHREIBUNG].fillna('').astype(str)
        # Werkzeug zur Textanalyse initialisieren (ignoriert unwichtige Füllwörter)
        vectorizer = TfidfVectorizer(stop_words=None, max_features=500, ngram_range=(1, 2))
        try:
            # Text in Zahlen umwandeln
            tfidf_features = vectorizer.fit_transform(df_processed[COL_BESCHREIBUNG])
            final_features_list.append(tfidf_features.toarray()) # Als normalen Zahlenteil hinzufügen
        except Exception as e:
            print(f"FEHLER (preprocess): Textanalyse fehlgeschlagen: {e}")
            pass # Mache trotzdem weiter, vielleicht klappen die anderen Teile

    # --- 2. Zielgruppen analysieren (Mehrfachauswahl möglich) ---
    if COL_ZIELGRUPPE in df_processed.columns:
        df_processed[COL_ZIELGRUPPE] = df_processed[COL_ZIELGRUPPE].fillna('').astype(str)
        # Teile "Familie, Kinder" in ["Familie", "Kinder"] auf
        df_processed[COL_ZIELGRUPPE] = df_processed[COL_ZIELGRUPPE].apply(
            lambda x: [tag.strip() for tag in x.split(',') if tag.strip()]
        )
        if any(df_processed[COL_ZIELGRUPPE]):
            # Werkzeug, das für jede Zielgruppe eine Ja/Nein-Spalte erzeugt
            mlb = MultiLabelBinarizer()
            try:
                # Zielgruppen in Ja/Nein-Zahlen (0/1) umwandeln
                zielgruppe_features = mlb.fit_transform(df_processed[COL_ZIELGRUPPE])
                final_features_list.append(zielgruppe_features)
            except Exception as e:
                print(f"FEHLER (preprocess): Zielgruppen-Analyse fehlgeschlagen: {e}")
                pass

    # --- 3. Preis, Aktivitätsart, Indoor/Outdoor analysieren ---
    numeric_features = [COL_PREIS] # Spalten mit einfachen Zahlen
    categorical_features_ohe = [COL_ART, COL_INDOOR_OUTDOOR] # Spalten mit Kategorien (Einfachauswahl)

    # Werkzeuge vorbereiten: Eins zum Skalieren von Zahlen, eins für Kategorien
    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())]) # Bringt Zahlen auf 0-1 Skala
    categorical_transformer_ohe = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]) # Macht Ja/Nein-Spalten

    # Baue einen Plan zusammen, welches Werkzeug auf welche Spalte angewendet wird
    transformers_list = []
    valid_numeric_features = [col for col in numeric_features if col in df_processed.columns]
    if valid_numeric_features:
         df_processed[valid_numeric_features] = df_processed[valid_numeric_features].fillna(0) # Fehlende Preise als 0 behandeln
         transformers_list.append(('num', numeric_transformer, valid_numeric_features))

    valid_ohe_features = [col for col in categorical_features_ohe if col in df_processed.columns]
    if valid_ohe_features:
         for col in valid_ohe_features: df_processed[col] = df_processed[col].fillna('Unbekannt') # Fehlende Kategorien als 'Unbekannt'
         transformers_list.append(('cat_ohe', categorical_transformer_ohe, valid_ohe_features))

    # Wende den Plan an, wenn es was zu tun gibt
    if transformers_list:
        # Führt die Skalierung und Kategorien-Umwandlung durch
        preprocessor_ct = ColumnTransformer(transformers=transformers_list, remainder='drop')
        try:
            ct_features = preprocessor_ct.fit_transform(df_processed)
            final_features_list.append(ct_features)
        except Exception as e:
            print(f"FEHLER (preprocess): Preis/Art/Ort-Analyse fehlgeschlagen: {e}")
            pass

    # --- 4. Alle Zahlen-Teile zum finalen "Fingerabdruck" zusammenfügen ---
    if not final_features_list:
        print("WARNUNG (preprocess): Konnte keine Merkmale extrahieren.")
        return None, None
    try:
        # Hängt alle Zahlenreihen (aus Text, Zielgruppe, Preis etc.) aneinander
        final_features_matrix = np.hstack(final_features_list)
        print(f"INFO (preprocess): 'Digitale Fingerabdrücke' erstellt (Form: {final_features_matrix.shape})")
        return None, final_features_matrix
    except Exception as e:
        print(f"FEHLER (preprocess): Beim Zusammenfügen der Merkmale: {e}")
        return None, None


def calculate_user_profile(
    liked_ids: List[int],
    disliked_ids: List[int], # Info: Dislikes werden momentan nicht fürs Profil genutzt
    features_matrix: Optional[np.ndarray],
    df: pd.DataFrame
    ) -> Optional[np.ndarray]:
    """
    Erstellt ein persönliches "Geschmacksprofil" basierend auf den Likes des Nutzers.

    Das Profil ist quasi der durchschnittliche "digitale Fingerabdruck" aller
    Aktivitäten, die der Nutzer mit "Like" 👍 bewertet hat. Es repräsentiert,
    welche Merkmale (aus Beschreibung, Art, Preis etc.) der Nutzer tendenziell bevorzugt.

    Args:
        liked_ids (List[int]): Die IDs der Aktivitäten, die der Nutzer geliked hat.
        disliked_ids (List[int]): IDs der nicht gemochten Aktivitäten (aktuell ignoriert).
        features_matrix (Optional[np.ndarray]): Die Tabelle mit den "digitalen Fingerabdrücken"
                                             aller Aktivitäten (von `preprocess_features`).
        df (pd.DataFrame): Die ursprüngliche Aktivitätstabelle (um IDs zu finden).

    Returns:
        Optional[np.ndarray]: Der "Geschmacks-Fingerabdruck" des Nutzers als Zahlenreihe,
                               oder None, wenn keine Likes vorhanden sind oder Fehler auftreten.
    """
    if features_matrix is None or features_matrix.shape[1] == 0 or df.empty:
        return None # Geht nicht ohne Fingerabdrücke oder Originaldaten
    if not liked_ids:
         return None # Geht nicht ohne Likes

    try:
        # Finde heraus, welche Zeilen in der Fingerabdruck-Tabelle zu den gelikten IDs gehören
        df[COL_ID] = pd.to_numeric(df[COL_ID], errors='coerce').fillna(-1).astype(int)
        liked_indices = df.index[df[COL_ID].isin(liked_ids)].tolist()
        if not liked_indices: return None
        valid_indices = [idx for idx in liked_indices if idx < features_matrix.shape[0]]
        if not valid_indices: return None
        # Hole die "Fingerabdrücke" der gelikten Aktivitäten
        liked_vectors = features_matrix[valid_indices]
    except Exception as e:
        print(f"FEHLER (user_profile): Konnte Likes nicht den Fingerabdrücken zuordnen: {e}")
        return None

    try:
        # Berechne den Durchschnitt für jedes Merkmal über alle gelikten Fingerabdrücke hinweg.
        # Das Ergebnis ist der durchschnittliche Fingerabdruck = das Geschmacksprofil.
        user_profile_vector = np.mean(liked_vectors, axis=0) # axis=0 heißt "durchschnitt pro Spalte"
        return user_profile_vector
    except Exception as e:
        print(f"FEHLER (user_profile): Konnte Durchschnitts-Geschmack nicht berechnen: {e}")
        return None


def get_profile_recommendations(
    user_profile: Optional[np.ndarray],
    features_matrix: Optional[np.ndarray],
    df: pd.DataFrame,
    rated_ids: Set[int],
    n: int = 5,
    num_exploration_suggestions: int = 0
    ) -> List[int]:
    """
    Findet Aktivitäten, die dem Geschmacksprofil des Nutzers ähneln,
    mit einer Option für gesteuerte Exploration zur Erhöhung der Vielfalt.

    Args:
        user_profile (Optional[np.ndarray]): Der "Geschmacks-Fingerabdruck" des Nutzers.
        features_matrix (Optional[np.ndarray]): Die "digitalen Fingerabdrücke" aller Aktivitäten.
        df (pd.DataFrame): Die ursprüngliche Aktivitätstabelle mit allen Aktivitätsdaten.
        rated_ids (Set[int]): Die IDs der bereits vom Nutzer bewerteten Aktivitäten.
        n (int): Die Gesamtzahl der zu generierenden Empfehlungen.
        num_exploration_suggestions (int): Die Anzahl der Empfehlungen (von n),
                                           die primär auf Exploration (Zufall, Diversität)
                                           statt auf direkter Profilähnlichkeit basieren sollen.

    Returns:
        List[int]: Eine Liste mit den IDs der empfohlenen Aktivitäten,
                   deren Länge maximal n ist. Kann leer sein, wenn keine
                   passenden Aktivitäten gefunden werden.
    """
    # Überprüfung der Eingabeparameter
    if user_profile is None or features_matrix is None or features_matrix.shape[0] == 0 or features_matrix.shape[1] == 0 or df.empty:
        # print("Debug (get_profile_recommendations): Ungültige Eingabe (Profil, Matrix oder df leer).") # Debug
        return []

    # Sicherstellen, dass das Nutzerprofil die korrekte Form für cosine_similarity hat (2D-Array)
    if user_profile.ndim == 1:
        user_profile = user_profile.reshape(1, -1)

    # Berechnung der Ähnlichkeits-Scores zwischen Nutzerprofil und allen Aktivitäten
    try:
        # Kosinus-Ähnlichkeit berechnet, wie ähnlich das Profil jeder Aktivität ist.
        profile_similarities = cosine_similarity(user_profile, features_matrix)
        # Erstelle eine Liste von Tupeln (Index der Aktivität, Ähnlichkeits-Score)
        sim_scores_with_indices = list(enumerate(profile_similarities[0]))
    except Exception as e:
         print(f"FEHLER (get_profile_recommendations): Ähnlichkeitsberechnung fehlgeschlagen: {e}")
         return []

    # Sortiere Aktivitäten nach Ähnlichkeit (höchste zuerst)
    sim_scores_with_indices.sort(key=lambda x: x[1], reverse=True)

    profile_based_ids: List[int] = []
    # Zielanzahl für rein profilbasierte Vorschläge
    num_profile_only_suggestions = n - num_exploration_suggestions

    # Sammle profilbasierte Vorschläge (die ähnlichsten, noch nicht bewerteten Aktivitäten)
    try:
        # Stelle sicher, dass die ID-Spalte numerisch ist für den Abgleich
        if COL_ID not in df.columns:
            print(f"FEHLER (get_profile_recommendations): Spalte '{COL_ID}' nicht im DataFrame.")
            return []
        df[COL_ID] = pd.to_numeric(df[COL_ID], errors='coerce').fillna(-1).astype(int)

        for index, score in sim_scores_with_indices:
            # Stoppe, wenn genug profilbasierte Vorschläge gesammelt wurden
            if len(profile_based_ids) >= num_profile_only_suggestions:
                break
            
            # Hole die ID der Aktivität aus dem DataFrame
            # .iloc[index] greift auf die Zeile an der Position 'index' zu
            activity_id = df.iloc[index].get(COL_ID)

            # Füge ID hinzu, wenn sie gültig ist und noch nicht bewertet wurde
            if activity_id is not None and activity_id != -1 and activity_id not in rated_ids:
                 profile_based_ids.append(int(activity_id))
    except KeyError:
        # Falls .iloc[index] aus irgendeinem Grund fehlschlägt (sollte nicht bei korrekten Indizes)
        print(f"FEHLER (get_profile_recommendations): Indexfehler beim Zugriff auf DataFrame-Zeile.")
         # Fahre mit den bis dahin gesammelten IDs fort oder gib eine leere Liste zurück
    except Exception as e:
        print(f"FEHLER (get_profile_recommendations): Beim Auswählen der profilbasierten Empfehlungen: {e}")
        # Im Fehlerfall könnten hier die bis dahin gesammelten IDs zurückgegeben werden,
        # aber die Funktion fährt fort, um ggf. explorative Vorschläge hinzuzufügen.

    # --- Exploration: Füge eine bestimmte Anzahl explorativer Vorschläge hinzu ---
    exploration_ids: List[int] = []
    # Berechne, wie viele Slots tatsächlich noch durch Exploration gefüllt werden müssen/können,
    # um die Gesamtzahl 'n' zu erreichen.
    actual_exploration_needed = n - len(profile_based_ids)

    if actual_exploration_needed > 0 and not df.empty:
        # Hole alle eindeutigen, gültigen IDs aus dem DataFrame
        all_db_ids = df[COL_ID].dropna().unique().tolist()

        # Kandidaten für Exploration:
        # - Müssen gültige Integer-IDs sein.
        # - Dürfen nicht bereits vom Nutzer bewertet worden sein.
        # - Dürfen nicht bereits in den profilbasierten Vorschlägen enthalten sein.
        candidate_exploration_ids = [
            int(id_val) for id_val in all_db_ids
            if isinstance(id_val, (int, float)) and pd.notna(id_val) and int(id_val) != -1
            and int(id_val) not in rated_ids
            and int(id_val) not in profile_based_ids # Verhindert Duplikate mit Profil-Vorschlägen
        ]
        
        # Mische die Kandidaten für eine zufällige Auswahl
        random.shuffle(candidate_exploration_ids)
        # Wähle die benötigte Anzahl an explorativen IDs aus
        exploration_ids.extend(candidate_exploration_ids[:actual_exploration_needed])

    # Kombiniere die profilbasierten und die explorativen Vorschläge
    final_recommendations = profile_based_ids + exploration_ids
    
    # Mische die finale Liste, damit die explorativen und profilbasierten Vorschläge
    # in einer zufälligen Reihenfolge erscheinen (verhindert, dass Exploration immer am Ende ist).
    random.shuffle(final_recommendations)

    # Stelle sicher, dass nicht mehr als n Empfehlungen zurückgegeben werden.
    return final_recommendations[:n]


# --- Hilfsfunktionen für die Visualisierung ("Was mag ich?") ---
# Diese Funktionen schauen sich die gelikten Aktivitäten an und bereiten
# Informationen auf, die dem Nutzer im Personalisierungs-Bereich angezeigt werden.

def calculate_preference_scores(
    liked_ids: List[int],
    df_all_activities: pd.DataFrame
    ) -> Optional[pd.Series]:
    """ Zählt, wie oft jede Aktivitäts-ART (Kultur, Sport etc.) geliked wurde. """
    if not liked_ids or df_all_activities.empty: return None
    if COL_ID not in df_all_activities.columns or COL_ART not in df_all_activities.columns: return None
    try:
        valid_liked_ids = [int(i) for i in liked_ids if isinstance(i, (int, float)) and pd.notna(i)]
        if not valid_liked_ids: return None
        # Finde alle Aktivitäten, die geliked wurden
        liked_activities = df_all_activities[df_all_activities[COL_ID].isin(valid_liked_ids)]
        if liked_activities.empty: return None
        # Zähle, wie oft jede Art vorkommt
        preference_scores = liked_activities[COL_ART].fillna('Unbekannt').value_counts()
        # Gibt eine Liste zurück, z.B. [("Kultur", 5), ("Natur", 2), ...]
        return preference_scores
    except Exception as e:
        print(f"FEHLER (pref_scores): Konnte Like-Scores (Art) nicht berechnen: {e}")
        return None

def generate_profile_label(preference_scores: Optional[pd.Series]) -> Optional[str]:
    """ Erzeugt einen kurzen Text, der den Nutzertyp beschreibt (z.B. "Kultur-Fan"). """
    if preference_scores is None or preference_scores.empty: return None
    try:
        # Finde die 1-2 am häufigsten gelikten Arten
        sorted_scores = preference_scores.sort_values(ascending=False)
        top_arten = sorted_scores.head(2).index.tolist(); top_scores = sorted_scores.head(2).values.tolist()
        if not top_arten: return None

        # Baue das Label zusammen
        label_parts = []
        if len(top_arten) == 1: label_parts.append(f"{top_arten[0]}-Fan") # Nur eine Art geliked
        elif len(top_arten) == 2: # Zwei Arten geliked
            art1, art2 = top_arten[0], top_arten[1]; score1, score2 = top_scores[0], top_scores[1]
            # Spezielle Namen für bestimmte Kombinationen
            if {art1, art2} == {"Sport", "Action"}: label_parts.append("Sport & Action Typ")
            elif {art1, art2} == {"Natur", "Wandern"}: label_parts.append("Naturfreund")
            elif {art1, art2} == {"Genuss", "Shopping"}: label_parts.append("Genuss & Shopping Typ")
            # Allgemeinere Namen
            elif score1 > score2 * 1.5: label_parts.append(f"{art1}-Fan (Interesse: {art2})") # Eine Art klar vorne
            else: label_parts.append(f"{art1} & {art2} Typ") # Beide ähnlich oft

        final_label = " ".join(label_parts)
        return final_label.strip() if final_label else None
    except Exception as e:
        print(f"FEHLER (profile_label): Konnte Profil-Label nicht erstellen: {e}")
        return None

def calculate_top_target_groups(
    liked_ids: List[int],
    df_all_activities: pd.DataFrame,
    top_n: int = 5
    ) -> Optional[pd.Series]:
    """ Findet die Top-Zielgruppen (z.B. Familie, Paare), die in den Likes vorkommen. """
    if not liked_ids or df_all_activities.empty: return None
    if COL_ID not in df_all_activities.columns or COL_ZIELGRUPPE not in df_all_activities.columns: return None
    try:
        valid_liked_ids = [int(i) for i in liked_ids if isinstance(i, (int, float)) and pd.notna(i)]
        if not valid_liked_ids: return None
        liked_activities = df_all_activities[df_all_activities[COL_ID].isin(valid_liked_ids)]
        if liked_activities.empty: return None

        # Sammle alle einzelnen Zielgruppen-Tags (aus "Familie, Kinder" wird "Familie" und "Kinder")
        all_tags = []
        zielgruppen_series = liked_activities[COL_ZIELGRUPPE].fillna('')
        for entry in zielgruppen_series:
            tags = [tag.strip() for tag in entry.split(',') if tag.strip()]
            all_tags.extend(tags)
        if not all_tags: return None

        # Zähle, wie oft jede Zielgruppe vorkommt und gib die häufigsten zurück
        tag_counts = Counter(all_tags)
        top_tags_series = pd.Series(tag_counts).sort_values(ascending=False).head(top_n)
        # Gibt eine Liste zurück, z.B. [("Familie", 4), ("Kinder", 3), ...]
        return top_tags_series
    except Exception as e:
        print(f"FEHLER (top_groups): Konnte Top-Zielgruppen nicht berechnen: {e}")
        return None

def get_liked_prices(
    liked_ids: List[int],
    df_all_activities: pd.DataFrame,
    include_free: bool = False # Sollen kostenlose Aktivitäten (Preis=0) mitgezählt werden?
    ) -> Optional[List[float]]:
    """ Sammelt die Preise aller Aktivitäten, die der Nutzer geliked hat. """
    if not liked_ids or df_all_activities.empty: return None
    if COL_ID not in df_all_activities.columns or COL_PREIS not in df_all_activities.columns: return None
    try:
        valid_liked_ids = [int(i) for i in liked_ids if isinstance(i, (int, float)) and pd.notna(i)]
        if not valid_liked_ids: return None
        liked_activities = df_all_activities[df_all_activities[COL_ID].isin(valid_liked_ids)]
        if liked_activities.empty: return None

        # Hole die Preise und wandle sie in Zahlen um
        prices_numeric = pd.to_numeric(liked_activities[COL_PREIS], errors='coerce')

        # Filtere die gültigen Preise heraus (und ignoriere ggf. die kostenlosen)
        if include_free:
            valid_prices = prices_numeric.dropna().tolist() # Nimm alle gültigen Preise
        else:
            valid_prices = prices_numeric[prices_numeric > 0].dropna().tolist() # Nur Preise über 0

        return valid_prices if valid_prices else None # Gib die Liste der Preise zurück
    except Exception as e:
        print(f"FEHLER (liked_prices): Konnte Preise der Likes nicht sammeln: {e}")
        return None