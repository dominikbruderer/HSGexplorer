# recommender.py
"""
Funktionen für das Empfehlungssystem der HSGexplorer App.

Dieses Modul beinhaltet die Logik zur Generierung personalisierter Empfehlungen
basierend auf Nutzerbewertungen. Es implementiert Techniken des Content-Based Filtering.

Requirement 5: Dieses Modul implementiert einen Kernteil der Machine Learning Anforderung.
Es umfasst:
- Feature Engineering: Umwandlung von Aktivitätsdaten in numerische Vektoren.
- Nutzerprofilierung: Erstellung eines Vektors, der Nutzerpräferenzen repräsentiert.
- Empfehlungsgenerierung: Vorschlagen von Aktivitäten basierend auf Profilähnlichkeit.
- Hilfsfunktionen: Berechnung von Metriken für die Visualisierung der Präferenzen.
"""

import pandas as pd
import numpy as np
import scipy.sparse # Wird ggf. von sklearn intern oder für hstack benötigt
import traceback # Für detaillierte Fehlermeldungen
import random # Für Exploration bei Empfehlungen
from collections import Counter
from typing import Tuple, Optional, List, Dict, Any, Set # Für Type Hints

# ML-Bibliotheken
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity # Wird für Profil-Empfehlungen benötigt

# Importiere Spaltennamen und Konfigurationen
try:
    from config import (
        COL_ID, COL_NAME, COL_ART, COL_ZIELGRUPPE, COL_INDOOR_OUTDOOR,
        COL_PREIS, COL_BESCHREIBUNG
    )
except ImportError:
    print("WARNUNG (recommender.py): config.py nicht gefunden, verwende Fallback-Spaltennamen.")
    COL_ID, COL_NAME = 'ID', 'Name'
    COL_ART, COL_ZIELGRUPPE = 'Art', 'Zielgruppe'
    COL_INDOOR_OUTDOOR, COL_PREIS = 'Indoor_Outdoor', 'Preis_Ca'
    COL_BESCHREIBUNG = 'Beschreibung'


def preprocess_features(df: pd.DataFrame) -> Tuple[None, Optional[np.ndarray]]:
    """
    Bereitet die Aktivitätsdaten für die Ähnlichkeitsberechnung vor (Feature Engineering).

    Wandelt verschiedene Arten von Daten (Text, Kategorien, Zahlen) in eine
    einheitliche numerische Repräsentation (Feature-Matrix) um, die für
    ML-Algorithmen wie die Kosinus-Ähnlichkeit geeignet ist. Diese Matrix
    wird für die Nutzerprofilierung und die profilbasierten Empfehlungen benötigt.

    Verwendete Techniken:
    - TF-IDF: Vektorisiert die Textbeschreibung (`COL_BESCHREIBUNG`).
    - MultiLabelBinarizer: Kodiert die `COL_ZIELGRUPPE`.
    - MinMaxScaler: Skaliert den Preis (`COL_PREIS`).
    - OneHotEncoder: Wandelt `COL_ART`, `COL_INDOOR_OUTDOOR` um.

    Args:
        df (pd.DataFrame): Der DataFrame mit den Rohdaten der Aktivitäten.

    Returns:
        Tuple[None, Optional[np.ndarray]]:
        - None: Platzhalter (Preprocessor-Objekte werden nicht extern benötigt).
        - features (Optional[np.ndarray]): Die resultierende numerische Feature-Matrix.
          None bei Fehlern oder leerem Input.
    """
    if df.empty:
        print("WARNUNG (preprocess): Leerer DataFrame erhalten.")
        return None, None

    df_processed = df.copy()
    final_features_list = [] # Liste zum Sammeln der einzelnen Feature-Arrays

    # --- 1. Text-Features: TF-IDF für Beschreibung ---
    if COL_BESCHREIBUNG in df_processed.columns:
        df_processed[COL_BESCHREIBUNG] = df_processed[COL_BESCHREIBUNG].fillna('').astype(str)
        vectorizer = TfidfVectorizer(
            stop_words='german', max_features=500, ngram_range=(1, 2)
        )
        try:
            tfidf_features = vectorizer.fit_transform(df_processed[COL_BESCHREIBUNG])
            final_features_list.append(tfidf_features.toarray())
        except Exception as e:
            print(f"FEHLER (preprocess): TF-IDF Vektorisierung fehlgeschlagen: {e}")
    else:
        print(f"WARNUNG (preprocess): Spalte '{COL_BESCHREIBUNG}' fehlt für TF-IDF.")

    # --- 2. Multi-Label Kategorien: Zielgruppe ---
    if COL_ZIELGRUPPE in df_processed.columns:
        df_processed[COL_ZIELGRUPPE] = df_processed[COL_ZIELGRUPPE].fillna('').astype(str)
        df_processed[COL_ZIELGRUPPE] = df_processed[COL_ZIELGRUPPE].apply(
            lambda x: [tag.strip() for tag in x.split(',') if tag.strip()]
        )
        if any(df_processed[COL_ZIELGRUPPE]):
            mlb = MultiLabelBinarizer()
            try:
                zielgruppe_features = mlb.fit_transform(df_processed[COL_ZIELGRUPPE])
                final_features_list.append(zielgruppe_features)
            except Exception as e:
                print(f"FEHLER (preprocess): MultiLabelBinarizer für '{COL_ZIELGRUPPE}' fehlgeschlagen: {e}")
    else:
        print(f"WARNUNG (preprocess): Spalte '{COL_ZIELGRUPPE}' fehlt.")

    # --- 3. Numerische und einfache Kategoriale Features ---
    numeric_features = [COL_PREIS]
    categorical_features_ohe = [COL_ART, COL_INDOOR_OUTDOOR]

    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
    categorical_transformer_ohe = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    transformers_list = []
    valid_numeric_features = [col for col in numeric_features if col in df_processed.columns]
    if valid_numeric_features:
         df_processed[valid_numeric_features] = df_processed[valid_numeric_features].fillna(0)
         transformers_list.append(('num', numeric_transformer, valid_numeric_features))

    valid_ohe_features = [col for col in categorical_features_ohe if col in df_processed.columns]
    if valid_ohe_features:
         for col in valid_ohe_features: df_processed[col] = df_processed[col].fillna('Unbekannt')
         transformers_list.append(('cat_ohe', categorical_transformer_ohe, valid_ohe_features))

    if transformers_list:
        preprocessor_ct = ColumnTransformer(transformers=transformers_list, remainder='drop')
        try:
            ct_features = preprocessor_ct.fit_transform(df_processed)
            final_features_list.append(ct_features)
        except Exception as e:
            print(f"FEHLER (preprocess): ColumnTransformer fehlgeschlagen: {e}")

    # --- 4. Kombiniere ALLE extrahierten Feature-Arrays ---
    if not final_features_list:
        print("WARNUNG (preprocess): Keine Features nach Preprocessing verfügbar.")
        return None, None

    try:
        # Nutze hstack für dichte Arrays; für sparse Matrizen wäre scipy.sparse.hstack nötig
        final_features_matrix = np.hstack(final_features_list)
        print(f"INFO (preprocess): Finale kombinierte Features Shape: {final_features_matrix.shape}")
        return None, final_features_matrix
    except Exception as e:
        print(f"FEHLER (preprocess): Beim Kombinieren der Features: {e}")
        return None, None


def calculate_user_profile(
    liked_ids: List[int],
    disliked_ids: List[int], # Aktuell nicht zur Profilbildung genutzt
    features_matrix: Optional[np.ndarray],
    df: pd.DataFrame
    ) -> Optional[np.ndarray]:
    """
    Berechnet den Profilvektor eines Nutzers basierend auf positiv bewerteten Aktivitäten.

    Der Profilvektor repräsentiert die durchschnittlichen Präferenzen des Nutzers,
    berechnet als Mittelwert der Feature-Vektoren der "gelikten" Aktivitäten.

    Requirement 5: ML - Nutzerprofilierung basierend auf Interaktionen.

    Args:
        liked_ids (List[int]): Liste der IDs der positiv bewerteten Aktivitäten.
        disliked_ids (List[int]): Liste der IDs der negativ bewerteten Aktivitäten.
        features_matrix (Optional[np.ndarray]): Die Feature-Matrix aller Aktivitäten.
        df (pd.DataFrame): Der DataFrame mit allen Aktivitäten (für Index-Lookup).

    Returns:
        Optional[np.ndarray]: Der Profilvektor des Nutzers (1D-Array) oder None
                               bei Fehlern oder wenn keine Likes vorhanden sind.
    """
    if features_matrix is None or features_matrix.shape[1] == 0 or df.empty:
        return None
    if not liked_ids:
         return None

    try:
        df[COL_ID] = pd.to_numeric(df[COL_ID], errors='coerce').fillna(-1).astype(int)
        liked_indices = df.index[df[COL_ID].isin(liked_ids)].tolist()
        if not liked_indices: return None

        valid_indices = [idx for idx in liked_indices if idx < features_matrix.shape[0]]
        if not valid_indices: return None

        liked_vectors_np = features_matrix[valid_indices]

    except Exception as e:
        print(f"FEHLER (user_profile): Beim Extrahieren der Like-Vektoren: {e}")
        return None

    try:
        user_profile_vector = np.mean(liked_vectors_np, axis=0)
        return user_profile_vector
    except Exception as e:
        print(f"FEHLER (user_profile): Bei der Berechnung des Mittelwerts: {e}")
        return None


def get_profile_recommendations(
    user_profile: Optional[np.ndarray],
    features_matrix: Optional[np.ndarray],
    df: pd.DataFrame,
    rated_ids: Set[int], # Set aller bereits bewerteten IDs (Likes + Dislikes)
    n: int = 5,
    exploration_rate: float = 0.15
    ) -> List[int]:
    """
    Empfiehlt Aktivitäten basierend auf der Ähnlichkeit zum Nutzerprofil (Content-Based).

    Berechnet die Kosinus-Ähnlichkeit zwischen dem Nutzerprofil und allen Aktivitäten.
    Schlägt die Top-N ähnlichsten, noch nicht bewerteten Aktivitäten vor.
    Beinhaltet optional Exploration und Mischen der Ergebnisse.

    Requirement 5: ML - Generierung personalisierter Empfehlungen.

    Args:
        user_profile (Optional[np.ndarray]): Der Profilvektor des Nutzers.
        features_matrix (Optional[np.ndarray]): Die Feature-Matrix aller Aktivitäten.
        df (pd.DataFrame): Der DataFrame mit allen Aktivitäten.
        rated_ids (Set[int]): Set der IDs aller bereits bewerteten Aktivitäten.
        n (int): Maximale Anzahl der Empfehlungen.
        exploration_rate (float): Wahrscheinlichkeit für eine zufällige Empfehlung.

    Returns:
        List[int]: Eine Liste der IDs der empfohlenen Aktivitäten.
    """
    if user_profile is None or features_matrix is None or features_matrix.shape[1] == 0 or df.empty:
        return []

    if user_profile.ndim == 1:
        user_profile = user_profile.reshape(1, -1)

    try:
        # Requirement 5: ML - Ähnlichkeitsberechnung Profil <-> Items
        profile_similarities = cosine_similarity(user_profile, features_matrix)
        sim_scores_with_indices = list(enumerate(profile_similarities[0]))
    except Exception as e:
         print(f"FEHLER (profile_recs): Bei Berechnung der Profil-Ähnlichkeit: {e}")
         return []

    sim_scores_with_indices.sort(key=lambda x: x[1], reverse=True)

    recommended_ids: List[int] = []
    try:
        df[COL_ID] = pd.to_numeric(df[COL_ID], errors='coerce').fillna(-1).astype(int)
        for index, score in sim_scores_with_indices:
            if len(recommended_ids) >= n: break
            activity_id = df.iloc[index].get(COL_ID)
            if activity_id is not None and activity_id != -1 and activity_id not in rated_ids:
                 recommended_ids.append(int(activity_id))
    except Exception as e:
        print(f"FEHLER (profile_recs): Beim Sammeln der empfohlenen IDs: {e}")
        return []

    # --- Exploration & Mischen ---
    if random.random() < exploration_rate and not df.empty:
        all_ids = df[COL_ID].unique().tolist()
        unrated_ids = [id_ for id_ in all_ids if isinstance(id_, (int, float)) and pd.notna(id_) and int(id_) not in rated_ids and int(id_) != -1]
        valid_unrated_ids_int = [int(id_) for id_ in unrated_ids if isinstance(id_, (int, float)) and pd.notna(id_)] # Sicher konvertieren

        if valid_unrated_ids_int:
            random_id = random.choice(valid_unrated_ids_int)
            if random_id not in recommended_ids:
                 if len(recommended_ids) < n:
                     recommended_ids.insert(0, random_id)
                 elif len(recommended_ids) == n:
                     recommended_ids[-1] = random_id # Ersetze letztes

    random.shuffle(recommended_ids)
    return recommended_ids


# --- Funktionen für Präferenz-Visualisierung ---
# Diese Funktionen bleiben unverändert, da sie weiterhin für die UI benötigt werden.

def calculate_preference_scores(
    liked_ids: List[int],
    df_all_activities: pd.DataFrame
    ) -> Optional[pd.Series]:
    """ Berechnet Scores für Aktivitätsarten basierend auf Likes. """
    # (Implementierung wie zuvor)
    if not liked_ids or df_all_activities.empty: return None
    if COL_ID not in df_all_activities.columns or COL_ART not in df_all_activities.columns: return None
    try:
        valid_liked_ids = [int(i) for i in liked_ids if isinstance(i, (int, float)) and pd.notna(i)]
        if not valid_liked_ids: return None
        liked_activities = df_all_activities[df_all_activities[COL_ID].isin(valid_liked_ids)]
        if liked_activities.empty: return None
        preference_scores = liked_activities[COL_ART].fillna('Unbekannt').value_counts()
        return preference_scores
    except Exception as e:
        print(f"FEHLER (pref_scores): In calculate_preference_scores: {e}")
        return None

def generate_profile_label(preference_scores: Optional[pd.Series]) -> Optional[str]:
    """ Generiert ein Label für das Nutzerprofil basierend auf Top-Arten. """
    # (Implementierung wie zuvor)
    if preference_scores is None or preference_scores.empty: return None
    try:
        sorted_scores = preference_scores.sort_values(ascending=False)
        top_arten = sorted_scores.head(2).index.tolist(); top_scores = sorted_scores.head(2).values.tolist()
        if not top_arten: return None
        label_parts = []
        if len(top_arten) == 1: label_parts.append(f"{top_arten[0]}-Fan")
        elif len(top_arten) == 2:
            art1, art2 = top_arten[0], top_arten[1]; score1, score2 = top_scores[0], top_scores[1]
            if {art1, art2} == {"Sport", "Action"}: label_parts.append("Sport & Action Typ")
            elif {art1, art2} == {"Natur", "Wandern"}: label_parts.append("Naturfreund")
            elif {art1, art2} == {"Genuss", "Shopping"}: label_parts.append("Genuss & Shopping Typ")
            elif score1 > score2 * 1.5: label_parts.append(f"{art1}-Fan (Interesse: {art2})")
            else: label_parts.append(f"{art1} & {art2} Typ")
        final_label = " ".join(label_parts)
        return final_label.strip() if final_label else None
    except Exception as e:
        print(f"FEHLER (profile_label): In generate_profile_label: {e}")
        return None

def calculate_top_target_groups(
    liked_ids: List[int],
    df_all_activities: pd.DataFrame,
    top_n: int = 5
    ) -> Optional[pd.Series]:
    """ Zählt Top-Zielgruppen-Tags in gelikten Aktivitäten. """
    # (Implementierung wie zuvor)
    if not liked_ids or df_all_activities.empty: return None
    if COL_ID not in df_all_activities.columns or COL_ZIELGRUPPE not in df_all_activities.columns: return None
    try:
        valid_liked_ids = [int(i) for i in liked_ids if isinstance(i, (int, float)) and pd.notna(i)]
        if not valid_liked_ids: return None
        liked_activities = df_all_activities[df_all_activities[COL_ID].isin(valid_liked_ids)]
        if liked_activities.empty: return None
        all_tags = []
        zielgruppen_series = liked_activities[COL_ZIELGRUPPE].fillna('')
        for entry in zielgruppen_series:
            tags = [tag.strip() for tag in entry.split(',') if tag.strip()]
            all_tags.extend(tags)
        if not all_tags: return None
        tag_counts = Counter(all_tags)
        top_tags_series = pd.Series(tag_counts).sort_values(ascending=False).head(top_n)
        return top_tags_series
    except Exception as e:
        print(f"FEHLER (top_groups): In calculate_top_target_groups: {e}")
        return None

def get_liked_prices(
    liked_ids: List[int],
    df_all_activities: pd.DataFrame,
    include_free: bool = False
    ) -> Optional[List[float]]:
    """ Extrahiert Preise der gelikten Aktivitäten. """
    # (Implementierung wie zuvor)
    if not liked_ids or df_all_activities.empty: return None
    if COL_ID not in df_all_activities.columns or COL_PREIS not in df_all_activities.columns: return None
    try:
        valid_liked_ids = [int(i) for i in liked_ids if isinstance(i, (int, float)) and pd.notna(i)]
        if not valid_liked_ids: return None
        liked_activities = df_all_activities[df_all_activities[COL_ID].isin(valid_liked_ids)]
        if liked_activities.empty: return None
        prices_numeric = pd.to_numeric(liked_activities[COL_PREIS], errors='coerce')
        if include_free: valid_prices = prices_numeric.dropna().tolist()
        else: valid_prices = prices_numeric[prices_numeric > 0].dropna().tolist()
        return valid_prices if valid_prices else None
    except Exception as e:
        print(f"FEHLER (liked_prices): In get_liked_prices: {e}")
        return None


# Angepasster Testblock ohne Item-zu-Item Empfehlungen
if __name__ == '__main__':
    print("\n--- Testlauf recommender.py (angepasst) ---")
    test_data = {
        COL_ID: [1, 2, 3, 4, 5], COL_NAME: ['Museum A', 'Wanderung B', 'Konzert C', 'Restaurant D', 'Museum E'],
        COL_BESCHREIBUNG: ['Moderne Kunst Ausstellung', 'Schöne Aussicht Berge', 'Klassische Musik live', 'Gutes Essen Wein', 'Alte Meisterwerke Kunst'],
        COL_ART: ['Kultur', 'Natur', 'Kultur', 'Genuss', 'Kultur'],
        COL_ZIELGRUPPE: ['Alle,Kulturinteressierte', 'Wanderer,Naturfreunde', 'Musikliebhaber', 'Paare,Geniesser', 'Kulturinteressierte,Senioren'],
        COL_INDOOR_OUTDOOR: ['Indoor', 'Outdoor', 'Indoor', 'Indoor', 'Indoor'], COL_PREIS: [15, 0, 50, 60, 12]
    }
    df_test = pd.DataFrame(test_data)
    print("Test DataFrame erstellt.")

    print("\nTeste Preprocessing...")
    _, test_features = preprocess_features(df_test)
    if test_features is not None:
        print(f"Feature-Matrix erstellt (Shape: {test_features.shape}).")

        print("\nTeste User Profile & Profil-basierte Empfehlungen...")
        test_liked_ids = [1, 5]; test_disliked_ids = [2]; test_rated_ids = set(test_liked_ids) | set(test_disliked_ids)
        test_user_profile = calculate_user_profile(test_liked_ids, test_disliked_ids, test_features, df_test)
        if test_user_profile is not None:
            print(f"User Profile Vektor berechnet (erste 5 Werte): {np.round(test_user_profile[:5], 2)}")
            recs_profile = get_profile_recommendations(test_user_profile, test_features, df_test, test_rated_ids, n=2, exploration_rate=0) # Exploration für Test deaktiviert
            print(f"Profil-basierte Empfehlungs-IDs (exkl. {test_rated_ids}): {recs_profile}")

        print("\nTeste Visualisierungs-Helfer...")
        pref_scores = calculate_preference_scores(test_liked_ids, df_test); print("Präferenz-Scores (Art):\n", pref_scores)
        profile_label = generate_profile_label(pref_scores); print(f"Generiertes Profil-Label: {profile_label}")
        top_groups = calculate_top_target_groups(test_liked_ids, df_test, top_n=3); print("Top Zielgruppen:\n", top_groups)
        liked_prices = get_liked_prices(test_liked_ids, df_test); print(f"Preise der gelikten Aktivitäten: {liked_prices}")

    else:
        print("Preprocessing fehlgeschlagen, weitere Tests übersprungen.")
    print("--- Testlauf beendet ---")