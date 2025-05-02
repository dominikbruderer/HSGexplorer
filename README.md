# HSGexplorer 🗺️

**Dein intelligenter Assistent für Aktivitäten in St. Gallen und Umgebung.**

## Überblick

HSGexplorer ist eine Streamlit-Webanwendung, die entwickelt wurde, um Nutzern bei der Entdeckung und Planung von Freizeitaktivitäten in der Region St. Gallen zu helfen. Die App löst das Problem, aus einer Vielzahl von Möglichkeiten passende Aktivitäten zu finden, indem sie verschiedene Filteroptionen, Wetterinformationen, eine Kartenvisualisierung und KI-gestützte Funktionen kombiniert. (Erfüllt Projektanforderung 1)

Die Anwendung bietet:
* Manuelle Filterung nach Datum, Aktivitätsart, Personenzahl und Budget.
* Optionale Berücksichtigung der aktuellen Wettervorhersage am Aktivitätsort.
* Eine interaktive Karte zur Visualisierung der Aktivitätenstandorte.
* Eine Funktion zur Verarbeitung natürlicher Sprache (NLP), um Aktivitäten basierend auf einer textuellen Beschreibung zu finden (mithilfe von Google Gemini).
* Personalisierte Aktivitätsvorschläge basierend auf Nutzerbewertungen (Likes/Dislikes).
* Visualisierung der gelernten Nutzerpräferenzen.

## Kern-Features

* **Interaktive Karte:** Zeigt gefilterte Aktivitäten oder Vorschläge auf einer Karte an (Folium). (Req. 3)
* **Wetter-Integration:** Ruft Wetterdaten von OpenWeatherMap ab und berücksichtigt sie bei der Filterung. (Req. 2)
* **KI-Suche:** Versteht natürlichsprachliche Anfragen und extrahiert Filter oder schlägt Aktivitäten vor (Google Gemini). (Req. 2 & 5)
* **Personalisierung:** Lernt Nutzerpräferenzen durch Bewertungen und gibt personalisierte Empfehlungen (Content-Based Filtering mit Scikit-learn). (Req. 5)
* **Dynamische Filter:** Bietet verschiedene Filteroptionen zur Eingrenzung der Suche. (Req. 4)
* **Visualisierungen:** Stellt Wetterdaten und Nutzerpräferenzen grafisch dar (Streamlit Charts, Plotly). (Req. 3)

## Wichtiger Hinweis: Entwicklung als Mensch-KI-Kollaboration

Diese Anwendung wurde im Rahmen des Kurses "Grundlagen und Methoden der Informatik für Wirtschaftswissenschaften" an der Universität St.Gallen in einer **engen kollaborativen Partnerschaft zwischen dem menschlichen Entwicklerteam und dem Large Language Model Gemini (Google)** entwickelt. Gemini diente dabei als Werkzeug und unterstützender Partner während des gesamten Entwicklungsprozesses.

Die Unterstützung durch Gemini umfasste insbesondere folgende Bereiche:
* **Code-Generierung:** Erstellung von Code-Bausteinen und Vorschläge für Funktionsimplementierungen basierend auf definierten Anforderungen.
* **Debugging:** Unterstützung bei der Identifizierung und Behebung von Fehlern im Code.
* **Refactoring & Strukturierung:** Ideen und Vorschläge zur Verbesserung der Code-Organisation, Lesbarkeit und Modularität.
* **Dokumentation:** Hilfe bei der Formulierung von Docstrings, Kommentaren und Teilen dieses READMEs.
* **Konzeptionelle Unterstützung:** Diskussion von alternativen Lösungsansätzen, Algorithmen und Design-Entscheidungen.

Der Entwicklungsprozess war iterativ: Anforderungen und Ideen wurden durch das Entwicklerteam formuliert, mit Gemini diskutiert, und von Gemini vorgeschlagener Code wurde stets **kritisch überprüft, angepasst, getestet und durch das Entwicklerteam bewusst integriert**. Gemäss den Richtlinien der Universität St.Gallen zum Umgang mit generativer KI wird diese signifikante Unterstützung durch Gemini hiermit vollständig transparent gemacht. Die Verantwortung für den finalen Code und die Anwendung liegt beim Entwicklerteam.

## Einrichtung und Ausführung

**Voraussetzungen:**
* Python (Version 3.9 oder höher empfohlen)
* pip (Python package installer)
