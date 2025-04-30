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

## Wichtiger Hinweis: Entwicklung mit KI-Unterstützung (Gemini)

Diese Anwendung wurde im Rahmen des Kurses "Introduction to Computer Science" an der Universität St.Gallen entwickelt. Ein wesentlicher Bestandteil des Entwicklungsprozesses war die **kollaborative Zusammenarbeit mit dem Large Language Model Gemini (Google)**.

Gemini wurde als aktiver Partner in folgenden Bereichen eingesetzt:
* **Code-Generierung:** Erstellung von Code-Snippets und Implementierung von Funktionen basierend auf Beschreibungen.
* **Debugging:** Identifizierung und Korrektur von Fehlern im Code (Syntax- und Logikfehler).
* **Refactoring & Strukturierung:** Vorschläge zur Verbesserung der Code-Struktur, Lesbarkeit und Modularisierung.
* **Dokumentation:** Formulierung von Docstrings, Kommentaren und wesentlichen Teilen dieses READMEs.
* **Konzeptionelle Unterstützung:** Diskussion von Lösungsansätzen und Algorithmen (z.B. für das Empfehlungssystem, Layout-Gestaltung).

Die Entwicklung war ein iterativer Prozess, bei dem Anforderungen und Ideen mit Gemini diskutiert und der von Gemini generierte Code überprüft, angepasst und integriert wurde. **Gemäss den Richtlinien der Universität St.Gallen zum Umgang mit generativer KI wird die massgebliche Unterstützung durch Gemini hiermit transparent gemacht.**

## Einrichtung und Ausführung

**Voraussetzungen:**
* Python (Version 3.9 oder höher empfohlen)
* pip (Python package installer)
